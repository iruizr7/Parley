use anyhow::{bail, Result};
use serde::{Deserialize, Serialize};
pub use speaker_core::{EmbeddingSpace, SourceInterval, SpeakerEmbedding};
use sqlx::SqlitePool;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimedSpeaker {
    pub start: f64,
    pub end: f64,
    pub speaker: String,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Overlap {
    pub start: f64,
    pub end: f64,
    pub speakers: Vec<String>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Word {
    pub start: f64,
    pub end: f64,
    pub text: String,
    #[serde(default)]
    pub probability: Option<f64>,
    #[serde(default)]
    pub speaker: Option<String>,
    #[serde(default)]
    pub overlapping_speakers: Vec<String>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Segment {
    pub start: f64,
    pub end: f64,
    pub text: String,
    #[serde(default)]
    pub speaker: Option<String>,
    #[serde(default)]
    pub overlapping_speakers: Vec<String>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Analysis {
    pub version: u32,
    #[serde(default)]
    pub audio_sha256: Option<String>,
    #[serde(default)]
    pub request_id: Option<String>,
    pub duration: f64,
    pub language: Option<String>,
    pub segments: Vec<Segment>,
    pub words: Vec<Word>,
    pub turns: Vec<TimedSpeaker>,
    pub overlaps: Vec<Overlap>,
    pub embeddings: Vec<SpeakerEmbedding>,
    #[serde(default)]
    pub settings: serde_json::Value,
    #[serde(default)]
    pub provenance: serde_json::Value,
    #[serde(default)]
    pub elapsed_seconds: f64,
    #[serde(default)]
    pub recognition_seconds: Option<f64>,
    #[serde(default)]
    pub diarization_seconds: Option<f64>,
    #[serde(default)]
    pub recognition_device: Option<String>,
    #[serde(default)]
    pub diarization_device: Option<String>,
    #[serde(default)]
    pub compute_type: Option<String>,
}

impl Analysis {
    pub fn validate(&self) -> Result<()> {
        if self.version != 1
            || !self.duration.is_finite()
            || self.duration <= 0.
            || !self.elapsed_seconds.is_finite()
            || self.elapsed_seconds < 0.
            || self
                .recognition_seconds
                .is_some_and(|t| !t.is_finite() || t < 0.)
            || self
                .diarization_seconds
                .is_some_and(|t| !t.is_finite() || t < 0.)
        {
            bail!("Invalid analysis duration/version");
        }
        let valid = |s: f64, e: f64| {
            s.is_finite() && e.is_finite() && s >= 0. && e >= s && e <= self.duration + 0.05
        };
        if self.segments.iter().any(|s| !valid(s.start, s.end))
            || self.words.iter().any(|s| !valid(s.start, s.end))
            || self.turns.iter().any(|s| !valid(s.start, s.end))
            || self
                .overlaps
                .iter()
                .any(|s| !valid(s.start, s.end) || s.speakers.len() < 2)
            || self.embeddings.iter().any(|e| {
                e.model_id.is_empty()
                    || e.vector.is_empty()
                    || !e.clean_seconds.is_finite()
                    || e.clean_seconds < 3.
                    || e.vector.iter().any(|v| !v.is_finite())
                    || e.vector.iter().map(|v| v * v).sum::<f32>() < 0.001
            })
        {
            bail!("The offline engine returned invalid timings or embeddings");
        }
        let labels: std::collections::HashSet<_> =
            self.turns.iter().map(|t| t.speaker.as_str()).collect();
        if self.turns.iter().any(|t| t.speaker.trim().is_empty())
            || (!labels.is_empty()
                && (self
                    .segments
                    .iter()
                    .filter_map(|s| s.speaker.as_deref())
                    .any(|s| !labels.contains(s))
                    || self
                        .words
                        .iter()
                        .filter_map(|w| w.speaker.as_deref())
                        .any(|s| !labels.contains(s))
                    || self
                        .overlaps
                        .iter()
                        .flat_map(|o| o.speakers.iter())
                        .any(|s| !labels.contains(s.as_str()))))
        {
            bail!("Worker returned an unknown speaker reference");
        }
        for e in &self.embeddings {
            super::identification::validate_embedding(e)?;
            if e.source_intervals
                .iter()
                .any(|i| !valid(i.start, i.end) || i.end <= i.start || i.speaker != e.speaker)
            {
                bail!("Invalid source evidence intervals");
            }
        }
        Ok(())
    }

    pub fn overlapping_speakers(&self, start: Option<f64>, end: Option<f64>) -> Vec<String> {
        let (Some(start), Some(end)) = (start, end) else {
            return vec![];
        };
        let mut speakers: Vec<_> = self
            .overlaps
            .iter()
            .filter(|o| o.start < end && o.end > start)
            .flat_map(|o| o.speakers.clone())
            .collect();
        speakers.sort();
        speakers.dedup();
        speakers
    }

    pub async fn transcript_segments(
        &self,
        _pool: Option<&SqlitePool>,
    ) -> Result<Vec<crate::audio::common::TranscriptSegment>> {
        // Materialization never guesses identities. Identification has its
        // own immutable run and cannot implicitly train a saved profile.
        Ok(self
            .segments
            .iter()
            .filter(|s| !s.text.trim().is_empty())
            .enumerate()
            .map(|(i, s)| crate::audio::common::TranscriptSegment {
                id: format!("transcript-{}", uuid::Uuid::new_v4()),
                text: s.text.trim().into(),
                timestamp: Some(chrono::Utc::now().to_rfc3339()),
                audio_start_time: Some(s.start),
                audio_end_time: Some(s.end),
                duration: Some(s.end - s.start),
                display_time: None,
                confidence: None,
                sequence_id: Some(i as u64),
                speaker: s.speaker.clone(),
                voice_profile_id: None,
                source: None,
            })
            .collect())
    }
}
