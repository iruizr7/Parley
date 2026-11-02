//! Semantic source identity and numerical quality of enrollment contributions.
//! Sample rows remain immutable provenance; repeated references are not votes.
use super::types::SourceInterval;
use anyhow::{bail, Result};
use std::collections::{HashMap, HashSet};

#[derive(Debug, sqlx::FromRow)]
pub(crate) struct EvidenceRow {
    pub id: String,
    pub space_id: String,
    pub source_meeting_id: Option<String>,
    pub audio_sha256: Option<String>,
    pub intervals_json: String,
    pub embedding_json: Option<String>,
    pub clean_seconds: f64,
}
pub(crate) const SAMPLE_COLUMNS: &str =
    "id,space_id,source_meeting_id,audio_sha256,intervals_json,embedding_json,clean_seconds";

/// Exact interval values are used; nearby excerpts are never quantized together.
/// Speaker labels, extraction run ids, JSON layout and vector noise are not part
/// of the audio identity. Unknown recordings remain distinct by sample id.
pub(crate) fn contribution_key(
    space: &str,
    meeting: Option<&str>,
    audio_sha256: Option<&str>,
    intervals: &[SourceInterval],
    unknown_sample_id: &str,
) -> Result<String> {
    if intervals.is_empty() {
        bail!("Evidence has no source intervals");
    }
    let mut spans = Vec::with_capacity(intervals.len());
    for interval in intervals {
        if !interval.start.is_finite()
            || !interval.end.is_finite()
            || interval.start < 0.
            || interval.end <= interval.start
        {
            bail!("Evidence has invalid source intervals");
        }
        let start = if interval.start == 0. {
            0.
        } else {
            interval.start
        };
        spans.push((interval.channel, start.to_bits(), interval.end.to_bits()));
    }
    spans.sort_unstable();
    spans.dedup();
    let source = if let Some(hash) = audio_sha256.map(str::trim).filter(|s| !s.is_empty()) {
        ("sha256", hash.to_ascii_lowercase())
    } else if let Some(meeting) = meeting.filter(|s| !s.is_empty()) {
        ("meeting", meeting.to_owned())
    } else {
        ("unknown_sample", unknown_sample_id.to_owned())
    };
    Ok(serde_json::to_string(&(space, source, spans))?)
}

#[derive(Debug, Clone)]
pub(crate) struct Quality {
    pub contributes: bool,
    pub duplicate_of: Option<String>,
    /// usable, duplicate_reference, invalid_embedding, invalid_intervals,
    /// insufficient_clean_evidence; unknown is reserved for older DTOs.
    pub status: &'static str,
}
#[derive(Default)]
pub(crate) struct Assessment {
    pub centroid: Option<Vec<f32>>,
    pub clean_seconds: f64,
    pub contribution_count: usize,
    pub quality: HashMap<String, Quality>,
}
fn valid_vector(vector: &[f32]) -> bool {
    !vector.is_empty()
        && vector.iter().all(|v| v.is_finite())
        && vector.iter().map(|v| v * v).sum::<f32>().is_finite()
        && vector.iter().map(|v| v * v).sum::<f32>() >= 0.001
}

/// Callers order rows by created_at,id so the retained contribution is stable.
pub(crate) fn assess(rows: &[EvidenceRow], dimension: Option<usize>) -> Assessment {
    let mut result = Assessment::default();
    let mut vectors: Vec<Vec<f32>> = vec![];
    let mut seen: HashMap<String, String> = HashMap::new();
    for row in rows {
        let mut quality = Quality {
            contributes: false,
            duplicate_of: None,
            status: "invalid_embedding",
        };
        let vector = row
            .embedding_json
            .as_deref()
            .and_then(|json| serde_json::from_str::<Vec<f32>>(json).ok());
        if let Some(vector) = vector.filter(|v| {
            valid_vector(v)
                && dimension.is_none_or(|dim| v.len() == dim)
                && vectors.first().is_none_or(|first| first.len() == v.len())
        }) {
            if !row.clean_seconds.is_finite() || row.clean_seconds < 3. {
                quality.status = "insufficient_clean_evidence";
            } else if let Ok(intervals) =
                serde_json::from_str::<Vec<SourceInterval>>(&row.intervals_json)
            {
                match contribution_key(
                    &row.space_id,
                    row.source_meeting_id.as_deref(),
                    row.audio_sha256.as_deref(),
                    &intervals,
                    &row.id,
                ) {
                    Ok(key) => {
                        if let Some(first) = seen.get(&key) {
                            quality.status = "duplicate_reference";
                            quality.duplicate_of = Some(first.clone());
                        } else {
                            seen.insert(key, row.id.clone());
                            quality.status = "usable";
                            quality.contributes = true;
                            result.clean_seconds += row.clean_seconds;
                            vectors.push(vector);
                        }
                    }
                    Err(_) => quality.status = "invalid_intervals",
                }
            } else {
                quality.status = "invalid_intervals";
            }
        }
        result.quality.insert(row.id.clone(), quality);
    }
    // Also label aliases whose vectors are damaged. Removal still addresses
    // their physical audio group rather than allowing a good alias to revive it.
    let mut first_reference: HashMap<String, String> = HashMap::new();
    for row in rows {
        let Ok(intervals) = serde_json::from_str::<Vec<SourceInterval>>(&row.intervals_json) else {
            continue;
        };
        let Ok(key) = contribution_key(
            &row.space_id,
            row.source_meeting_id.as_deref(),
            row.audio_sha256.as_deref(),
            &intervals,
            &row.id,
        ) else {
            continue;
        };
        let first = first_reference
            .entry(key.clone())
            .or_insert_with(|| row.id.clone());
        let owner = seen.get(&key).unwrap_or(first);
        if owner != &row.id {
            if let Some(quality) = result.quality.get_mut(&row.id) {
                quality.duplicate_of = Some(owner.clone());
            }
        }
    }
    result.contribution_count = vectors.len();
    if !vectors.is_empty() {
        let centroid = crate::speaker_diarization::embedding_math::average_and_normalize(&vectors);
        if valid_vector(&centroid) {
            result.centroid = Some(centroid);
        }
    }
    result
}

pub(crate) fn unique_keys(rows: &[EvidenceRow], dimension: Option<usize>) -> HashSet<String> {
    let assessment = assess(rows, dimension);
    rows.iter()
        .filter(|row| {
            assessment
                .quality
                .get(&row.id)
                .is_some_and(|quality| quality.contributes)
        })
        .filter_map(|row| {
            let intervals: Vec<SourceInterval> = serde_json::from_str(&row.intervals_json).ok()?;
            contribution_key(
                &row.space_id,
                row.source_meeting_id.as_deref(),
                row.audio_sha256.as_deref(),
                &intervals,
                &row.id,
            )
            .ok()
        })
        .collect()
}
