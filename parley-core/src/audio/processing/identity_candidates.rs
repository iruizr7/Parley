//! Saved identity comparisons for review. Scores are cosine similarities, not
//! calibrated probabilities, and refer to a whole detected speaker group.
use super::types::Analysis;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use sqlx::SqlitePool;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IdentityCandidate {
    pub speaker: String,
    pub profile_id: String,
    pub space_id: String,
    pub score: f32,
}

/// Read the latest identification for the current grouping. Older runs without
/// rankings remain unranked; opening a picker must never start inference.
pub async fn for_segment(pool: &SqlitePool, meeting: &str, segment: &str) -> Result<Vec<IdentityCandidate>> {
    let mut tx = pool.begin().await?;
    let row: Option<(String, String, f64, f64)> = sqlx::query_as(
        "SELECT a.analysis_json,s.group_version,t.audio_start_time,t.audio_end_time FROM transcripts t JOIN meeting_processing_state s ON s.meeting_id=t.meeting_id JOIN meeting_transcription_analysis a ON a.meeting_id=t.meeting_id WHERE t.meeting_id=? AND t.id=? AND t.audio_start_time IS NOT NULL AND t.audio_end_time IS NOT NULL",
    ).bind(meeting).bind(segment).fetch_optional(&mut *tx).await?;
    let Some((json, version, start, end)) = row else { return Ok(vec![]); };
    let analysis: Analysis = serde_json::from_str(&json)?;
    let groups: std::collections::BTreeSet<_> = analysis.segments.iter()
        .filter(|s| s.start == start && s.end == end)
        .filter_map(|s| s.speaker.as_deref()).collect();
    if groups.len() != 1 { return Ok(vec![]); }
    let json: Option<String> = sqlx::query_scalar(
        "SELECT result_json FROM identification_runs WHERE meeting_id=? AND group_version=? AND json_type(result_json,'$.suggestions')='array' ORDER BY created_at DESC,id DESC LIMIT 1",
    ).bind(meeting).bind(version).fetch_optional(&mut *tx).await?;
    let Some(json) = json else { return Ok(vec![]); };
    let result: serde_json::Value = serde_json::from_str(&json)?;
    let Some(value) = result.get("candidate_scores") else { return Ok(vec![]); };
    let candidates: Vec<IdentityCandidate> = serde_json::from_value(value.clone())?;
    Ok(single_space_ranking(candidates, groups.first().copied().unwrap()))
}

pub(crate) fn single_space_ranking(candidates: Vec<IdentityCandidate>, speaker: &str) -> Vec<IdentityCandidate> {
    let mut candidates: Vec<_> = candidates.into_iter()
        .filter(|c| c.speaker == speaker && c.score.is_finite()).collect();
    let spaces: std::collections::BTreeSet<_> = candidates.iter().map(|c| c.space_id.as_str()).collect();
    // Similarities from different embedding models are not comparable.
    if spaces.len() != 1 { return vec![]; }
    candidates.sort_by(|a, b| b.score.total_cmp(&a.score).then_with(|| a.profile_id.cmp(&b.profile_id)));
    candidates
}
