use super::types::*;
use crate::speaker_diarization::matching_policy as policy;
use anyhow::Result;
use serde_json::{json, Value};
use sqlx::SqlitePool;
use std::collections::{BTreeMap, HashMap};

fn cosine(a: &[f32], b: &[f32]) -> Option<f32> {
    if a.len() != b.len() || a.is_empty() || a.iter().chain(b).any(|v| !v.is_finite()) {
        return None;
    }
    let norm = (a.iter().map(|v| v * v).sum::<f32>() * b.iter().map(|v| v * v).sum::<f32>()).sqrt();
    let score = a.iter().zip(b).map(|(a, b)| a * b).sum::<f32>() / norm;
    (norm > 0. && score.is_finite()).then_some(score)
}
pub fn validate_embedding(e: &SpeakerEmbedding) -> Result<()> {
    if e.vector.is_empty()
        || e.vector.iter().any(|v| !v.is_finite())
        || !e.clean_seconds.is_finite()
        || e.clean_seconds < 3.
        || e.vector.iter().map(|v| v * v).sum::<f32>() < 0.001
    {
        anyhow::bail!("Invalid or insufficient clean embedding evidence");
    }
    if let Some(s) = &e.embedding_space {
        if s.id.is_empty() || s.dimension != e.vector.len() {
            anyhow::bail!("Embedding space/dimension mismatch");
        }
    }
    Ok(())
}
#[derive(sqlx::FromRow)]
struct Profile {
    id: String,
    name: String,
    space_id: String,
    embedding_json: String,
    is_self: bool,
}
pub(crate) struct MatchReport {
    pub candidate_scores: Vec<super::identity_candidates::IdentityCandidate>,
    pub matches: HashMap<String, (String, String)>,
    /// Private run metadata; no vectors or contact details are retained here.
    pub resolved_policy: Value,
}
/// Compatibility API. All decision policy and its resolved context live below.
pub async fn match_profiles_for_meeting(
    pool: &SqlitePool,
    meeting: &str,
    embeddings: &[SpeakerEmbedding],
) -> Result<HashMap<String, (String, String)>> {
    Ok(match_with_policy(pool, meeting, embeddings).await?.matches)
}
pub(crate) async fn match_with_policy(
    pool: &SqlitePool,
    meeting: &str,
    embeddings: &[SpeakerEmbedding],
) -> Result<MatchReport> {
    let expected = crate::speaker_diarization::service::expected_profile_ids(pool, meeting).await;
    let rows:Vec<Profile>=sqlx::query_as("SELECT v.id,v.name,e.space_id,e.embedding_json,v.is_self FROM profile_enrollments e JOIN voice_profiles v ON v.id=e.profile_id WHERE e.status='ready' AND e.embedding_json IS NOT NULL").fetch_all(pool).await?;
    let mut proposals: HashMap<String, Vec<(String, String)>> = HashMap::new();
    let mut contexts = BTreeMap::new();
    let mut candidate_scores = Vec::new();
    for e in embeddings {
        validate_embedding(e)?;
        let Some(space) = &e.embedding_space else {
            continue;
        };
        let comparable: Vec<_> = rows
            .iter()
            .filter(|row| row.space_id == space.id)
            .filter_map(|row| {
                let stored: Vec<f32> = serde_json::from_str(&row.embedding_json).ok()?;
                Some((row, cosine(&e.vector, &stored)?))
            })
            .collect();
        // As in legacy live matching, an alias is established only when the
        // real self profile is loaded in this comparable extraction space.
        let own = comparable
            .iter()
            .find(|(row, _)| row.is_self)
            .map(|(row, _)| *row);
        let mut scores:Vec<_>=comparable.iter().map(|(row,score)| {
            let self_alias=policy::is_self_alias(&row.id,Some(&row.name),own.map(|own|own.id.as_str()),own.map(|own|own.name.as_str()));
            let invited=expected.as_ref().map(|ids|ids.contains(&row.id));
            let threshold=policy::saved_threshold(&space.model,self_alias,invited);
            contexts.insert((space.id.clone(),row.id.clone()),json!({"space_id":space.id,"profile_id":row.id,"is_self":row.is_self,"self_or_same_name_alias":self_alias,"expected_attendee":invited,"threshold":threshold,"baseline":policy::saved_baseline(&space.model)}));
            (*score,threshold,*row)
        }).collect();
        candidate_scores.extend(scores.iter().map(|(score, _, row)| super::identity_candidates::IdentityCandidate {
            speaker: e.speaker.clone(), profile_id: row.id.clone(), space_id: space.id.clone(), score: *score,
        }));
        scores.sort_by(|a, b| a.2.id.cmp(&b.2.id));
        let decision_scores: Vec<_> = scores
            .iter()
            .map(|(score, threshold, _)| (*score, *threshold))
            .collect();
        if let Some(index) = policy::saved_choice(&decision_scores) {
            let row = scores[index].2;
            proposals.entry(e.speaker.clone()).or_default().push((
                row.id.clone(),
                if row.is_self {
                    "Me".into()
                } else {
                    row.name.clone()
                },
            ));
        }
    }
    let matches = proposals
        .into_iter()
        .filter_map(|(speaker, candidates)| {
            let first = candidates.first()?;
            candidates
                .iter()
                .all(|candidate| candidate.0 == first.0)
                .then(|| (speaker, first.clone()))
        })
        .collect();
    let mut expected_ids = expected.map(|ids| ids.into_iter().collect::<Vec<_>>());
    if let Some(ids) = &mut expected_ids {
        ids.sort();
    }
    let mut resolved_policy = policy_snapshot();
    resolved_policy["context_status"] = "resolved".into();
    resolved_policy["expected_profile_ids"] = json!(expected_ids);
    resolved_policy["candidate_contexts"] = json!(contexts.into_values().collect::<Vec<_>>());
    Ok(MatchReport {
        candidate_scores,
        matches,
        resolved_policy,
    })
}

pub(crate) fn policy_snapshot() -> Value {
    json!({
        "version":policy::IDENTIFICATION_POLICY_VERSION,"qualification":"conservative heuristic; held-out calibration required",
        "campplus_baseline":policy::saved_baseline("campplus"),"wespeaker_baseline":policy::saved_baseline("wespeaker"),
        "non_attendee_penalty":policy::NON_ATTENDEE_PENALTY,"ambiguity_margin":policy::AMBIGUITY_MARGIN,
        "ambiguity_scope":"all comparable finite profiles; raw similarity",
        "self_alias_rule":"trimmed case-insensitive name of loaded self in exact space; no source-based boost for normalized mixed audio",
        "context_status":"unavailable for failed or incomplete matching"
    })
}
