//! Re-diarize an existing meeting from its stored audio (#115).
//!
//! #114 cut speaker over-fragmentation for new recordings and
//! retranscriptions, but meetings recorded before it keep their old labels
//! (often 14–28 "Speaker N"s). This re-runs speaker embedding and the
//! offline refinement over the meeting's saved audio, using the transcript's
//! existing segment boundaries — Whisper isn't involved, so the text is
//! untouched and it takes seconds rather than minutes.
//!
//! Lines the user has named are left alone: anything linked to a voice
//! profile, or carrying a name other than a generic placeholder ("Speaker
//! N", "Me", "Speaker"). Only the generic lines are relabelled, and their
//! embeddings still take part in the clustering.

use std::collections::HashSet;
use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::{anyhow, Result};
use serde::Serialize;
use sqlx::SqlitePool;

use crate::events::{EventSinkExt, SharedEventSink};

/// What a re-diarization changed.
#[derive(Debug, Clone, Serialize)]
pub struct RediarizeReport {
    pub meeting_id: String,
    /// Distinct speaker labels before and after.
    pub speakers_before: usize,
    pub speakers_after: usize,
    /// Transcript lines relabelled.
    pub changed: usize,
    /// Lines left alone because the user named them.
    pub kept: usize,
}

static RUNNING: AtomicBool = AtomicBool::new(false);

/// Whether a re-diarization is running.
pub fn is_running() -> bool {
    RUNNING.load(Ordering::SeqCst)
}

/// A placeholder label the diarizer assigned, as opposed to a name the user
/// gave.
pub fn is_generic_label(label: Option<&str>) -> bool {
    let Some(label) = label.map(str::trim).filter(|l| !l.is_empty()) else {
        return true;
    };
    matches!(label, "Speaker" | "Me" | "mic" | "system" | "Unknown")
        || label
            .strip_prefix("Speaker ")
            .is_some_and(|n| !n.is_empty() && n.chars().all(|c| c.is_ascii_digit()))
}

struct RediarizeGuard;
impl Drop for RediarizeGuard {
    fn drop(&mut self) {
        RUNNING.store(false, Ordering::SeqCst);
    }
}
/// Re-detect using the configured speaker package; recognition and confirmed
/// assignments are handled by the same service as the processing dialog.
pub async fn rediarize_meeting(
    sink: &SharedEventSink,
    pool: &SqlitePool,
    meeting_id: &str,
) -> Result<RediarizeReport> {
    if RUNNING.swap(true, Ordering::SeqCst) {
        return Err(anyhow!(
            "Speakers are already being re-detected for another meeting"
        ));
    }
    let _guard = RediarizeGuard;
    let before: Vec<(String, Option<String>, Option<String>)> =
        sqlx::query_as("SELECT id,speaker,voice_profile_id FROM transcripts WHERE meeting_id=?")
            .bind(meeting_id)
            .fetch_all(pool)
            .await?;
    let speakers_before = distinct(before.iter().map(|r| r.1.clone()));
    let kept = before
        .iter()
        .filter(|r| r.2.is_some() || !is_generic_label(r.1.as_deref()))
        .count();
    let config = crate::audio::processing::config::load_config(pool).await?;
    let count = super::service::expected_speaker_count(pool, meeting_id)
        .await
        .unwrap_or(0)
        .min(64) as i32;
    static CANCEL: AtomicBool = AtomicBool::new(false);
    crate::audio::processing::service::redetect_speakers(pool,meeting_id,crate::audio::processing::service::ProcessingOptions{config,language:None,num_speakers:count},&CANCEL,|stage,percent,message|{let _=sink.emit_event("speaker-analysis-progress",&serde_json::json!({"meeting_id":meeting_id,"stage":stage,"percent":percent,"message":message}));}).await?;
    let after: Vec<(String, Option<String>, Option<String>)> =
        sqlx::query_as("SELECT id,speaker,voice_profile_id FROM transcripts WHERE meeting_id=?")
            .bind(meeting_id)
            .fetch_all(pool)
            .await?;
    let speakers_after = distinct(after.iter().map(|r| r.1.clone()));
    let changed = after
        .iter()
        .filter(|r| {
            before
                .iter()
                .any(|old| old.0 == r.0 && (old.1 != r.1 || old.2 != r.2))
        })
        .count();
    let _ = sink.emit_event(
        "speakers-refined",
        &serde_json::json!({"meeting_id":meeting_id,"changed_count":changed}),
    );
    Ok(RediarizeReport {
        meeting_id: meeting_id.into(),
        speakers_before,
        speakers_after,
        changed,
        kept,
    })
}

fn distinct(labels: impl IntoIterator<Item = Option<String>>) -> usize {
    labels
        .into_iter()
        .flatten()
        .map(|l| l.trim().to_string())
        .filter(|l| !l.is_empty())
        .collect::<HashSet<_>>()
        .len()
}

/// Meetings with saved audio and at least one generic speaker label, newest
/// first — the candidates for "re-detect speakers in all meetings".
pub async fn meetings_to_rediarize(pool: &SqlitePool) -> Result<Vec<String>> {
    let rows: Vec<(String, Option<String>)> = sqlx::query_as(
        "SELECT DISTINCT m.id, m.folder_path FROM meetings m
         JOIN transcripts t ON t.meeting_id = m.id
         WHERE COALESCE(m.status, 'completed') != 'recording'
           AND t.voice_profile_id IS NULL
           AND (t.speaker IS NULL OR t.speaker = 'Me' OR t.speaker LIKE 'Speaker%')
         ORDER BY m.created_at DESC",
    )
    .fetch_all(pool)
    .await?;
    Ok(rows
        .into_iter()
        .filter(|(_, folder)| {
            folder.as_deref().is_some_and(|f| {
                crate::audio::retranscription::find_audio_file(std::path::Path::new(f)).is_ok()
            })
        })
        .map(|(id, _)| id)
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generic_labels_are_placeholders_only() {
        for generic in [
            None,
            Some(""),
            Some("Speaker"),
            Some("Speaker 3"),
            Some("Speaker 12"),
            Some("Me"),
            Some("mic"),
        ] {
            assert!(is_generic_label(generic), "{generic:?}");
        }
        for named in [
            Some("Priya Shah"),
            Some("Speaker A"),
            Some("Speakerphone"),
            Some("Meg"),
        ] {
            assert!(!is_generic_label(named), "{named:?}");
        }
    }

    #[test]
    fn distinct_ignores_blank_labels() {
        let labels = [
            Some("A".to_string()),
            Some(" A ".to_string()),
            None,
            Some(String::new()),
            Some("B".into()),
        ];
        assert_eq!(distinct(labels), 2);
    }
}

/// Manual check against a scratch copy of a real data directory (never the
/// live one): re-diarizes each listed meeting and prints the report.
///
/// ```text
/// XDG_DATA_HOME=/scratch/share PARLEY_REDIARIZE_MEETINGS=<id>,<id> \
/// cargo test -p parley-core --release rediarize_scratch_copy -- --ignored --nocapture
/// ```
#[cfg(test)]
mod manual {
    #[tokio::test]
    #[ignore]
    async fn rediarize_scratch_copy() {
        crate::bootstrap::init_paths();
        let dir = crate::paths::app_data_dir().unwrap();
        let live = dirs::home_dir()
            .unwrap()
            .join(".local/share/io.github.hankanman.Parley");
        assert!(
            std::env::var_os("XDG_DATA_HOME").is_some() && dir != live,
            "point XDG_DATA_HOME at a scratch copy"
        );
        let pool = sqlx::SqlitePool::connect(&format!(
            "sqlite://{}",
            dir.join("meeting_minutes.sqlite").display()
        ))
        .await
        .unwrap();
        let sink: crate::events::SharedEventSink = std::sync::Arc::new(crate::events::NullSink);
        let ids = std::env::var("PARLEY_REDIARIZE_MEETINGS").unwrap_or_default();
        let ids: Vec<String> = if ids.is_empty() {
            super::meetings_to_rediarize(&pool).await.unwrap()
        } else {
            ids.split(',').map(str::to_string).collect()
        };
        for id in ids {
            let started = std::time::Instant::now();
            match super::rediarize_meeting(&sink, &pool, &id).await {
                Ok(r) => eprintln!(
                    "{id}: {} -> {} speakers, {} changed, {} kept ({:.1}s)",
                    r.speakers_before,
                    r.speakers_after,
                    r.changed,
                    r.kept,
                    started.elapsed().as_secs_f32()
                ),
                Err(e) => eprintln!("{id}: error: {e}"),
            }
        }
    }
}
