// Retranscription module - allows re-processing stored audio with different settings

#[cfg(test)]
use super::common::create_transcript_segments;
use super::constants::AUDIO_EXTENSIONS;
use crate::events::{EventSink, EventSinkExt, SharedEventSink};
use anyhow::{anyhow, Result};
use log::warn;
use serde::{Deserialize, Serialize};
use sqlx::SqlitePool;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};

/// Global flag to track if retranscription is in progress
pub static RETRANSCRIPTION_IN_PROGRESS: AtomicBool = AtomicBool::new(false);

/// Global flag to signal cancellation
static RETRANSCRIPTION_CANCELLED: AtomicBool = AtomicBool::new(false);

/// RAII guard for RETRANSCRIPTION_IN_PROGRESS flag
/// Ensures flag is cleared even if retranscription panics or returns early
struct RetranscriptionGuard;

impl RetranscriptionGuard {
    /// Create guard and set flag atomically
    fn acquire() -> Result<Self, String> {
        if RETRANSCRIPTION_IN_PROGRESS
            .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
            .is_err()
        {
            return Err("Retranscription already in progress".to_string());
        }
        Ok(RetranscriptionGuard)
    }
}

impl Drop for RetranscriptionGuard {
    fn drop(&mut self) {
        RETRANSCRIPTION_IN_PROGRESS.store(false, Ordering::SeqCst);
    }
}

/// Progress update emitted during retranscription
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetranscriptionProgress {
    pub meeting_id: String,
    pub stage: String, // "decoding", "transcribing", "saving"
    pub progress_percentage: u32,
    pub message: String,
}

/// Result of retranscription
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetranscriptionResult {
    pub meeting_id: String,
    pub segments_count: usize,
    pub duration_seconds: f64,
    pub language: Option<String>,
}

/// Error during retranscription
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetranscriptionError {
    pub meeting_id: String,
    pub error: String,
}

/// Check if retranscription is currently in progress
pub fn is_retranscription_in_progress() -> bool {
    RETRANSCRIPTION_IN_PROGRESS.load(Ordering::SeqCst)
}

/// Cancel ongoing retranscription
pub fn cancel_retranscription() {
    RETRANSCRIPTION_CANCELLED.store(true, Ordering::SeqCst);
}

/// Compatibility wrapper that preserves explicit overrides and delegates all
/// inference, identity policy and publication to the typed pipeline service.
pub async fn start_retranscription_with(
    sink: SharedEventSink,
    pool: Option<SqlitePool>,
    meeting_id: String,
    meeting_folder_path: String,
    language: Option<String>,
    model: Option<String>,
    provider: Option<String>,
) -> Result<RetranscriptionResult> {
    start_processing_with(
        sink,
        pool,
        meeting_id,
        meeting_folder_path,
        language,
        model,
        provider,
        0,
    )
    .await
}
pub async fn start_processing_with(
    sink: SharedEventSink,
    pool: Option<SqlitePool>,
    meeting_id: String,
    _meeting_folder_path: String,
    language: Option<String>,
    model: Option<String>,
    provider: Option<String>,
    num_speakers: i32,
) -> Result<RetranscriptionResult> {
    let pool = pool.ok_or_else(|| anyhow!("Database unavailable"))?;
    if provider
        .as_deref()
        .is_some_and(|p| !["localWhisper", "whisper"].contains(&p))
    {
        return Err(anyhow!(
            "Select a supported speech engine in the processing dialog"
        ));
    }
    let mut config = super::processing::config::load_config(&pool).await?;
    if let Some(model) = model {
        config.speech_model = model;
    }
    let outcome = start_pipeline_processing(
        sink,
        pool.clone(),
        meeting_id.clone(),
        super::processing::service::ProcessingOptions {
            config,
            language: language.clone(),
            num_speakers,
        },
    )
    .await?;
    let analysis = super::processing::runs::load(&pool, &meeting_id)
        .await?
        .ok_or_else(|| anyhow!("Successful processing result unavailable"))?;
    Ok(RetranscriptionResult {
        meeting_id,
        segments_count: outcome.segments_count,
        duration_seconds: analysis.duration,
        language: analysis.language.or(language),
    })
}

/// Legacy recording completion entry point. It snapshots the chosen defaults
/// and never selects another model based on a hidden accuracy ranking.
pub fn spawn_auto_refine(
    sink: SharedEventSink,
    pool: Option<SqlitePool>,
    meeting_id: String,
    _meeting_folder_path: String,
) {
    tokio::spawn(async move {
        if let Some(pool) = &pool {
            let enabled =
                super::recording_preferences::load_recording_preferences(Some(pool.clone()))
                    .await
                    .map(|p| p.auto_refine)
                    .unwrap_or(false);
            if enabled {
                match super::processing::config::load_config(pool).await {
                    Ok(config) => {
                        if let Err(error) = start_pipeline_processing(
                            sink.clone(),
                            pool.clone(),
                            meeting_id.clone(),
                            super::processing::service::ProcessingOptions {
                                config,
                                language: None,
                                num_speakers: 0,
                            },
                        )
                        .await
                        {
                            warn!(
                                "Final processing failed; saved preview remains current: {error}"
                            );
                        }
                    }
                    Err(error) => {
                        let _ = sink.emit_event(
                            "retranscription-error",
                            &serde_json::json!({"meeting_id":meeting_id,"error":error.to_string()}),
                        );
                    }
                }
            }
        }
        crate::summary::auto_summary::maybe_auto_summarize(
            sink.clone(),
            pool.clone(),
            meeting_id.clone(),
        )
        .await;
        crate::chapters::maybe_auto_chapter(sink, pool, meeting_id).await;
    });
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetranscriptionStarted {
    pub meeting_id: String,
    pub message: String,
}

/// Find audio file in meeting folder
/// Tries common names first, then scans for any file with an audio extension
pub(crate) fn find_audio_file(folder: &Path) -> Result<PathBuf> {
    let candidates = [
        "audio.mp4",
        "audio.m4a",
        "audio.wav",
        "audio.mp3",
        "audio.flac",
        "audio.ogg",
        "recording.mp4",
        "audio.mkv",
        "audio.webm",
        "audio.wma",
    ];

    for name in candidates {
        let path = folder.join(name);
        if path.is_file() {
            return Ok(path);
        }
    }

    // Fallback: scan folder for any file with an audio extension
    if let Ok(entries) = std::fs::read_dir(folder) {
        for entry in entries.flatten() {
            let path = entry.path();
            if !path.is_file() {
                continue;
            }
            if let Some(ext) = path.extension() {
                let ext = ext.to_string_lossy().to_lowercase();
                if AUDIO_EXTENSIONS.contains(&ext.as_str()) {
                    return Ok(path);
                }
            }
        }
    }

    Err(anyhow!("No audio file found in: {}", folder.display()))
}

/// Emit a `retranscription-progress` event through `sink`.
fn emit_progress(
    sink: &dyn EventSink,
    meeting_id: &str,
    stage: &str,
    progress: u32,
    message: &str,
) {
    let _ = sink.emit_event(
        "retranscription-progress",
        &RetranscriptionProgress {
            meeting_id: meeting_id.to_string(),
            stage: stage.to_string(),
            progress_percentage: progress,
            message: message.to_string(),
        },
    );
}

#[cfg(test)]
mod tests {
    use super::super::common::BatchTranscript;
    use super::*;

    fn bt(text: &str, start_ms: f64, end_ms: f64) -> BatchTranscript {
        BatchTranscript {
            words: vec![],
            text: text.to_string(),
            start_ms,
            end_ms,
            speaker: None,
            voice_profile_id: None,
            sequence_id: 0,
            source: Default::default(),
        }
    }

    #[test]
    fn test_create_transcript_segments_empty() {
        let transcripts: Vec<BatchTranscript> = vec![];
        let segments = create_transcript_segments(&transcripts);
        assert!(segments.is_empty());
    }

    #[test]
    fn test_create_transcript_segments_single() {
        let transcripts = vec![bt("Hello world", 0.0, 1500.0)]; // 0-1.5 seconds
        let segments = create_transcript_segments(&transcripts);

        assert_eq!(segments.len(), 1);
        assert_eq!(segments[0].text, "Hello world");
        assert_eq!(segments[0].audio_start_time, Some(0.0));
        assert_eq!(segments[0].audio_end_time, Some(1.5));
        assert_eq!(segments[0].duration, Some(1.5));
    }

    #[test]
    fn test_create_transcript_segments_multiple() {
        let transcripts = vec![
            bt("First segment", 0.0, 2000.0),
            bt("Second segment", 3000.0, 5000.0),
            bt("Third segment", 6500.0, 8000.0),
        ];
        let segments = create_transcript_segments(&transcripts);

        assert_eq!(segments.len(), 3);

        assert_eq!(segments[0].text, "First segment");
        assert_eq!(segments[0].audio_start_time, Some(0.0));
        assert_eq!(segments[0].audio_end_time, Some(2.0));
        assert_eq!(segments[0].duration, Some(2.0));

        assert_eq!(segments[1].text, "Second segment");
        assert_eq!(segments[1].audio_start_time, Some(3.0));
        assert_eq!(segments[1].audio_end_time, Some(5.0));
        assert_eq!(segments[1].duration, Some(2.0));

        assert_eq!(segments[2].text, "Third segment");
        assert_eq!(segments[2].audio_start_time, Some(6.5));
        assert_eq!(segments[2].audio_end_time, Some(8.0));
        assert_eq!(segments[2].duration, Some(1.5));
    }

    #[test]
    fn test_create_transcript_segments_trims_whitespace() {
        let transcripts = vec![bt("  Hello with spaces  ", 0.0, 1000.0)];
        let segments = create_transcript_segments(&transcripts);

        assert_eq!(segments.len(), 1);
        assert_eq!(segments[0].text, "Hello with spaces");
    }

    #[test]
    fn test_create_transcript_segments_generates_unique_ids() {
        let transcripts = vec![
            bt("Segment one", 0.0, 1000.0),
            bt("Segment two", 1000.0, 2000.0),
        ];
        let segments = create_transcript_segments(&transcripts);

        assert_eq!(segments.len(), 2);
        assert_ne!(segments[0].id, segments[1].id);
        assert!(segments[0].id.starts_with("transcript-"));
        assert!(segments[1].id.starts_with("transcript-"));
    }

    #[test]
    fn test_create_transcript_segments_carries_speaker_attribution() {
        let transcripts = vec![
            BatchTranscript {
                words: vec![],
                text: "Hello".into(),
                start_ms: 0.0,
                end_ms: 1000.0,
                speaker: Some("Speaker 1".into()),
                voice_profile_id: None,
                sequence_id: 0,
                source: Default::default(),
            },
            BatchTranscript {
                words: vec![],
                text: "Hi".into(),
                start_ms: 1000.0,
                end_ms: 2000.0,
                speaker: Some("Alice".into()),
                voice_profile_id: Some("profile-abc".into()),
                sequence_id: 1,
                source: crate::speaker_diarization::SegmentSource::Microphone,
            },
        ];
        let segments = create_transcript_segments(&transcripts);
        assert_eq!(segments[0].speaker.as_deref(), Some("Speaker 1"));
        assert!(segments[0].voice_profile_id.is_none());
        assert_eq!(segments[1].speaker.as_deref(), Some("Alice"));
        assert_eq!(segments[1].voice_profile_id.as_deref(), Some("profile-abc"));
        // The channel a segment came from is stored when known (#102).
        assert!(segments[0].source.is_none());
        assert_eq!(segments[1].source.as_deref(), Some("mic"));
    }

    #[test]
    fn test_cancellation_flag() {
        // Reset flag to known state
        RETRANSCRIPTION_CANCELLED.store(false, Ordering::SeqCst);
        RETRANSCRIPTION_IN_PROGRESS.store(false, Ordering::SeqCst);

        assert!(!is_retranscription_in_progress());

        // Test cancellation
        cancel_retranscription();
        assert!(RETRANSCRIPTION_CANCELLED.load(Ordering::SeqCst));

        // Reset for other tests
        RETRANSCRIPTION_CANCELLED.store(false, Ordering::SeqCst);
    }

    #[test]
    fn test_find_audio_file_common_candidates() {
        let dir = tempfile::tempdir().unwrap();

        // No audio file → error
        assert!(find_audio_file(dir.path()).is_err());

        // Create audio.mp4 — should be found first
        std::fs::write(dir.path().join("audio.mp4"), b"fake").unwrap();
        let found = find_audio_file(dir.path()).unwrap();
        assert_eq!(found.file_name().unwrap(), "audio.mp4");
    }

    #[test]
    fn test_find_audio_file_supports_every_import_extension() {
        for extension in AUDIO_EXTENSIONS {
            let folder = tempfile::tempdir().unwrap();
            let media = folder.path().join(format!("audio.{extension}"));
            std::fs::write(&media, b"media").unwrap();
            assert_eq!(find_audio_file(folder.path()).unwrap(), media);
        }
    }

    #[test]
    fn test_find_audio_file_skips_directories_with_media_names() {
        let folder = tempfile::tempdir().unwrap();
        std::fs::create_dir(folder.path().join("audio.mp4")).unwrap();
        assert!(find_audio_file(folder.path()).is_err());
        let video = folder.path().join("audio.mkv");
        std::fs::write(&video, b"media").unwrap();
        assert_eq!(find_audio_file(folder.path()).unwrap(), video);

        std::fs::remove_file(video).unwrap();
        std::fs::create_dir(folder.path().join("not_a_file.wav")).unwrap();
        let fallback = folder.path().join("imported.FLAC");
        std::fs::write(&fallback, b"media").unwrap();
        assert_eq!(find_audio_file(folder.path()).unwrap(), fallback);
    }

    #[test]
    fn test_find_audio_file_non_mp4_extensions() {
        let dir = tempfile::tempdir().unwrap();

        // Create audio.wav (imported as .wav, not .mp4)
        std::fs::write(dir.path().join("audio.wav"), b"fake").unwrap();
        let found = find_audio_file(dir.path()).unwrap();
        assert_eq!(found.file_name().unwrap(), "audio.wav");
    }

    #[test]
    fn test_find_audio_file_fallback_scan() {
        let dir = tempfile::tempdir().unwrap();

        // Create a file with an audio extension but non-standard name
        std::fs::write(dir.path().join("my_recording.flac"), b"fake").unwrap();
        // Also add a non-audio file that should be ignored
        std::fs::write(dir.path().join("notes.txt"), b"text").unwrap();

        let found = find_audio_file(dir.path()).unwrap();
        assert_eq!(found.file_name().unwrap(), "my_recording.flac");
    }

    #[test]
    fn test_find_audio_file_priority_order() {
        let dir = tempfile::tempdir().unwrap();

        // Create both audio.m4a and audio.mp4 — mp4 should win (listed first in candidates)
        std::fs::write(dir.path().join("audio.m4a"), b"fake").unwrap();
        std::fs::write(dir.path().join("audio.mp4"), b"fake").unwrap();
        let found = find_audio_file(dir.path()).unwrap();
        assert_eq!(found.file_name().unwrap(), "audio.mp4");
    }

    #[test]
    fn test_find_audio_file_empty_folder() {
        let dir = tempfile::tempdir().unwrap();
        let result = find_audio_file(dir.path());
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("No audio file found"));
    }

    #[test]
    fn test_find_audio_file_nonexistent_folder() {
        let result = find_audio_file(Path::new("/nonexistent/path/12345"));
        assert!(result.is_err());
    }

    #[test]
    fn test_audio_extensions_constant() {
        // Verify all expected formats are covered
        assert!(AUDIO_EXTENSIONS.contains(&"mp4"));
        assert!(AUDIO_EXTENSIONS.contains(&"m4a"));
        assert!(AUDIO_EXTENSIONS.contains(&"wav"));
        assert!(AUDIO_EXTENSIONS.contains(&"mp3"));
        assert!(AUDIO_EXTENSIONS.contains(&"flac"));
        assert!(AUDIO_EXTENSIONS.contains(&"ogg"));
        assert!(AUDIO_EXTENSIONS.contains(&"aac"));
        // FFmpeg-backed formats
        assert!(AUDIO_EXTENSIONS.contains(&"mkv"));
        assert!(AUDIO_EXTENSIONS.contains(&"webm"));
        assert!(AUDIO_EXTENSIONS.contains(&"wma"));
        // Non-audio formats
        assert!(!AUDIO_EXTENSIONS.contains(&"txt"));
        assert!(!AUDIO_EXTENSIONS.contains(&"pdf"));
    }
}

/// Process a saved recording using a captured pipeline, sharing the existing
/// job guard, cancellation action and progress events with manual reprocessing.
pub async fn start_pipeline_processing(
    sink: SharedEventSink,
    pool: SqlitePool,
    meeting_id: String,
    options: super::processing::service::ProcessingOptions,
) -> Result<super::processing::service::ProcessingOutcome> {
    start_pipeline_processing_shared(sink, pool, meeting_id, options, None).await
}

pub(crate) async fn start_pipeline_processing_on_target(
    sink: SharedEventSink,
    pool: SqlitePool,
    meeting_id: String,
    options: super::processing::service::ProcessingOptions,
    target: super::processing::config::ProcessingTarget,
) -> Result<super::processing::service::ProcessingOutcome> {
    start_pipeline_processing_shared(sink, pool, meeting_id, options, Some(target)).await
}

async fn start_pipeline_processing_shared(
    sink: SharedEventSink,
    pool: SqlitePool,
    meeting_id: String,
    options: super::processing::service::ProcessingOptions,
    target: Option<super::processing::config::ProcessingTarget>,
) -> Result<super::processing::service::ProcessingOutcome> {
    let _guard = RetranscriptionGuard::acquire().map_err(|e| anyhow!(e))?;
    RETRANSCRIPTION_CANCELLED.store(false, Ordering::SeqCst);
    let progress = |stage: &str, percent: u32, message: &str| {
        emit_progress(sink.as_ref(), &meeting_id, stage, percent, message)
    };
    let result = match target {
        Some(target) => {
            super::processing::service::process_recording_on_target(
                &pool,
                &meeting_id,
                options,
                target,
                &RETRANSCRIPTION_CANCELLED,
                progress,
            )
            .await
        }
        None => {
            super::processing::service::process_recording(
                &pool,
                &meeting_id,
                options,
                &RETRANSCRIPTION_CANCELLED,
                progress,
            )
            .await
        }
    };

    match &result {
        Ok(outcome) => {
            let _=sink.emit_event("retranscription-complete",&serde_json::json!({"meeting_id":meeting_id,"segments_count":outcome.segments_count,"run_id":outcome.run_id,"identification_error":outcome.identification_error}));
        }
        Err(error) => {
            let _ = sink.emit_event(
                "retranscription-error",
                &RetranscriptionError {
                    meeting_id,
                    error: error.to_string(),
                },
            );
        }
    }
    result
}
