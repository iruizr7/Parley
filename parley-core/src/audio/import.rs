// Audio file import module - allows importing external audio files as new meetings

use crate::audio::decoder::decode_audio_file;
use crate::database::repositories::meeting::MeetingsRepository;
use crate::events::{EventSink, EventSinkExt, SharedEventSink};
use anyhow::{anyhow, Result};
use log::{debug, warn};
use serde::{Deserialize, Serialize};
use sqlx::SqlitePool;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};

use super::audio_processing::create_meeting_folder;
#[cfg(test)]
use super::common::create_transcript_segments;
#[cfg(test)]
use super::common::write_transcripts_json;
use super::constants::AUDIO_EXTENSIONS;
use super::recording_preferences::get_default_recordings_folder;

/// Global flag to track if import is in progress
pub static IMPORT_IN_PROGRESS: AtomicBool = AtomicBool::new(false);

/// Global flag to signal cancellation
static IMPORT_CANCELLED: AtomicBool = AtomicBool::new(false);

/// Own an incomplete copy until its saved recording is durably registered.
/// Once processing can start, the meeting owns the source for recovery.
struct PendingImportFolder(Option<PathBuf>);
impl Drop for PendingImportFolder {
    fn drop(&mut self) {
        if let Some(path) = self.0.as_ref() {
            let _ = std::fs::remove_dir_all(path);
        }
    }
}

/// RAII guard for IMPORT_IN_PROGRESS flag
/// Ensures flag is cleared even if import panics or returns early
struct ImportGuard;

impl ImportGuard {
    /// Create guard and set flag atomically
    fn acquire() -> Result<Self, String> {
        if IMPORT_IN_PROGRESS
            .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
            .is_err()
        {
            return Err("Import already in progress".to_string());
        }
        Ok(ImportGuard)
    }
}

impl Drop for ImportGuard {
    fn drop(&mut self) {
        IMPORT_IN_PROGRESS.store(false, Ordering::SeqCst);
    }
}

/// Maximum file size: 20GB (prevents OOM and excessive processing time)
const MAX_FILE_SIZE_BYTES: u64 = 20 * 1024 * 1024 * 1024; // 20GB

/// Information about a selected audio file
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioFileInfo {
    pub path: String,
    pub filename: String,
    pub duration_seconds: f64,
    pub size_bytes: u64,
    pub format: String,
}

/// Progress update emitted during import
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImportProgress {
    pub stage: String, // "copying", "decoding", "vad", "transcribing", "saving"
    pub progress_percentage: u32,
    pub message: String,
    /// Measured progress of the current subtask; absent when its size is unknown.
    #[serde(default)]
    pub stage_percentage: Option<u32>,
}

/// Result of import
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImportResult {
    pub meeting_id: String,
    pub title: String,
    pub segments_count: usize,
    pub duration_seconds: f64,
}

/// Error during import
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImportError {
    pub error: String,
    #[serde(default)]
    pub meeting_id: Option<String>,
    #[serde(default)]
    pub retained_recording: bool,
    #[serde(default)]
    pub cancelled: bool,
}
#[derive(Debug)]
struct RetainedImportError {
    meeting_id: String,
    error: String,
    cancelled: bool,
}
impl std::fmt::Display for RetainedImportError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}; the imported recording was kept in its meeting",
            self.error
        )
    }
}
impl std::error::Error for RetainedImportError {}
impl ImportError {
    fn from_failure(error: &anyhow::Error) -> Self {
        let retained = error.downcast_ref::<RetainedImportError>();
        Self {
            error: error.to_string(),
            meeting_id: retained.map(|e| e.meeting_id.clone()),
            retained_recording: retained.is_some(),
            cancelled: retained.is_some_and(|e| e.cancelled),
        }
    }
}

/// Warning emitted during import (non-fatal)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImportWarning {
    pub warning: String,
    pub details: Option<String>,
}

/// Response when import is started
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImportStarted {
    pub message: String,
}

/// Check if import is currently in progress
pub fn is_import_in_progress() -> bool {
    IMPORT_IN_PROGRESS.load(Ordering::SeqCst)
}

/// Cancel ongoing import
pub fn cancel_import() {
    IMPORT_CANCELLED.store(true, Ordering::SeqCst);
}

/// Validate an audio file and return its info using metadata-only approach
/// Falls back to full decode if metadata is unavailable
pub fn validate_audio_file(path: &Path) -> Result<AudioFileInfo> {
    validate_audio_file_reporting(path, |_| {})
}
fn validate_audio_file_reporting(path: &Path, progress: impl Fn(&str)) -> Result<AudioFileInfo> {
    // Check file exists
    if !path.exists() {
        return Err(anyhow!("File does not exist: {}", path.display()));
    }

    // Check extension
    let extension = path
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_lowercase())
        .unwrap_or_default();

    if !AUDIO_EXTENSIONS.contains(&extension.as_str()) {
        return Err(anyhow!(
            "Unsupported format: .{}. Supported: {}",
            extension,
            AUDIO_EXTENSIONS.join(", ")
        ));
    }

    // Get file size
    let metadata = std::fs::metadata(path).map_err(|e| anyhow!("Cannot read file: {}", e))?;
    let size_bytes = metadata.len();

    // Check file size limit
    if size_bytes > MAX_FILE_SIZE_BYTES {
        return Err(anyhow!(
            "File too large: {:.2}GB. Maximum supported size is {}GB",
            size_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
            MAX_FILE_SIZE_BYTES / (1024 * 1024 * 1024)
        ));
    }

    // Get filename without extension for title
    let filename = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("Imported Audio")
        .to_string();

    // Try fast metadata-only validation first
    let duration_seconds = match extract_duration_from_metadata(path) {
        Ok(duration) => {
            debug!("Got duration from metadata: {:.2}s (fast path)", duration);
            duration
        }
        Err(e) => {
            // Fallback to full decode if metadata unavailable
            warn!(
                "Metadata extraction failed: {}, falling back to full decode",
                e
            );
            progress("Decoding media to determine its duration");
            let decoded = decode_audio_file(path)?;
            decoded.duration_seconds
        }
    };

    Ok(AudioFileInfo {
        path: path.to_string_lossy().to_string(),
        filename,
        duration_seconds,
        size_bytes,
        format: extension.to_uppercase(),
    })
}

/// Extract duration from audio file metadata without full decode
/// Returns error if metadata is unavailable, triggering fallback to full decode
fn extract_duration_from_metadata(path: &Path) -> Result<f64> {
    use symphonia::core::formats::FormatOptions;
    use symphonia::core::io::MediaSourceStream;
    use symphonia::core::meta::MetadataOptions;
    use symphonia::core::probe::Hint;

    // Open the file
    let file =
        std::fs::File::open(path).map_err(|e| anyhow!("Failed to open audio file: {}", e))?;

    let mss = MediaSourceStream::new(Box::new(file), Default::default());

    // Set up format hint based on file extension
    let mut hint = Hint::new();
    if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
        hint.with_extension(ext);
    }

    // Probe the file format (lightweight operation)
    let probed = symphonia::default::get_probe()
        .format(
            &hint,
            mss,
            &FormatOptions::default(),
            &MetadataOptions::default(),
        )
        .map_err(|e| anyhow!("Failed to probe audio format: {}", e))?;

    let format = probed.format;

    // Find the first audio track
    use symphonia::core::codecs::CODEC_TYPE_NULL;
    let track = format
        .tracks()
        .iter()
        .find(|t| t.codec_params.codec != CODEC_TYPE_NULL)
        .ok_or_else(|| anyhow!("No audio track found in file"))?;

    // Extract duration from metadata
    let sample_rate = track
        .codec_params
        .sample_rate
        .ok_or_else(|| anyhow!("Unknown sample rate"))?;

    let n_frames = track
        .codec_params
        .n_frames
        .ok_or_else(|| anyhow!("Frame count not available in metadata"))?;

    let duration_seconds = n_frames as f64 / sample_rate as f64;

    debug!(
        "Extracted metadata: {}Hz, {} frames, {:.2}s",
        sample_rate, n_frames, duration_seconds
    );

    Ok(duration_seconds)
}

/// Compatibility entry point; all execution uses the typed pipeline service.
pub async fn start_import_with(
    sink: SharedEventSink,
    pool: Option<SqlitePool>,
    source_path: String,
    title: String,
    language: Option<String>,
    model: Option<String>,
    provider: Option<String>,
    num_speakers: i32,
) -> Result<ImportResult> {
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
    start_import_with_options(
        sink,
        Some(pool),
        source_path,
        title,
        super::processing::service::ProcessingOptions {
            config,
            language,
            num_speakers,
        },
    )
    .await
}

async fn run_import(
    sink: &SharedEventSink,
    pool: &SqlitePool,
    source_path: String,
    title: String,
    options: super::processing::service::ProcessingOptions,
) -> Result<ImportResult> {
    run_import_into(
        sink,
        pool,
        source_path,
        title,
        options,
        &get_default_recordings_folder(),
        &IMPORT_CANCELLED,
    )
    .await
}

async fn run_import_into(
    sink: &SharedEventSink,
    pool: &SqlitePool,
    source_path: String,
    title: String,
    options: super::processing::service::ProcessingOptions,
    recordings_folder: &PathBuf,
    cancelled: &AtomicBool,
) -> Result<ImportResult> {
    let source = PathBuf::from(source_path);
    emit_progress(sink, "validating", 0, "Inspecting the selected recording");
    let info = validate_audio_file_reporting(&source, |message| emit_progress(sink, "validating", 0, message))?;
    super::processing::config::validate_config(&options.config)?;
    let target = super::processing::config::get_target_for_config(pool, &options.config).await?;
    if cancelled.load(Ordering::SeqCst) {
        return Err(anyhow!("Import cancelled"));
    }
    let prepared = super::import_media::prepare(&source, info.duration_seconds, cancelled, |stage, percent, message| {
        emit_pipeline_progress(sink, stage, percent, message);
    }).await?;
    let source = &prepared.path;
    emit_progress(sink, "copying", 5, "Saving the imported recording");
    let meeting_folder = create_meeting_folder(recordings_folder, &title, false)?;
    let mut pending_folder = PendingImportFolder(Some(meeting_folder.clone()));
    let dest_filename = format!(
        "audio.{}",
        source.extension().and_then(|e| e.to_str()).unwrap_or("wav")
    );
    let dest_path = meeting_folder.join(&dest_filename);
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    let mut input = tokio::fs::File::open(&source).await?;
    let mut output = tokio::fs::File::create(&dest_path).await?;
    let total = input.metadata().await?.len();
    let mut copied = 0u64;
    let mut last_percent = None;
    let mut bytes = vec![0u8; 1024 * 1024];
    loop {
        if cancelled.load(Ordering::SeqCst) {
            return Err(anyhow!("Import cancelled"));
        }
        let n = input.read(&mut bytes).await?;
        if n == 0 {
            break;
        }
        output.write_all(&bytes[..n]).await?;
        copied += n as u64;
        let percent = ((copied as u128 * 100) / total.max(1) as u128).min(100) as u32;
        if last_percent != Some(percent) {
            emit_measured_progress(sink, "copying", percent, "Copying the audio into the meeting folder");
            last_percent = Some(percent);
        }
    }
    emit_progress(sink, "flushing", 9, "Flushing the saved recording to disk");
    output.flush().await?;
    output.sync_all().await?;
    drop(output);
    drop(input);
    // The meeting now owns its durable audio copy. Do not retain another
    // temporary copy throughout a potentially long remote processing job.
    drop(prepared);
    emit_progress(sink, "registering", 10, "Registering the saved recording in the library");
    let meeting_id = format!("meeting-{}", uuid::Uuid::new_v4());
    let now = chrono::Utc::now().to_rfc3339();
    let metadata = super::common::MeetingMetadata {
        version: Some("1.0".into()),
        meeting_id: Some(meeting_id.clone()),
        meeting_name: Some(title.clone()),
        created_at: Some(now.clone()),
        completed_at: Some(now),
        duration_seconds: Some(info.duration_seconds),
        audio_file: Some(dest_filename),
        status: Some("completed".into()),
        origin: Some("import".into()),
        ..Default::default()
    };
    super::common::write_metadata(&meeting_folder, &metadata, || metadata.clone())?;
    MeetingsRepository::create_imported_meeting(
        pool,
        &meeting_id,
        &title,
        &meeting_folder.to_string_lossy(),
        &dest_path.to_string_lossy(),
        info.duration_seconds,
        cancelled,
    )
    .await?;
    // From this point onward every failed/cancelled job keeps the visible
    // meeting and source recording. Detached workers can be safely resumed.
    pending_folder.0 = None;
    let _ = sink.emit_event(
        "import-saved",
        &ImportResult {
            meeting_id: meeting_id.clone(),
            title: title.clone(),
            segments_count: 0,
            duration_seconds: info.duration_seconds,
        },
    );
    let outcome = super::processing::service::process_recording_on_target(
        pool,
        &meeting_id,
        options,
        target,
        cancelled,
        |stage, percent, message| emit_pipeline_progress(sink, stage, percent, message),
    )
    .await
    .map_err(|error| {
        anyhow::Error::new(RetainedImportError {
            meeting_id: meeting_id.clone(),
            error: error.to_string(),
            cancelled: cancelled.load(Ordering::SeqCst),
        })
    })?;
    if let Some(error) = outcome.identification_error {
        let _ = sink.emit_event(
            "speaker-identification-error",
            &serde_json::json!({"meeting_id":meeting_id,"error":error}),
        );
    }
    let duration_seconds = super::processing::runs::load(pool, &meeting_id)
        .await
        .ok()
        .flatten()
        .map(|a| a.duration)
        .unwrap_or(info.duration_seconds);
    emit_progress(sink, "complete", 100, "Import complete");
    Ok(ImportResult {
        meeting_id,
        title,
        segments_count: outcome.segments_count,
        duration_seconds,
    })
}

fn emit_pipeline_progress(sink: &dyn EventSink, stage: &str, percent: u32, message: &str) {
    if matches!(stage, "audio_integrity" | "model_integrity" | "recognition" | "verifying"
        | "audio_conversion" | "audio_decoding" | "audio_resampling" | "vad_progress"
        | "worker_decoding_progress" | "diarizing_progress" | "embedding_progress" | "uploading" | "audio_extraction" | "audio_compression") {
        emit_measured_progress(sink, stage, percent, message);
    } else {
        emit_progress(sink, stage, 10 + percent.min(100) * 85 / 100, message);
    }
}

/// Emit an `import-progress` event through `sink`.
fn emit_progress(sink: &dyn EventSink, stage: &str, progress: u32, message: &str) {
    let _ = sink.emit_event(
        "import-progress",
        &ImportProgress {
            stage: stage.to_string(),
            progress_percentage: progress,
            message: message.to_string(),
            stage_percentage: None,
        },
    );
}

fn emit_measured_progress(sink: &dyn EventSink, stage: &str, percent: u32, message: &str) {
    let _ = sink.emit_event("import-progress", &ImportProgress {
        stage: stage.into(),
        // Keep the legacy field for other clients; the desktop displays only
        // stage_percentage, explicitly labelled as current-subtask progress.
        progress_percentage: match stage {
            "copying" => percent.min(100) / 10,
            "audio_integrity" => 10 + percent.min(100) / 20,
            "recognition" => 20 + percent.min(100) * 55 / 100,
            "model_integrity" => 75,
            "verifying" => 90 + percent.min(100) / 20,
            _ => percent.min(100),
        },
        message: message.into(),
        stage_percentage: Some(percent.min(100)),
    });
}

#[cfg(test)]
mod tests {
    use crate::audio::common::TranscriptSegment;
    #[test]
    fn long_phase_measurements_reach_the_import_ui_without_reweighting() {
        let sink = crate::events::RecordingSink::new();
        for stage in ["audio_conversion", "audio_decoding", "audio_resampling", "vad_progress", "worker_decoding_progress", "diarizing_progress", "embedding_progress", "uploading", "audio_extraction", "audio_compression"] {
            super::emit_pipeline_progress(&sink, stage, 37, "Working");
        }
        super::emit_pipeline_progress(&sink, "diarizing", 0, "Grouping speakers");
        let events = sink.payloads("import-progress");
        assert!(events[..10].iter().all(|event| event["stage_percentage"] == 37));
        assert!(events[10]["stage_percentage"].is_null());
    }
    #[test]
    fn incomplete_import_copy_is_removed_and_original_is_kept() {
        let root = tempfile::tempdir().unwrap();
        let original = root.path().join("original.mkv");
        std::fs::write(&original, b"original").unwrap();
        let folder = root.path().join("new-import");
        std::fs::create_dir(&folder).unwrap();
        std::fs::copy(&original, folder.join("audio.mkv")).unwrap();
        drop(super::PendingImportFolder(Some(folder.clone())));
        assert!(!folder.exists());
        assert_eq!(std::fs::read(original).unwrap(), b"original");
    }
    #[test]
    fn committed_import_keeps_its_recording() {
        let root = tempfile::tempdir().unwrap();
        let folder = root.path().join("committed");
        std::fs::create_dir(&folder).unwrap();
        std::fs::write(folder.join("audio.mkv"), b"saved").unwrap();
        let mut guard = super::PendingImportFolder(Some(folder.clone()));
        guard.0 = None;
        drop(guard);
        assert!(folder.join("audio.mkv").is_file());
    }
    use super::super::common::{split_segment_at_silence, BatchTranscript};
    use super::*;

    async fn isolated_pool() -> SqlitePool {
        let pool = sqlx::sqlite::SqlitePoolOptions::new()
            .max_connections(1)
            .connect("sqlite::memory:")
            .await
            .unwrap();
        sqlx::migrate!("./migrations").run(&pool).await.unwrap();
        pool
    }
    fn fictional_wav(path: &Path) {
        // Twelve seconds of PCM16 mono silence; no external media fixture.
        let data_size = 12 * 16000 * 2u32;
        let mut bytes = Vec::with_capacity(44 + data_size as usize);
        bytes.extend_from_slice(b"RIFF");
        bytes.extend_from_slice(&(36 + data_size).to_le_bytes());
        bytes.extend_from_slice(b"WAVEfmt ");
        bytes.extend_from_slice(&16u32.to_le_bytes());
        bytes.extend_from_slice(&1u16.to_le_bytes());
        bytes.extend_from_slice(&1u16.to_le_bytes());
        bytes.extend_from_slice(&16000u32.to_le_bytes());
        bytes.extend_from_slice(&32000u32.to_le_bytes());
        bytes.extend_from_slice(&2u16.to_le_bytes());
        bytes.extend_from_slice(&16u16.to_le_bytes());
        bytes.extend_from_slice(b"data");
        bytes.extend_from_slice(&data_size.to_le_bytes());
        bytes.resize(44 + data_size as usize, 0);
        std::fs::write(path, bytes).unwrap();
    }
    async fn unavailable_pipeline(
        pool: &SqlitePool,
    ) -> super::super::processing::service::ProcessingOptions {
        use super::super::processing::config::*;
        let target = ProcessingTarget {
            id: "fictional-local".into(),
            name: "Fictional unavailable device".into(),
            device: "fictional-unavailable".into(),
            ..Default::default()
        };
        save_target(pool, &target).await.unwrap();
        super::super::processing::service::ProcessingOptions {
            config: PipelineConfig {
                target_id: target.id,
                identification: Identification::Off,
                ..Default::default()
            },
            language: Some("en".into()),
            num_speakers: 2,
        }
    }
    #[tokio::test]
    async fn processing_failure_retains_imported_source_meeting_and_recovery_history() {
        let pool = isolated_pool().await;
        let root = tempfile::tempdir().unwrap();
        let original = root.path().join("fictional.wav");
        fictional_wav(&original);
        let sink = std::sync::Arc::new(crate::events::RecordingSink::new());
        let shared: SharedEventSink = sink.clone();
        let options = unavailable_pipeline(&pool).await;
        let failure = run_import_into(
            &shared,
            &pool,
            original.to_string_lossy().into_owned(),
            "Fictional retained import".into(),
            options,
            &root.path().join("saved"),
            &AtomicBool::new(false),
        )
        .await
        .unwrap_err();
        let event = ImportError::from_failure(&failure);
        assert!(event.retained_recording);
        assert!(!event.cancelled);
        let meeting = event.meeting_id.unwrap();
        let (folder, audio): (String, String) =
            sqlx::query_as("SELECT folder_path,audio_path FROM meetings WHERE id=?")
                .bind(&meeting)
                .fetch_one(&pool)
                .await
                .unwrap();
        let original_audio = decode_audio_file(&original).unwrap();
        let saved_audio = decode_audio_file(Path::new(&audio)).unwrap();
        assert_eq!(original_audio.sample_rate, saved_audio.sample_rate);
        assert_eq!(original_audio.channels, saved_audio.channels);
        assert_eq!(original_audio.samples, saved_audio.samples);
        std::fs::remove_file(original).unwrap();
        assert!(
            Path::new(&audio).is_file(),
            "the owned import remains even when the external original disappears"
        );
        let metadata: serde_json::Value = serde_json::from_slice(
            &std::fs::read(Path::new(&folder).join("metadata.json")).unwrap(),
        )
        .unwrap();
        assert_eq!(metadata["audio_file"], Path::new(&audio).file_name().unwrap().to_str().unwrap());
        assert_eq!(metadata["origin"], "import");
        let history = super::super::processing::runs::list_runs(&pool, &meeting)
            .await
            .unwrap();
        assert_eq!(history.len(), 1);
        assert_eq!(history[0].outcome, "failed");
        assert_eq!(history[0].requested["reconnectable"], false);
        assert!(super::super::processing::service::resume_processing(
            &pool,
            &meeting,
            &history[0].id,
            &AtomicBool::new(false),
            |_, _, _| panic!("native failure cannot reconnect to a worker")
        )
        .await
        .unwrap_err()
        .to_string()
        .contains("did not create a reconnectable worker job"));
        let request = history[0].requested["request_id"].as_str().unwrap();
        let snapshot: String = sqlx::query_scalar("SELECT value FROM app_settings WHERE key=?")
            .bind(format!("processing_recovery_v1:{request}"))
            .fetch_one(&pool)
            .await
            .unwrap();
        let snapshot: serde_json::Value = serde_json::from_str(&snapshot).unwrap();
        assert_eq!(snapshot["options"]["num_speakers"], 2);
        assert_eq!(snapshot["options"]["language"], "en");
        assert_eq!(sink.payloads("import-saved")[0]["meeting_id"], meeting);
        let events = sink.payloads("import-progress");
        let stages: Vec<_> = events.iter().filter_map(|e| e["stage"].as_str()).collect();
        let position = |stage| stages.iter().position(|s| *s == stage).unwrap();
        if super::super::ffmpeg::find_ffmpeg_path().is_some() {
            assert!(position("audio_compression") < position("copying"));
            assert!(events.iter().any(|e| e["stage"] == "audio_compression" && e["stage_percentage"] == 100));
        }
        assert!(position("copying") < position("flushing"));
        assert!(position("flushing") < position("registering"));
        assert!(position("registering") < position("audio_integrity"));
        assert!(events.iter().any(|e| e["stage"] == "copying" && e["stage_percentage"] == 100));
        assert!(events.iter().any(|e| e["stage"] == "audio_integrity" && e["stage_percentage"] == 100));

        let count: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM transcripts")
            .fetch_one(&pool)
            .await
            .unwrap();
        assert_eq!(count, 0);
    }
    #[cfg(unix)]
    struct DeleteTargetBeforeCopy {
        pool: SqlitePool,
        target_id: String,
    }
    #[cfg(unix)]
    impl EventSink for DeleteTargetBeforeCopy {
        fn emit_value(&self, event: &str, payload: serde_json::Value) -> Result<(), String> {
            if event == "import-progress" && payload["stage"] == "copying" {
                // This sink is used only for the first copying event. SQLx's
                // SQLite worker is driven by the test's second runtime thread
                // while this synchronous event callback waits for the edit.
                let pool = self.pool.clone();
                let id = self.target_id.clone();
                let runtime = tokio::runtime::Handle::current();
                std::thread::spawn(move || {
                    runtime.block_on(super::super::processing::config::delete_target(&pool, &id))
                })
                .join()
                .unwrap()
                .map_err(|e| e.to_string())?;
            }
            Ok(())
        }
    }

    #[cfg(unix)]
    fn recovered_fixture_analysis() -> serde_json::Value {
        serde_json::json!({
            "version":1,"duration":12.,"language":"en",
            "segments":[{"start":1.,"end":5.,"text":"Fictional recovered import.","speaker":"Speaker 1"}],
            "words":[],"turns":[{"start":0.,"end":10.,"speaker":"Speaker 1"}],"overlaps":[],
            "embeddings":[{
                "speaker":"Speaker 1","model_id":"campplus","vector":[1.,0.],"clean_seconds":4.,"sample_count":1,
                "embedding_space":{
                    "id":speaker_core::campplus_space_id("fictional-checkpoint-hash"),"model":"campplus",
                    "revision":"fictional-checkpoint-hash","preprocessing":"pcm16k-mono-ffmpeg-v1",
                    "normalization":"l2","aggregation":"longest-three-trimmed-10s-v1",
                    "extraction_version":"parley-native-v2","dimension":2
                },
                "source_intervals":[{"speaker":"Speaker 1","start":1.,"end":5.,"channel":0}]
            }],
            "settings":{},"provenance":{"synthetic_transport_fixture":true}
        })
    }
    #[cfg(unix)]
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn retained_import_reconnects_through_real_transport_once_and_runs_cached_identification()
    {
        use super::super::processing::{config::*, executor, runs, service};
        let pool = isolated_pool().await;
        let root = tempfile::tempdir().unwrap();
        let original = root.path().join("fictional.wav");
        fictional_wav(&original);
        let target = executor::test_failure_target(&root.path().join("worker")).unwrap();
        save_target(&pool, &target).await.unwrap();
        let options = service::ProcessingOptions {
            config: PipelineConfig {
                target_id: target.id.clone(),
                speech_engine: SpeechEngine::FasterWhisper,
                speech_model: "large-v3-turbo-int8".into(),
                speaker_pipeline: SpeakerPipeline::Community1,
                identification: Identification::Campplus,
                ..Default::default()
            },
            language: Some("en".into()),
            num_speakers: 2,
        };
        let sink: SharedEventSink = std::sync::Arc::new(DeleteTargetBeforeCopy {
            pool: pool.clone(),
            target_id: target.id.clone(),
        });
        let cancelled = AtomicBool::new(false);
        let failure = run_import_into(
            &sink,
            &pool,
            original.to_string_lossy().into_owned(),
            "Fictional reconnect".into(),
            options,
            &root.path().join("saved"),
            &cancelled,
        )
        .await
        .unwrap_err();
        let meeting = ImportError::from_failure(&failure)
            .meeting_id
            .unwrap_or_else(|| {
                panic!("Expected a retained import after target deletion: {failure:#}")
            });
        assert!(
            get_target(&pool, &target.id).await.is_err(),
            "the selected target was deleted after import began"
        );
        let failed = runs::list_runs(&pool, &meeting).await.unwrap().remove(0);
        assert_eq!(failed.outcome, "failed");
        assert_eq!(failed.requested["reconnectable"], true);
        let request = failed.requested["request_id"].as_str().unwrap();
        assert_eq!(
            executor::status(&target, request).await.unwrap()["state"],
            "failed"
        );
        executor::seed_test_result(&target, request, recovered_fixture_analysis())
            .await
            .unwrap();
        // Recovery must use the original private target rather than a later
        // edit with the same public target ID and another empty runtime.
        let mut edited = target.clone();
        edited.worker_path = Some(
            root.path()
                .join("different-runtime")
                .to_string_lossy()
                .into_owned(),
        );
        save_target(&pool, &edited).await.unwrap();
        let outcome =
            service::resume_processing(&pool, &meeting, &failed.id, &cancelled, |_, _, _| {})
                .await
                .unwrap();
        assert_eq!(outcome.run_id, request);
        assert_eq!(outcome.segments_count, 1);
        assert!(outcome.identification_error.is_none());
        let published = runs::list_runs(&pool, &meeting).await.unwrap();
        assert_eq!(
            published
                .iter()
                .filter(|r| r.operation == "identification")
                .count(),
            1,
            "recovery runs the selected identification after publication"
        );
        assert_eq!(
            published
                .iter()
                .filter(|r| r.operation == "process" && r.outcome == "succeeded")
                .count(),
            1
        );
        let again =
            service::resume_processing(&pool, &meeting, &failed.id, &cancelled, |_, _, _| {})
                .await
                .unwrap();
        assert_eq!(again.run_id, outcome.run_id);
        assert_eq!(
            runs::list_runs(&pool, &meeting).await.unwrap().len(),
            published.len()
        );
        let meetings: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM meetings")
            .fetch_one(&pool)
            .await
            .unwrap();
        let rows: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM transcripts")
            .fetch_one(&pool)
            .await
            .unwrap();
        let samples: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM enrollment_samples")
            .fetch_one(&pool)
            .await
            .unwrap();
        assert_eq!((meetings, rows, samples), (1, 1, 0));
        let audio: String = sqlx::query_scalar("SELECT audio_path FROM meetings WHERE id=?")
            .bind(&meeting)
            .fetch_one(&pool)
            .await
            .unwrap();
        assert!(Path::new(&audio).is_file());
        let source: String =
            sqlx::query_scalar("SELECT transcript FROM transcripts WHERE meeting_id=?")
                .bind(&meeting)
                .fetch_one(&pool)
                .await
                .unwrap();
        assert_eq!(source, "Fictional recovered import.");
    }
    #[cfg(unix)]
    #[tokio::test]
    async fn recovered_import_keeps_transcript_when_followup_identification_fails() {
        use super::super::processing::{config::*, executor, runs, service};
        let pool = isolated_pool().await;
        let root = tempfile::tempdir().unwrap();
        let original = root.path().join("fictional.wav");
        fictional_wav(&original);
        let target = executor::test_failure_target(&root.path().join("worker")).unwrap();
        save_target(&pool, &target).await.unwrap();
        let options = service::ProcessingOptions {
            config: PipelineConfig {
                target_id: target.id.clone(),
                speech_engine: SpeechEngine::FasterWhisper,
                speech_model: "large-v3-turbo-int8".into(),
                speaker_pipeline: SpeakerPipeline::Community1,
                identification: Identification::Wespeaker,
                ..Default::default()
            },
            language: None,
            num_speakers: 0,
        };
        let sink: SharedEventSink = std::sync::Arc::new(crate::events::NullSink);
        let cancelled = AtomicBool::new(false);
        let failure = run_import_into(
            &sink,
            &pool,
            original.to_string_lossy().into_owned(),
            "Fictional separate identification failure".into(),
            options,
            &root.path().join("saved"),
            &cancelled,
        )
        .await
        .unwrap_err();
        let meeting = ImportError::from_failure(&failure).meeting_id.unwrap();
        let failed = runs::list_runs(&pool, &meeting).await.unwrap().remove(0);
        let request = failed.requested["request_id"].as_str().unwrap();
        executor::seed_test_result(&target, request, recovered_fixture_analysis())
            .await
            .unwrap();
        let mut edited = target.clone();
        let redirected = root.path().join("different-runtime");
        edited.worker_path = Some(redirected.to_string_lossy().into_owned());
        save_target(&pool, &edited).await.unwrap();
        delete_target(&pool, &target.id).await.unwrap();
        let outcome =
            service::resume_processing(&pool, &meeting, &failed.id, &cancelled, |_, _, _| {})
                .await
                .unwrap();
        assert_eq!(outcome.segments_count, 1);
        assert!(outcome.identification_error.is_some());
        let retained_jobs = std::fs::read_dir(root.path().join("worker/jobs"))
            .unwrap()
            .filter_map(Result::ok)
            .filter(|e| e.file_type().is_ok_and(|t| t.is_dir()))
            .count();
        assert_eq!(retained_jobs, 1, "follow-up extraction reaches the original worker even after the target was edited and deleted");
        assert!(
            !redirected.exists(),
            "automatic follow-up extraction must inherit the original target snapshot"
        );
        let history = runs::list_runs(&pool, &meeting).await.unwrap();
        assert!(history
            .iter()
            .any(|r| r.operation == "process" && r.outcome == "succeeded"));
        assert!(history
            .iter()
            .any(|r| r.operation == "identification" && r.outcome == "failed"));
        let count: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM transcripts WHERE meeting_id=?")
            .bind(&meeting)
            .fetch_one(&pool)
            .await
            .unwrap();
        assert_eq!(count, 1);
    }

    struct CancelAfterSaved(std::sync::Arc<AtomicBool>);
    impl EventSink for CancelAfterSaved {
        fn emit_value(&self, event: &str, _: serde_json::Value) -> Result<(), String> {
            if event == "import-saved" {
                self.0.store(true, Ordering::SeqCst);
            }
            Ok(())
        }
    }
    #[tokio::test]
    async fn cancellation_before_copy_has_no_meeting_but_saved_import_cancellation_keeps_source() {
        let pool = isolated_pool().await;
        let root = tempfile::tempdir().unwrap();
        let original = root.path().join("fictional.wav");
        fictional_wav(&original);
        let options = unavailable_pipeline(&pool).await;
        let sink: SharedEventSink = std::sync::Arc::new(crate::events::NullSink);
        let before = run_import_into(
            &sink,
            &pool,
            original.to_string_lossy().into_owned(),
            "Fictional before copy".into(),
            options.clone(),
            &root.path().join("saved"),
            &AtomicBool::new(true),
        )
        .await
        .unwrap_err();
        assert!(!ImportError::from_failure(&before).retained_recording);
        let count: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM meetings")
            .fetch_one(&pool)
            .await
            .unwrap();
        assert_eq!(count, 0);
        let cancelled = std::sync::Arc::new(AtomicBool::new(false));
        let sink: SharedEventSink = std::sync::Arc::new(CancelAfterSaved(cancelled.clone()));
        let failure = run_import_into(
            &sink,
            &pool,
            original.to_string_lossy().into_owned(),
            "Fictional after copy".into(),
            options,
            &root.path().join("saved"),
            &cancelled,
        )
        .await
        .unwrap_err();
        let event = ImportError::from_failure(&failure);
        assert!(event.cancelled && event.retained_recording);
        let meeting = event.meeting_id.unwrap();
        let audio: String = sqlx::query_scalar("SELECT audio_path FROM meetings WHERE id=?")
            .bind(&meeting)
            .fetch_one(&pool)
            .await
            .unwrap();
        assert!(Path::new(&audio).is_file());
        assert!(original.is_file());
        let history = super::super::processing::runs::list_runs(&pool, &meeting)
            .await
            .unwrap();
        assert_eq!(history.len(), 1);
        assert_eq!(history[0].outcome, "cancelled");
    }

    #[test]
    fn test_audio_extensions() {
        assert!(AUDIO_EXTENSIONS.contains(&"mp4"));
        assert!(AUDIO_EXTENSIONS.contains(&"wav"));
        assert!(AUDIO_EXTENSIONS.contains(&"mp3"));
        assert!(!AUDIO_EXTENSIONS.contains(&"txt"));
    }

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
        let transcripts = vec![bt("Hello world", 0.0, 1500.0)];
        let segments = create_transcript_segments(&transcripts);

        assert_eq!(segments.len(), 1);
        assert_eq!(segments[0].text, "Hello world");
        assert_eq!(segments[0].audio_start_time, Some(0.0));
        assert_eq!(segments[0].audio_end_time, Some(1.5));
    }

    #[test]
    fn test_cancellation_flag() {
        IMPORT_CANCELLED.store(false, Ordering::SeqCst);
        IMPORT_IN_PROGRESS.store(false, Ordering::SeqCst);

        assert!(!is_import_in_progress());

        cancel_import();
        assert!(IMPORT_CANCELLED.load(Ordering::SeqCst));

        // Reset
        IMPORT_CANCELLED.store(false, Ordering::SeqCst);
    }

    #[test]
    fn test_extract_duration_from_metadata_wav() {
        // Test with sample WAV file if available
        let test_path = Path::new("../../backend/whisper.cpp/samples/jfk.wav");
        if test_path.exists() {
            let result = extract_duration_from_metadata(test_path);
            // Should succeed and return a reasonable duration
            assert!(result.is_ok());
            let duration = result.unwrap();
            assert!(
                duration > 0.0 && duration < 60.0,
                "Duration {} seems unreasonable",
                duration
            );
        }
    }

    #[test]
    fn test_extract_duration_from_metadata_mp3() {
        // Test with sample MP3 file if available
        let test_path = Path::new("../../backend/whisper.cpp/samples/jfk.mp3");
        if test_path.exists() {
            let result = extract_duration_from_metadata(test_path);
            // MP3 files may not have n_frames metadata, so fallback is expected
            // We just verify it doesn't panic
            let _ = result;
        }
    }

    #[test]
    fn test_validate_audio_file_with_metadata() {
        // Test validation with actual audio file
        let test_path = Path::new("../../backend/whisper.cpp/samples/jfk.wav");
        if test_path.exists() {
            let result = validate_audio_file(test_path);
            assert!(result.is_ok());
            let info = result.unwrap();
            assert_eq!(info.format, "WAV");
            assert!(info.duration_seconds > 0.0);
            assert!(info.size_bytes > 0);
        }
    }

    #[test]
    fn test_validate_audio_file_nonexistent() {
        let result = validate_audio_file(Path::new("/nonexistent/file.mp4"));
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("does not exist"));
    }

    #[test]
    fn test_validate_audio_file_wrong_extension() {
        // Create a temporary file with wrong extension
        let temp_dir = std::env::temp_dir();
        let temp_file = temp_dir.join("test_audio.txt");
        let _ = std::fs::write(&temp_file, b"dummy content");

        let result = validate_audio_file(&temp_file);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Unsupported format"));

        // Cleanup
        let _ = std::fs::remove_file(temp_file);
    }

    #[test]
    fn test_split_segment_at_silence_short_segment() {
        // Segment shorter than max — returned as-is
        let segment = crate::audio::vad::SpeechSegment {
            samples: vec![0.1; 16000], // 1 second
            start_timestamp_ms: 0.0,
            end_timestamp_ms: 1000.0,
            confidence: 0.9,
            source: crate::audio::recording_state::DeviceType::Microphone,
        };
        let result = split_segment_at_silence(&segment, 25 * 16000);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].samples.len(), 16000);
    }

    #[test]
    fn test_split_segment_at_silence_splits_long_segment() {
        // 60-second segment of low-level noise with a silent gap at ~25s
        let mut samples = vec![0.01f32; 60 * 16000];
        // Insert silence at 25 seconds (sample 400000)
        for i in (25 * 16000)..(25 * 16000 + 3200) {
            samples[i] = 0.0;
        }
        let segment = crate::audio::vad::SpeechSegment {
            samples,
            start_timestamp_ms: 0.0,
            end_timestamp_ms: 60_000.0,
            confidence: 0.9,
            source: crate::audio::recording_state::DeviceType::Microphone,
        };

        let result = split_segment_at_silence(&segment, 25 * 16000);
        assert!(
            result.len() >= 2,
            "Should split into at least 2 segments, got {}",
            result.len()
        );

        // All sub-segments should have samples
        for (i, seg) in result.iter().enumerate() {
            assert!(!seg.samples.is_empty(), "Segment {} is empty", i);
            assert!(
                seg.start_timestamp_ms < seg.end_timestamp_ms,
                "Segment {} has invalid timestamps: {} >= {}",
                i,
                seg.start_timestamp_ms,
                seg.end_timestamp_ms
            );
        }
    }

    #[test]
    fn test_split_segment_at_silence_no_silence_uses_overlap() {
        // Continuous speech (constant energy) — should still split with overlap
        let segment = crate::audio::vad::SpeechSegment {
            samples: vec![0.5f32; 60 * 16000], // 60 seconds of "speech"
            start_timestamp_ms: 0.0,
            end_timestamp_ms: 60_000.0,
            confidence: 0.9,
            source: crate::audio::recording_state::DeviceType::Microphone,
        };

        let result = split_segment_at_silence(&segment, 25 * 16000);
        assert!(result.len() >= 2);

        // Total samples should exceed input due to overlap
        let total_samples: usize = result.iter().map(|s| s.samples.len()).sum();
        assert!(
            total_samples >= 60 * 16000,
            "Overlap should not lose samples"
        );
    }

    #[test]
    fn test_write_transcripts_json() {
        let dir = tempfile::tempdir().unwrap();
        let segments = vec![
            TranscriptSegment {
                id: "t-1".to_string(),
                text: "Hello world".to_string(),
                timestamp: Some("2024-01-01T00:00:00Z".to_string()),
                audio_start_time: Some(0.0),
                audio_end_time: Some(1.5),
                duration: Some(1.5),
                display_time: None,
                confidence: None,
                sequence_id: None,
                speaker: None,
                voice_profile_id: None,
                source: None,
            },
            TranscriptSegment {
                id: "t-2".to_string(),
                text: "Second segment".to_string(),
                timestamp: Some("2024-01-01T00:00:01Z".to_string()),
                audio_start_time: Some(2.0),
                audio_end_time: Some(3.5),
                duration: Some(1.5),
                display_time: None,
                confidence: None,
                sequence_id: None,
                speaker: None,
                voice_profile_id: None,
                source: None,
            },
        ];

        let result = write_transcripts_json(dir.path(), &segments);
        assert!(
            result.is_ok(),
            "write_transcripts_json failed: {:?}",
            result
        );

        // Verify file exists and is valid JSON
        let path = dir.path().join("transcripts.json");
        assert!(path.exists());

        let content = std::fs::read_to_string(&path).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&content).unwrap();
        assert_eq!(parsed["total_segments"], 2);
        assert_eq!(parsed["version"], "1.0");
        assert_eq!(parsed["segments"][0]["text"], "Hello world");
        assert_eq!(parsed["segments"][1]["text"], "Second segment");
        assert_eq!(parsed["segments"][0]["sequence_id"], 0);
        assert_eq!(parsed["segments"][1]["sequence_id"], 1);

        // Verify temp file was cleaned up
        assert!(!dir.path().join(".transcripts.json.tmp").exists());
    }

    #[test]
    fn test_write_import_metadata() {
        let dir = tempfile::tempdir().unwrap();

        let metadata = super::super::common::MeetingMetadata {
            version: Some("1.0".to_string()),
            meeting_id: Some("meeting-123".to_string()),
            meeting_name: Some("Test Meeting".to_string()),
            created_at: Some("2024-01-01T00:00:00Z".to_string()),
            completed_at: Some("2024-01-01T00:00:00Z".to_string()),
            duration_seconds: Some(1800.0),
            audio_file: Some("audio.mp4".to_string()),
            transcript_file: Some("transcripts.json".to_string()),
            status: Some("completed".to_string()),
            origin: Some("import".to_string()),
            ..Default::default()
        };
        let result =
            super::super::common::write_metadata(dir.path(), &metadata, || metadata.clone());
        assert!(result.is_ok(), "write_metadata failed: {:?}", result);

        let path = dir.path().join("metadata.json");
        assert!(path.exists());

        let content = std::fs::read_to_string(&path).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&content).unwrap();
        assert_eq!(parsed["version"], "1.0");
        assert_eq!(parsed["meeting_id"], "meeting-123");
        assert_eq!(parsed["meeting_name"], "Test Meeting");
        assert_eq!(parsed["duration_seconds"], 1800.0);
        assert_eq!(parsed["audio_file"], "audio.mp4");
        assert_eq!(parsed["status"], "completed");
        assert_eq!(parsed["origin"], "import");
    }

    /// Integration test that decodes a real audio file and runs VAD.
    /// Run with: TEST_AUDIO_PATH=/path/to/audio.mp4 cargo test -- --ignored --nocapture
    #[test]
    #[ignore]
    fn test_import_pipeline_decode_vad() {
        let audio_path = std::env::var("TEST_AUDIO_PATH")
            .expect("Set TEST_AUDIO_PATH to run this integration test");

        let path = Path::new(&audio_path);
        assert!(path.exists(), "Audio file not found: {}", audio_path);

        // Step 1: Decode
        println!("Decoding {}...", audio_path);
        let decoded =
            crate::audio::decoder::decode_audio_file(path).expect("Failed to decode audio file");
        println!(
            "Decoded: {:.2}s, {}Hz, {} channels, {} samples",
            decoded.duration_seconds,
            decoded.sample_rate,
            decoded.channels,
            decoded.samples.len()
        );

        // Step 2: Resample to 16kHz mono
        println!("Resampling to 16kHz mono...");
        let samples = decoded.to_whisper_format();
        println!(
            "Resampled: {} samples ({:.2}s at 16kHz)",
            samples.len(),
            samples.len() as f64 / 16000.0
        );

        // Step 3: Run VAD with both redemption times and compare
        for redemption_ms in [400u32, 2000] {
            println!("\n--- VAD with redemption_time={}ms ---", redemption_ms);
            let segments = crate::audio::vad::get_speech_chunks_with_progress(
                &samples,
                redemption_ms,
                |progress, count| {
                    if progress % 20 == 0 {
                        println!("  VAD progress: {}% ({} segments)", progress, count);
                    }
                    true
                },
            )
            .expect("VAD failed");

            let total_segments = segments.len();
            println!("Found {} segments", total_segments);

            if !segments.is_empty() {
                let durations: Vec<f64> = segments
                    .iter()
                    .map(|s| s.end_timestamp_ms - s.start_timestamp_ms)
                    .collect();
                let total_speech: f64 = durations.iter().sum();
                let avg = total_speech / durations.len() as f64;
                let min = durations.iter().cloned().fold(f64::INFINITY, f64::min);
                let max = durations.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

                println!(
                    "Stats: avg={:.0}ms, min={:.0}ms, max={:.0}ms, total_speech={:.1}s/{:.1}s ({:.0}%)",
                    avg, min, max,
                    total_speech / 1000.0,
                    decoded.duration_seconds,
                    (total_speech / 1000.0 / decoded.duration_seconds) * 100.0
                );

                // Segments over 25s that would be split
                let oversized = durations.iter().filter(|d| **d > 25_000.0).count();
                println!("Segments >25s (would be split): {}", oversized);

                // Basic sanity checks
                assert!(total_speech > 0.0, "No speech detected");
                for (i, seg) in segments.iter().enumerate() {
                    assert!(!seg.samples.is_empty(), "Segment {} has no samples", i);
                    assert!(
                        seg.end_timestamp_ms > seg.start_timestamp_ms,
                        "Segment {} has invalid timestamps",
                        i
                    );
                }
            }
        }
    }
}

/// Import with an immutable, meeting-specific pipeline snapshot.
pub async fn start_import_with_options(
    sink: SharedEventSink,
    pool: Option<SqlitePool>,
    source_path: String,
    title: String,
    options: super::processing::service::ProcessingOptions,
) -> Result<ImportResult> {
    let _guard = ImportGuard::acquire().map_err(|e| anyhow!(e))?;
    IMPORT_CANCELLED.store(false, Ordering::SeqCst);
    let result = match pool {
        Some(pool) => run_import(&sink, &pool, source_path, title, options).await,
        None => Err(anyhow!("Database unavailable")),
    };
    match &result {
        Ok(r) => {
            let _=sink.emit_event("import-complete",&serde_json::json!({"meeting_id":r.meeting_id,"title":r.title,"segments_count":r.segments_count,"duration_seconds":r.duration_seconds}));
        }
        Err(e) => {
            let mut error = ImportError::from_failure(e);
            error.cancelled |= IMPORT_CANCELLED.load(Ordering::SeqCst);
            let _ = sink.emit_event("import-error", &error);
        }
    }
    result
}
