use super::{native::native_speakers, types::*};
use anyhow::{anyhow, Result};
use std::{path::Path, sync::atomic::AtomicBool};

/// Uses the existing batch pipeline as a reproducible baseline in a separate
/// process. No meeting or voice-profile database is opened or modified.
pub async fn benchmark_native(path: &Path, model: &str, language: String) -> Result<Analysis> {
    let started = std::time::Instant::now();
    crate::speaker_diarization::model::set_models_dir();
    let path_owned = path.to_owned();
    let decoded =
        tokio::task::spawn_blocking(move || crate::audio::decoder::decode_audio_file(&path_owned))
            .await??;
    let duration = decoded.duration_seconds;
    let samples = decoded.to_whisper_format();
    let mut analysis = Analysis {
        version: 1,
        audio_sha256: None,
        request_id: None,
        duration,
        language: Some(language.clone()),
        segments: vec![],
        words: vec![],
        turns: vec![],
        overlaps: vec![],
        embeddings: vec![],
        settings: serde_json::json!({"recognizer":"native","diarization":"native","model":model,"language":language}),
        provenance: serde_json::Value::Null,
        elapsed_seconds: 0.,
        recognition_seconds: None,
        diarization_seconds: None,
        recognition_device: Some("native build backend".into()),
        diarization_device: None,
        compute_type: None,
    };
    native_speakers(path, &mut analysis, 0).await?;
    let turns: Vec<_> = analysis
        .turns
        .iter()
        .map(|t| crate::speaker_diarization::offline::SpeakerTurn {
            start: t.start as f32,
            end: t.end as f32,
            speaker: t
                .speaker
                .trim_start_matches("Speaker ")
                .parse::<i32>()
                .unwrap_or(1)
                - 1,
        })
        .collect();
    let speech =
        tokio::task::spawn_blocking(move || crate::audio::vad::get_speech_chunks(&samples, 800))
            .await??;
    let engine = std::sync::Arc::new(crate::whisper_engine::WhisperEngine::new_with_models_dir(
        Some(
            crate::paths::app_data_dir()
                .map_err(|e| anyhow!(e))?
                .join("models"),
        ),
    )?);
    engine.discover_models().await?;
    engine.load_model(model).await?;
    let diarizer = crate::speaker_diarization::service::build_diarizer(None).await?;
    static CANCEL: AtomicBool = AtomicBool::new(false);
    let batch = crate::audio::common::run_batch_transcription(
        &speech,
        Some(language),
        engine.clone(),
        diarizer,
        Some(&turns),
        None,
        None,
        &crate::glossary::PreparedGlossary::default(),
        &CANCEL,
        |_, _, _| {},
    )
    .await?;
    analysis.segments = batch
        .into_iter()
        .map(|s| Segment {
            start: s.start_ms / 1000.,
            end: s.end_ms / 1000.,
            text: s.text,
            speaker: s.speaker,
            overlapping_speakers: vec![],
        })
        .collect();
    analysis.elapsed_seconds = started.elapsed().as_secs_f64();
    engine.unload_model().await;
    Ok(analysis)
}
