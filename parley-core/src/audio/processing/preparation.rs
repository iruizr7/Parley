//! Route each selected artifact to the registry its execution adapter consumes.
use super::config::*;
use anyhow::{anyhow, Result};
use serde_json::{json, Value};
use std::{collections::BTreeSet, sync::Arc};
use tokio::sync::mpsc;
static PREPARE_GATE: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());
#[derive(Debug, Default, PartialEq, Eq)]
pub(crate) struct PreparationPlan {
    pub app_speech: Option<String>,
    pub app_campplus: bool,
    pub app_segmentation: bool,
    pub worker: Vec<String>,
}
pub(crate) fn plan(
    target: &ProcessingTarget,
    config: &PipelineConfig,
    speech: bool,
) -> PreparationPlan {
    let local = target.kind == TargetKind::Local;
    let mut p = PreparationPlan::default();
    let mut worker = BTreeSet::new();
    if speech {
        if local && config.speech_engine == SpeechEngine::WhisperCpp {
            p.app_speech = Some(config.speech_model.clone());
        } else {
            worker.insert(config.speech_model.clone());
        }
    }
    match config.speaker_pipeline {
        SpeakerPipeline::Sherpa if local => {
            p.app_campplus = true;
            p.app_segmentation = true;
        }
        SpeakerPipeline::Sherpa => {
            worker.insert("campplus".into());
            worker.insert("sherpa-segmentation".into());
        }
        SpeakerPipeline::Community1 => {
            worker.insert("community1".into());
        }
    }
    match config.identification {
        Identification::Campplus if local => p.app_campplus = true,
        Identification::Campplus => {
            worker.insert("campplus".into());
        }
        Identification::Wespeaker => {
            worker.insert("community1".into());
        }
        _ => {}
    }
    p.worker = worker.into_iter().collect();
    p
}
pub async fn prepare_pipeline(
    target: &ProcessingTarget,
    config: &PipelineConfig,
    progress: impl Fn(&str, u32, &str) + Send + Sync,
) -> Result<Value> {
    validate_config(config)?;
    prepare(target, config, true, progress).await
}
pub async fn prepare_speaker_pipeline(
    target: &ProcessingTarget,
    pipeline: SpeakerPipeline,
    identification: Identification,
    progress: impl Fn(&str, u32, &str) + Send + Sync,
) -> Result<Value> {
    let config = PipelineConfig {
        target_id: target.id.clone(),
        speaker_pipeline: pipeline,
        identification,
        ..Default::default()
    };
    prepare(target, &config, false, progress).await
}
async fn prepare(
    target: &ProcessingTarget,
    config: &PipelineConfig,
    speech: bool,
    progress: impl Fn(&str, u32, &str) + Send + Sync,
) -> Result<Value> {
    let _guard = PREPARE_GATE.lock().await;
    let p = plan(target, config, speech);
    let mut prepared = vec![];
    if let Some(model) = &p.app_speech {
        let directory = super::native::local_model_path(model)?
            .parent()
            .ok_or_else(|| anyhow!("Native model registry unavailable"))?
            .to_owned();
        let existing = {
            let guard = crate::whisper_engine::models::WHISPER_ENGINE
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            guard.as_ref().cloned()
        };
        let engine = if let Some(engine) = existing {
            if engine.get_models_directory().await == directory {
                engine
            } else {
                Arc::new(crate::whisper_engine::WhisperEngine::new_with_models_dir(
                    Some(directory),
                )?)
            }
        } else {
            Arc::new(crate::whisper_engine::WhisperEngine::new_with_models_dir(
                Some(directory),
            )?)
        };
        let models = engine.discover_models().await?;
        let ready = models.iter().any(|m| {
            m.name == *model && matches!(m.status, crate::whisper_engine::ModelStatus::Available)
        });
        if !ready {
            if models.iter().any(|m| {
                m.name == *model
                    && matches!(
                        m.status,
                        crate::whisper_engine::ModelStatus::Corrupted { .. }
                    )
            }) {
                engine.delete_model(model).await?;
            }
            progress(
                "downloading",
                0,
                "Downloading the selected native speech model into the app registry",
            );
            let (tx, rx) = mpsc::unbounded_channel();
            let callback = Box::new(move |percent: u8| {
                let _ = tx.send(percent as u32);
            });
            forward(
                engine.download_model(model, Some(callback)),
                rx,
                &progress,
                "Downloading native speech model",
            )
            .await?;
        }
        prepared.push(json!({"id":model,"registry":"app","ready":true}));
    }
    if p.app_campplus {
        crate::speaker_diarization::model::set_models_dir();
        // Repair an interrupted/invalid selected artifact before asking the existing
        // manager to download it. A non-empty partial file is not a ready model.
        if !super::native::probe_embedding_local().await? {
            if let Some(path) =
                crate::speaker_diarization::model::default_model_path().filter(|p| p.is_file())
            {
                std::fs::remove_file(path)?;
            }
        }
        let (tx, rx) = mpsc::unbounded_channel();
        let sink: crate::events::SharedEventSink = Arc::new(DownloadSink(tx));
        progress(
            "downloading",
            0,
            "Preparing CAMPPlus in the app speaker registry",
        );
        forward(
            async {
                crate::speaker_diarization::service::download_speaker_model(sink)
                    .await
                    .map_err(anyhow::Error::msg)
            },
            rx,
            &progress,
            "Downloading CAMPPlus speaker model",
        )
        .await?;
        if !super::native::probe_embedding_local().await? {
            return Err(anyhow!(
                "Downloaded CAMPPlus artifact did not pass its execution check"
            ));
        }
        prepared.push(json!({"id":"campplus","registry":"app","ready":true}));
    }
    if p.app_segmentation {
        crate::speaker_diarization::model::set_models_dir();
        if !super::native::probe_speakers_local().await? {
            if let Some(path) = crate::speaker_diarization::model::pyannote_segmentation_path()
                .filter(|p| p.is_file())
            {
                std::fs::remove_file(path)?;
            }
        }
        progress(
            "downloading",
            0,
            "Preparing Sherpa segmentation in the app registry",
        );
        crate::speaker_diarization::service::ensure_pyannote_segmentation_model()
            .await
            .map_err(anyhow::Error::msg)?;
        if !super::native::probe_speakers_local().await? {
            return Err(anyhow!(
                "Downloaded Sherpa package did not pass its execution check"
            ));
        }
        prepared.push(json!({"id":"sherpa-segmentation","registry":"app","ready":true}));
    }
    let worker = if p.worker.is_empty() {
        None
    } else {
        Some(super::executor::prepare_artifacts(target, config, &p.worker, &progress).await?)
    };
    progress(
        "ready",
        100,
        "Selected models prepared; check readiness to verify execution",
    );
    Ok(json!({"version":1,"operation":"prepare","prepared":prepared,"worker":worker,"issues":[]}))
}
struct DownloadSink(mpsc::UnboundedSender<u32>);
impl crate::events::EventSink for DownloadSink {
    fn emit_value(&self, _event: &str, payload: Value) -> std::result::Result<(), String> {
        if let Some(percent) = payload["progress"].as_u64() {
            let _ = self.0.send(percent.min(100) as u32);
        }
        Ok(())
    }
}
async fn forward<F: std::future::Future<Output = Result<()>>>(
    future: F,
    mut updates: mpsc::UnboundedReceiver<u32>,
    progress: &(impl Fn(&str, u32, &str) + Send + Sync),
    message: &str,
) -> Result<()> {
    tokio::pin!(future);
    let mut open = true;
    loop {
        tokio::select! {result=&mut future=>return result,percent=updates.recv(),if open=>match percent{Some(percent)=>progress("downloading",percent,message),None=>open=false}}
    }
}
