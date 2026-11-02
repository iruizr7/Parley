use super::types::*;
use anyhow::{anyhow, Result};
use std::path::Path;
pub(crate) static NATIVE_GATE: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

// Successful checks are reusable within this app session. File identity includes
// timestamps and (on Unix) inode/change time so replaced artifacts invalidate it.
#[derive(Clone, Debug, PartialEq, Eq)]
struct ProbeKey {
    model: String,
    device: String,
    files: Vec<String>,
}
fn file_stamp(path: &Path) -> String {
    let Ok(m) = path.metadata() else { return format!("{}:missing", path.display()); };
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        format!("{}:{}:{:?}:{}:{}:{}", path.display(), m.len(), m.modified().ok(), m.ino(), m.ctime(), m.ctime_nsec())
    }
    #[cfg(not(unix))]
    { format!("{}:{}:{:?}", path.display(), m.len(), m.modified().ok()) }
}
static PROBE_CACHE: tokio::sync::Mutex<Vec<(ProbeKey, serde_json::Value)>> = tokio::sync::Mutex::const_new(Vec::new());
static LOCAL_ENGINE: tokio::sync::Mutex<Option<std::sync::Arc<crate::whisper_engine::WhisperEngine>>> = tokio::sync::Mutex::const_new(None);

/// Reuse Parley's CPU speaker engine when only recognition was switched.
/// The optional recognizer does not silently change the speaker engine.
pub async fn native_speakers(path: &Path, analysis: &mut Analysis, count: i32) -> Result<()> {
    let path_owned = path.to_owned();
    let turns = tokio::task::spawn_blocking(move || -> Result<_> {
        let samples = normalize_speaker_audio(&path_owned)?;
        let segmentation = crate::speaker_diarization::model::pyannote_segmentation_path()
            .ok_or_else(|| anyhow!("Native speaker models are not configured"))?;
        let embedding = crate::speaker_diarization::model::default_model_path()
            .ok_or_else(|| anyhow!("Native speaker model is not configured"))?;
        crate::speaker_diarization::offline::diarize_offline(
            &samples,
            &segmentation,
            &embedding,
            count,
            4,
        )
    })
    .await??;
    analysis.turns = turns
        .into_iter()
        .map(|t| TimedSpeaker {
            start: t.start as f64,
            end: t.end as f64,
            speaker: format!("Speaker {}", t.speaker + 1),
        })
        .collect();
    annotate(analysis);
    add_native_evidence(path, analysis).await?;
    analysis.diarization_device = Some("cpu".into());
    let segmentation = crate::speaker_diarization::model::pyannote_segmentation_path()
        .ok_or_else(|| anyhow!("Speaker model unavailable"))?;
    let embedding = crate::speaker_diarization::model::default_model_path()
        .ok_or_else(|| anyhow!("Speaker model unavailable"))?;
    analysis.provenance["speaker_artifacts"] = serde_json::json!({"segmentation_sha256":super::runs::audio_hash(segmentation).await?,"embedding_sha256":super::runs::audio_hash(embedding).await?,"runtime":speaker_core::runtime_info(),"clustering":"parley-offline-v1","device":"cpu","num_speakers":count});
    analysis.validate()
}

/// Extracts reference CAMPPlus evidence from actual clean intervals. Every pooled
/// vector retains exactly the excerpts that contributed to it.
pub async fn extract_native_embeddings(
    path: &Path,
    model: &Path,
    intervals: &[SourceInterval],
) -> Result<Vec<SpeakerEmbedding>> {
    let path = path.to_owned();
    let model = model.to_owned();
    let intervals = intervals.to_vec();
    tokio::task::spawn_blocking(move || -> Result<Vec<SpeakerEmbedding>> {
        let samples = normalize_speaker_audio(&path)?;
        speaker_core::extract_clean_embeddings(&samples, &model, &intervals)
    })
    .await?
}
pub(crate) async fn add_native_evidence(path: &Path, analysis: &mut Analysis) -> Result<()> {
    let Some(model) =
        crate::speaker_diarization::model::default_model_path().filter(|p| p.is_file())
    else {
        return Ok(());
    };
    let intervals = super::service::clean_intervals(analysis);
    analysis
        .embeddings
        .extend(extract_native_embeddings(path, &model, &intervals).await?);
    Ok(())
}

pub fn annotate(analysis: &mut Analysis) {
    let mut events: Vec<_> = analysis
        .turns
        .iter()
        .flat_map(|t| {
            [
                (t.start, 1i32, t.speaker.clone()),
                (t.end, -1i32, t.speaker.clone()),
            ]
        })
        .collect();
    events.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
    let mut active = std::collections::BTreeMap::<String, i32>::new();
    let mut previous = 0.;
    analysis.overlaps.clear();
    for (at, delta, speaker) in events {
        let speakers: Vec<_> = active
            .iter()
            .filter(|(_, n)| **n > 0)
            .map(|(s, _)| s.clone())
            .collect();
        if at > previous && speakers.len() > 1 {
            analysis.overlaps.push(Overlap {
                start: previous,
                end: at,
                speakers,
            });
        }
        *active.entry(speaker).or_default() += delta;
        previous = at;
    }
    let speaker_for = |start: f64, end: f64| {
        analysis
            .turns
            .iter()
            .filter_map(|t| {
                let overlap = (end.min(t.end) - start.max(t.start)).max(0.);
                if overlap <= 0. {
                    return None;
                }
                let coverage = overlap / (end - start).max(0.001);
                Some((
                    coverage >= 0.8,
                    if coverage >= 0.8 {
                        -(t.end - t.start)
                    } else {
                        coverage
                    },
                    &t.speaker,
                ))
            })
            .max_by(|a, b| a.0.cmp(&b.0).then(a.1.total_cmp(&b.1)))
            .map(|(_, _, s)| s.clone())
    };
    if !analysis.words.is_empty()
        && analysis.provenance["word_alignment"]["unaligned_words"]
            .as_u64()
            .unwrap_or(0)
            == 0
    {
        let mut segments: Vec<Segment> = vec![];
        for original in &analysis.segments {
            let words: Vec<_> = analysis.words.iter().filter(|w| w.start >= original.start - 1e-6 && w.end <= original.end + 1e-6).collect();
            let text = words.iter().map(|w| w.text.as_str()).collect::<String>();
            if text.split_whitespace().ne(original.text.split_whitespace()) {
                let mut fallback = original.clone();
                fallback.speaker = speaker_for(original.start, original.end);
                fallback.overlapping_speakers = analysis.overlapping_speakers(Some(original.start), Some(original.end));
                segments.push(fallback);
                continue;
            }
            let source_start = segments.len();
            for w in words {
                let same_source = segments.len() > source_start;
                let speaker = speaker_for(w.start, w.end);
                let overlapping_speakers = analysis.overlapping_speakers(Some(w.start), Some(w.end));
                if let Some(s) = segments.last_mut().filter(|s| {
                    same_source && s.speaker == speaker
                        && s.overlapping_speakers == overlapping_speakers
                        && w.start - s.end <= 1.
                        && w.end - s.start <= 15.
                }) {
                    s.end = w.end;
                    s.text.push_str(&w.text);
                } else {
                    segments.push(Segment {
                        start: w.start,
                        end: w.end,
                        text: w.text.clone(),
                        speaker,
                        overlapping_speakers,
                    });
                }
            }
        }
        analysis.segments = segments;
    } else {
        let labels: Vec<_> = analysis
            .segments
            .iter()
            .map(|s| {
                (
                    speaker_for(s.start, s.end),
                    analysis.overlapping_speakers(Some(s.start), Some(s.end)),
                )
            })
            .collect();
        for (s, (speaker, overlapping)) in analysis.segments.iter_mut().zip(labels) {
            s.speaker = speaker;
            s.overlapping_speakers = overlapping;
        }
    }
    let word_labels: Vec<_> = analysis
        .words
        .iter()
        .map(|w| {
            (
                speaker_for(w.start, w.end),
                analysis.overlapping_speakers(Some(w.start), Some(w.end)),
            )
        })
        .collect();
    for (word, (speaker, overlapping)) in analysis.words.iter_mut().zip(word_labels) {
        word.speaker = speaker;
        word.overlapping_speakers = overlapping;
    }
}

/// Enumerate the actual linked ggml devices instead of inferring support from
/// the host's name. An explicitly requested unavailable accelerator is an error.
pub fn resolve_device(requested: &str) -> Result<String> {
    let mut available = vec!["cpu".to_owned()];
    #[cfg(target_os = "linux")]
    unsafe {
        whisper_rs::whisper_rs_sys::ggml_backend_load_all();
        for index in 0..whisper_rs::whisper_rs_sys::ggml_backend_dev_count() {
            let device = whisper_rs::whisper_rs_sys::ggml_backend_dev_get(index);
            let kind = whisper_rs::whisper_rs_sys::ggml_backend_dev_type(device);
            if kind==whisper_rs::whisper_rs_sys::ggml_backend_dev_type_GGML_BACKEND_DEVICE_TYPE_GPU||kind==whisper_rs::whisper_rs_sys::ggml_backend_dev_type_GGML_BACKEND_DEVICE_TYPE_IGPU {
    let backend=whisper_rs::whisper_rs_sys::ggml_backend_dev_backend_reg(device);let name=std::ffi::CStr::from_ptr(whisper_rs::whisper_rs_sys::ggml_backend_reg_name(backend)).to_string_lossy().to_lowercase();
    if ["cuda","vulkan","metal"].contains(&name.as_str()){available.push(name);}
   }
        }
    }
    if requested == "auto" {
        return Ok(available.last().cloned().unwrap_or_else(|| "cpu".into()));
    }
    if !available.iter().any(|d| d == requested) {
        return Err(anyhow!("This Parley build has no usable {requested} device; select an available target device explicitly"));
    }
    Ok(requested.into())
}
pub fn local_model_path(model: &str) -> Result<std::path::PathBuf> {
    Ok(crate::paths::app_data_dir()
        .map_err(|e| anyhow!(e))?
        .join("models")
        .join(format!("ggml-{model}.bin")))
}
async fn local_engine() -> Result<std::sync::Arc<crate::whisper_engine::WhisperEngine>> {
    let mut slot = LOCAL_ENGINE.lock().await;
    if slot.is_none() {
        *slot = Some(std::sync::Arc::new(crate::whisper_engine::WhisperEngine::new_with_models_dir(
            Some(crate::paths::app_data_dir().map_err(|e| anyhow!(e))?.join("models")),
        )?));
    }
    let engine = slot.as_ref().unwrap().clone();
    drop(slot);
    engine.discover_models().await?;
    Ok(engine)
}
pub async fn local_analysis(path: &Path, language: Option<String>) -> Result<Analysis> {
    let path = path.to_owned();
    let duration = tokio::task::spawn_blocking(move || {
        crate::audio::decoder::decode_audio_file(&path).map(|a| a.duration_seconds)
    })
    .await??;
    Ok(serde_json::from_value(
        serde_json::json!({"version":1,"duration":duration,"language":language,"segments":[],"words":[],"turns":[],"overlaps":[],"embeddings":[]}),
    )?)
}
/// Local native ASR shares Parley's existing decoder, VAD and whisper engine.
/// Speaker analysis is deliberately a separate step and cannot train profiles.
pub async fn recognize_local(
    path: &Path,
    model: &str,
    language: Option<String>,
    device: &str,
    cancelled: &std::sync::atomic::AtomicBool,
    progress: impl Fn(&str, u32, &str) + Send + Sync,
) -> Result<Analysis> {
    use std::sync::atomic::Ordering;
    let started = std::time::Instant::now();
    let device = resolve_device(device)?;
    progress("decoding", 0, "Decoding audio and converting to 16 kHz mono");
    let path_owned = path.to_owned();
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
    let task = tokio::task::spawn_blocking(move || -> Result<_> {
        let decode_tx = tx.clone();
        let decoded = crate::audio::decoder::decode_audio_file_with_progress(&path_owned, Some(Box::new(move |p, message| {
            let stage = if message.contains("FFmpeg") { "audio_conversion" } else { "audio_decoding" };
            decode_tx.send((stage, p, message.to_owned())).is_ok()
        })))?;
        let duration = decoded.duration_seconds;
        let _ = tx.send(("audio_resampling", 0, "Mixing channels and resampling audio to 16 kHz".into()));
        let resample_tx = tx.clone();
        let samples = decoded.to_whisper_format_with_progress(Some(Box::new(move |p, _| {
            resample_tx.send(("audio_resampling", p, "Resampling audio to 16 kHz".into())).is_ok()
        })))?;
        let _ = tx.send(("audio_resampling", 100, "Resampling audio to 16 kHz".into()));
        Ok((duration, samples))
    });
    while let Some((stage, percent, message)) = rx.recv().await {
        if cancelled.load(Ordering::SeqCst) { rx.close(); break; }
        progress(stage, percent, &message);
    }
    let (duration, samples) = task.await??;
    if cancelled.load(Ordering::SeqCst) { anyhow::bail!("Processing cancelled"); }
    let mut analysis: Analysis = serde_json::from_value(serde_json::json!({
        "version":1,"duration":duration,"language":language,
        "segments":[],"words":[],"turns":[],"overlaps":[],"embeddings":[]
    }))?;
    progress("vad_progress", 0, "Finding speech intervals in the recording");
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
    let task = tokio::task::spawn_blocking(move || {
        crate::audio::vad::get_speech_chunks_with_progress(&samples, 800, move |percent, _| tx.send(percent).is_ok())
    });
    while let Some(percent) = rx.recv().await {
        if cancelled.load(Ordering::SeqCst) { rx.close(); break; }
        progress("vad_progress", percent, "Finding speech intervals in the recording");
    }
    let speech = task.await??;
    progress("vad_progress", 100, "Finding speech intervals in the recording");
    if cancelled.load(Ordering::SeqCst) {
        anyhow::bail!("Processing cancelled");
    }
    progress("model_loading", 0, "Loading the selected speech model");
    let engine = local_engine().await?;
    engine.load_model_on_device(model, &device).await?;
    progress("recognition", 0, "Transcribing speech intervals");
    let batch = crate::audio::common::run_batch_transcription(
        &speech,
        language,
        engine.clone(),
        None,
        None,
        None,
        None,
        &crate::glossary::PreparedGlossary::default(),
        cancelled,
        |i, total, _| {
            progress(
                "recognition",
                i as u32 * 100 / total.max(1) as u32,
                "Transcribing on the selected local device",
            )
        },
    )
    .await;
    engine.unload_model().await;
    let batch = batch?;
    progress("recognition", 100, "Speech recognition complete");
    analysis.words = batch.iter().flat_map(|b| b.words.iter()).map(|w| super::types::Word {
        start: w.start, end: w.end, text: w.text.clone(), probability: Some(w.probability as f64),
        speaker: None, overlapping_speakers: vec![],
    }).collect();
    analysis.segments = crate::audio::common::create_transcript_segments(&batch)
        .into_iter()
        .map(|s| Segment {
            start: s.audio_start_time.unwrap_or(0.),
            end: s.audio_end_time.unwrap_or(0.),
            text: s.text,
            speaker: None,
            overlapping_speakers: vec![],
        })
        .collect();
    let artifact = local_model_path(model)?;
    let cached_digest = PROBE_CACHE.lock().await.iter()
        .find(|(key, _)| key.model == model && key.files.first() == Some(&file_stamp(&artifact)))
        .and_then(|(_, report)| report["artifacts"][0]["sha256"].as_str().map(str::to_owned));
    let hash = if let Some(digest) = cached_digest {
        progress("model_integrity", 100, "Reusing the verified speech model fingerprint");
        digest
    } else {
        super::runs::audio_hash_with_progress(artifact, |percent| {
            progress("model_integrity", percent.into(), "Checking speech model provenance");
        }).await?
    };
    analysis.recognition_device = Some(device.clone());
    analysis.compute_type = Some("whisper.cpp backend-native mixed precision".into());
    analysis.elapsed_seconds = started.elapsed().as_secs_f64();
    analysis.provenance = serde_json::json!({"speech_engine":"whisper_cpp","engine_version":whisper_rs::get_whisper_version(),"wrapper_version":"whisper-rs-0.16.0","speech_model":{"checkpoint":"large-v3-turbo","artifact":model,"sha256":hash,"stored_weights":if model.ends_with("q5_0"){"Q5_0"}else{"floating-point"}},"speech_device":device,"compute_type":"backend-native mixed precision; per-operation precision not exposed","decoder":{"vad_min_silence_ms":800,"profile":"parley-batch-v2","word_timestamps":true}});
    analysis.provenance["word_alignment"] = serde_json::json!({
        "method": "whisper.cpp token timestamps",
        "timed_words": analysis.words.iter().map(|w| w.text.split_whitespace().count()).sum::<usize>(),
        "timed_units": analysis.words.len(),
        "grouped_word_units": analysis.words.iter().filter(|w| w.text.split_whitespace().count() > 1).count(),
        "segment_fallback_count": batch.iter().filter(|b| b.words.is_empty()).count(),
    });
    Ok(analysis)
}

/// Probe the installed native artifact with the real execution path. This does
/// not provision Python, download models or require an optional worker.
pub async fn probe_local(
    config: &super::config::PipelineConfig,
    target: &super::config::ProcessingTarget,
) -> Result<serde_json::Value> {
    probe_local_with_progress(config, target, std::sync::Arc::new(|_, _| {})).await
}

pub async fn probe_local_with_progress(
    config: &super::config::PipelineConfig,
    target: &super::config::ProcessingTarget,
    progress: std::sync::Arc<dyn Fn(&str, Option<u8>) + Send + Sync>,
) -> Result<serde_json::Value> {
    let device = resolve_device(&target.device)?;
    let path = local_model_path(&config.speech_model)?;
    crate::speaker_diarization::model::set_models_dir();
    let segmentation = crate::speaker_diarization::model::pyannote_segmentation_path();
    let embedding = crate::speaker_diarization::model::default_model_path();
    let key = ProbeKey {
        model: config.speech_model.clone(), device: device.clone(),
        files: std::iter::once(&path).chain(segmentation.iter()).chain(embedding.iter())
            .map(|p| file_stamp(p)).collect(),
    };
    if let Some((_, report)) = PROBE_CACHE.lock().await.iter().find(|(cached_key, _)| *cached_key == key) {
        progress("Using the verified local model setup…", None);
        return Ok(report.clone());
    }
    progress("Checking speech model…", None);
    let _guard = match NATIVE_GATE.try_lock() {
        Ok(guard) => guard,
        Err(_) => {
            progress("Waiting for another local model check…", None);
            NATIVE_GATE.lock().await
        }
    };
    // Another editor may have completed the same check while this one waited.
    if let Some((_, report)) = PROBE_CACHE.lock().await.iter().find(|(cached_key, _)| *cached_key == key) {
        return Ok(report.clone());
    }
    if crate::whisper_engine::LIVE_ENGINE_LEASE.is_live_leased() {
        return Err(anyhow!(
            "Local live preview is using the native engine; check readiness after recording stops"
        ));
    }
    let ready = path.metadata().map(|m| m.len() > 0).unwrap_or(false);
    let speaker_assets = segmentation.as_ref().is_some_and(|p| p.is_file())
        && embedding.as_ref().is_some_and(|p| p.is_file());
    progress("Checking speaker models on this target…", None);
    let speaker_ready = if speaker_assets {
        let s = segmentation.clone().unwrap();
        let e = embedding.clone().unwrap();
        tokio::task::spawn_blocking(move || {
            crate::speaker_diarization::offline::diarize_offline(&vec![0.; 48000], &s, &e, 0, 1)
                .is_ok()
        })
        .await?
    } else {
        false
    };
    if !ready {
        return Ok(
            serde_json::json!({"engines":[{"id":"whisper_cpp","installed":true,"available":false,"verified":false,"device":device}],"speaker_pipelines":[{"id":"sherpa","installed":true,"available":speaker_ready,"verified":speaker_ready,"device":"cpu"}],"artifacts":[{"id":config.speech_model,"ready":false},{"id":"campplus","ready":speaker_assets},{"id":"sherpa-segmentation","ready":speaker_assets}],"resolved":{"speech_device":device,"speaker_device":"cpu"},"native_in_process":true}),
        );
    }
    progress("Loading speech model…", None);
    let engine = local_engine().await?;
    if PROBE_CACHE.lock().await.iter().any(|(old, _)|
        old.model == key.model && old.files.first() != key.files.first()) {
        engine.unload_model().await;
    }
    engine
        .load_model_on_device(&config.speech_model, &device)
        .await?;
    let device_label = match device.as_str() {
        "vulkan" => "Vulkan",
        "cuda" => "CUDA",
        "metal" => "Metal",
        _ => "CPU",
    };
    progress(
        &format!("Testing speech recognition on {device_label}…"),
        None,
    );
    let decoded = engine
        .transcribe_audio_with_confidence(vec![0.; 48000], Some("en".into()), None)
        .await;
    // Keep the verified model resident for the next real transcription.
    decoded?;
    let digest = super::runs::audio_hash_with_progress(path, move |percent| {
        progress("Reading speech model for integrity check", Some(percent));
    })
    .await?;
    let report = serde_json::json!({"protocol_version":1,"engines":[{"id":"whisper_cpp","installed":true,"available":true,"verified":true,"device":device,"model":config.speech_model,"compute_type":"whisper.cpp backend-native mixed precision"}],"speaker_pipelines":[{"id":"sherpa","installed":true,"available":speaker_ready,"verified":speaker_ready,"device":"cpu"}],"artifacts":[{"id":config.speech_model,"ready":true,"sha256":digest}],"resolved":{"speech_device":device,"speaker_device":"cpu"},"native_in_process":true});
    let mut cache = PROBE_CACHE.lock().await;
    cache.retain(|(old, _)| old.model != key.model || old.device != key.device);
    cache.push((key, report.clone()));
    Ok(report)
}

pub async fn probe_embedding_local() -> Result<bool> {
    let _guard = NATIVE_GATE.lock().await;
    crate::speaker_diarization::model::set_models_dir();
    let Some(model) =
        crate::speaker_diarization::model::default_model_path().filter(|p| p.is_file())
    else {
        return Ok(false);
    };
    tokio::task::spawn_blocking(move || {
        let Ok(embedder) = crate::speaker_diarization::SpeakerEmbedder::from_path(&model, 1) else {
            return false;
        };
        embedder
            .embed(&vec![0.; 48000])
            .is_ok_and(|v| !v.is_empty() && v.iter().all(|x| x.is_finite()))
    })
    .await
    .map_err(Into::into)
}

pub fn device_index(requested: &str) -> Result<i32> {
    #[cfg(target_os = "linux")]
    unsafe {
        let mut gpu_index = 0;
        for index in 0..whisper_rs::whisper_rs_sys::ggml_backend_dev_count() {
            let device = whisper_rs::whisper_rs_sys::ggml_backend_dev_get(index);
            let kind = whisper_rs::whisper_rs_sys::ggml_backend_dev_type(device);
            if kind==whisper_rs::whisper_rs_sys::ggml_backend_dev_type_GGML_BACKEND_DEVICE_TYPE_GPU||kind==whisper_rs::whisper_rs_sys::ggml_backend_dev_type_GGML_BACKEND_DEVICE_TYPE_IGPU {
    let reg=whisper_rs::whisper_rs_sys::ggml_backend_dev_backend_reg(device);let name=std::ffi::CStr::from_ptr(whisper_rs::whisper_rs_sys::ggml_backend_reg_name(reg)).to_string_lossy();if name.eq_ignore_ascii_case(requested){return Ok(gpu_index);}gpu_index+=1;
  }
        }
    }
    Err(anyhow!("Requested GPU device is unavailable"))
}

pub use speaker_core::campplus_space_id;

/// Local and remote speaker extraction consume the same normalized format,
/// avoiding incompatible spaces caused by different source decoders/resamplers.
fn normalize_speaker_audio(path: &Path) -> Result<Vec<f32>> {
    let ffmpeg = crate::audio::ffmpeg::find_ffmpeg_path()
        .ok_or_else(|| anyhow!("FFmpeg is unavailable for speaker preprocessing"))?;
    let temp = tempfile::Builder::new().suffix(".wav").tempfile()?;
    let status = std::process::Command::new(ffmpeg)
        .args(["-nostdin", "-loglevel", "error", "-y", "-i"])
        .arg(path)
        .args(["-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le"])
        .arg(temp.path())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()?;
    if !status.success() {
        return Err(anyhow!(
            "Could not normalize the recording for speaker analysis"
        ));
    }
    speaker_core::read_pcm16_wav(temp.path())
}

pub fn device_backend_at_index(wanted: i32) -> Option<String> {
    #[cfg(target_os = "linux")]
    unsafe {
        let mut gpu_index = 0;
        for index in 0..whisper_rs::whisper_rs_sys::ggml_backend_dev_count() {
            let device = whisper_rs::whisper_rs_sys::ggml_backend_dev_get(index);
            let kind = whisper_rs::whisper_rs_sys::ggml_backend_dev_type(device);
            if kind==whisper_rs::whisper_rs_sys::ggml_backend_dev_type_GGML_BACKEND_DEVICE_TYPE_GPU||kind==whisper_rs::whisper_rs_sys::ggml_backend_dev_type_GGML_BACKEND_DEVICE_TYPE_IGPU {if gpu_index==wanted{let reg=whisper_rs::whisper_rs_sys::ggml_backend_dev_backend_reg(device);return Some(std::ffi::CStr::from_ptr(whisper_rs::whisper_rs_sys::ggml_backend_reg_name(reg)).to_string_lossy().to_lowercase());}gpu_index+=1;}
        }
    }
    None
}

/// Speaker-only readiness does not inspect or load any speech model/device.
pub async fn probe_speakers_local() -> Result<bool> {
    let _guard = NATIVE_GATE.lock().await;
    crate::speaker_diarization::model::set_models_dir();
    let (Some(segmentation), Some(embedding)) = (
        crate::speaker_diarization::model::pyannote_segmentation_path(),
        crate::speaker_diarization::model::default_model_path(),
    ) else {
        return Ok(false);
    };
    if !segmentation.is_file() || !embedding.is_file() {
        return Ok(false);
    }
    tokio::task::spawn_blocking(move || {
        crate::speaker_diarization::offline::diarize_offline(
            &vec![0.; 48000],
            &segmentation,
            &embedding,
            0,
            1,
        )
        .is_ok()
    })
    .await
    .map_err(Into::into)
}

#[cfg(test)]
mod readiness_cache_tests {
    use super::*;
    #[test]
    fn replacement_and_missing_artifacts_invalidate_cached_checks() {
        let root = std::env::temp_dir().join(format!("parley-probe-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir(&root).unwrap();
        let path = root.join("model.bin");
        let missing = file_stamp(&path);
        std::fs::write(&path, b"first").unwrap();
        let original = file_stamp(&path);
        assert_ne!(missing, original);
        let replacement = root.join("replacement.bin");
        std::fs::write(&replacement, b"new model").unwrap();
        std::fs::rename(replacement, &path).unwrap();
        assert_ne!(original, file_stamp(&path));
        std::fs::remove_file(&path).unwrap();
        assert_eq!(missing, file_stamp(&path));
        std::fs::remove_dir(root).unwrap();
    }
}
