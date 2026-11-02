//! Orchestration. Adapters return data; repositories publish successful results.
const IDENTIFICATION_POLICY: &str =
    crate::speaker_diarization::matching_policy::IDENTIFICATION_POLICY_VERSION;
pub use super::preparation::{prepare_pipeline, prepare_speaker_pipeline};
use super::{config::*, runs, types::*};
use anyhow::{anyhow, bail, Result};
use serde::{Deserialize, Serialize};
use sqlx::{Row, SqliteConnection, SqlitePool};
use std::{
    path::PathBuf,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingOptions {
    pub config: PipelineConfig,
    pub language: Option<String>,
    pub num_speakers: i32,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingOutcome {
    pub run_id: String,
    pub segments_count: usize,
    pub identification_error: Option<String>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IdentificationOptions {
    pub identification: Identification,
    pub target_id: String,
    #[serde(default)]
    pub replace_confirmed: bool,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IdentitySuggestion {
    pub speaker: String,
    pub profile_id: Option<String>,
    pub name: Option<String>,
    pub score: Option<f32>,
    pub reason: Option<String>,
    pub confirmed: bool,
    #[serde(default)]
    pub current_profile_id: Option<String>,
    #[serde(default)]
    pub current_name: Option<String>,
    #[serde(default)]
    pub replacement: bool,
}
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct IdentityAssignment {
    pub profile_id: Option<String>,
    pub name: Option<String>,
}
#[derive(Default)]
struct ActiveIdentity {
    assignment: IdentityAssignment,
    protected: bool,
    ambiguous: bool,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IdentificationOutcome {
    pub run_id: String,
    pub suggestions: Vec<IdentitySuggestion>,
}

/// List installable options without loading or downloading any model.
pub async fn target_choices(pool: &SqlitePool, target_id: &str) -> Result<PipelineChoices> {
    let target = get_target(pool, target_id).await?;
    let choices = if target.kind == TargetKind::Local {
        local_choices()
    } else {
        choices_for_platform(&super::executor::platform_info(&target).await?)
    };
    if choices.engines.is_empty() {
        bail!("This target's operating system or architecture has no supported processing pipeline; choose a supported target");
    }
    Ok(choices)
}

pub async fn check_readiness(pool: &SqlitePool, config: &PipelineConfig) -> Result<Readiness> {
    check_readiness_with_progress(pool, config, |_, _| {}).await
}

/// Qualify the selected pipeline, reporting stages and measured integrity-read
/// percentages. A percentage describes its current stage, not overall readiness.
pub async fn check_readiness_with_progress(
    pool: &SqlitePool,
    config: &PipelineConfig,
    progress: impl Fn(&str, Option<u8>) + Send + Sync + 'static,
) -> Result<Readiness> {
    let progress = Arc::new(progress);
    progress("Checking target connection…", None);
    let target = get_target_for_config(pool, config).await?;
    let choices = target_choices(pool, &config.target_id).await;
    let (choices, platform_issue) = match choices {
        Ok(choices) => (choices, None),
        Err(error) => (PipelineChoices::default(), Some(error.to_string())),
    };
    let mut r = Readiness {
        engines: choices.engines,
        speaker_pipelines: choices.speaker_pipelines,
        models: speech_models()
            .into_iter()
            .filter(|m| m.engine == config.speech_engine && !m.experimental)
            .collect(),
        identifications: vec![
            Identification::Same,
            Identification::Off,
            Identification::Campplus,
            Identification::Wespeaker,
        ],
        ..Default::default()
    };
    if let Err(e) = validate_config(config) {
        r.issues.push(e.to_string());
    }
    if let Some(issue) = platform_issue {
        r.issues.push(issue);
        return Ok(r);
    }
    if !r.engines.contains(&config.speech_engine) {
        r.issues.push("Choose a speech engine compatible with this target's operating system and architecture".into());
        return Ok(r);
    }
    let probe =
        if target.kind == TargetKind::Local && config.speech_engine == SpeechEngine::WhisperCpp {
            let native =
                super::native::probe_local_with_progress(config, &target, progress.clone()).await;
            if config.speaker_pipeline == SpeakerPipeline::Community1 {
                let mut worker_target = target.clone();
                if worker_target.device == "auto" {
                    worker_target.device = super::native::resolve_device("auto")?;
                }
                progress("Checking speaker models on this target…", None);
                match (
                    native,
                    super::executor::probe_config(&worker_target, config).await,
                ) {
                    (Ok(mut local), Ok(remote)) => {
                        if let (Some(local_speakers), Some(remote_speakers)) = (
                            local["speaker_pipelines"].as_array_mut(),
                            remote["speaker_pipelines"].as_array(),
                        ) {
                            local_speakers.extend(
                                remote_speakers
                                    .iter()
                                    .filter(|p| p["id"] == "community1")
                                    .cloned(),
                            );
                        }
                        Ok(local)
                    }
                    (Ok(local), Err(_)) => Ok(local),
                    (Err(e), _) => Err(e),
                }
            } else {
                native
            }
        } else {
            progress("Checking speech and speaker models on this target…", None);
            super::executor::probe_config(&target, config).await
        };
    match probe {
        Ok(mut c) => {
            if target.kind == TargetKind::Local
                && config.speaker_pipeline == SpeakerPipeline::Sherpa
                && config.speech_engine != SpeechEngine::WhisperCpp
            {
                if let Err(error) = require_speech_only_operation(&c) {
                    r.issues.push(error.to_string());
                }

                progress("Checking speaker models on this target…", None);
                let ready = super::native::probe_speakers_local().await?;
                let mut speakers = c["speaker_pipelines"]
                    .as_array()
                    .cloned()
                    .unwrap_or_default();
                speakers.retain(|s| s["id"] != "sherpa");
                speakers.push(serde_json::json!({"id":"sherpa","installed":true,"available":ready,"verified":ready,"device":"cpu","registry":"app"}));
                c["speaker_pipelines"] = speakers.into();
            }
            if target.kind == TargetKind::Ssh && r.engines.is_empty() {
                let choices = choices_for_platform(&c["platform"]);
                r.engines = choices.engines;
                r.speaker_pipelines = choices.speaker_pipelines;
            }
            r.capabilities = c.clone();
            let engine_id = serde_json::to_value(config.speech_engine)?
                .as_str()
                .unwrap()
                .to_owned();
            let speaker_id = serde_json::to_value(config.speaker_pipeline)?
                .as_str()
                .unwrap()
                .to_owned();
            let speech_artifact_ready = c["artifacts"].as_array().is_some_and(|a| {
                a.iter()
                    .any(|a| a["id"] == config.speech_model && a["ready"] == true)
            });
            if !capability_available(&c, "engines", &engine_id) {
                r.issues.push(
                    if capability_installed(&c, "engines", &engine_id) {
                        if speech_artifact_ready {
                            "The selected speech model/device did not pass its execution check"
                        } else {
                            "Download the selected speech model to verify engine and device"
                        }
                    } else {
                        "Install the selected speech engine in target setup"
                    }
                    .into(),
                );
            }
            if !capability_available(&c, "speaker_pipelines", &speaker_id) {
                r.issues.push(
                    if capability_installed(&c, "speaker_pipelines", &speaker_id) {
                        "Download or verify the selected speaker package in target setup"
                    } else {
                        "Install the selected speaker pipeline in target setup"
                    }
                    .into(),
                );
            }
            r.speech_device = c["resolved"]["speech_device"].as_str().map(str::to_owned);
            r.speaker_device = c["resolved"]["speaker_device"].as_str().map(str::to_owned);
            r.compute_type = c["resolved"]["compute_type"].as_str().map(str::to_owned);
            let engine = serde_json::to_value(config.speech_engine)?
                .as_str()
                .unwrap()
                .to_owned();
            let speaker = serde_json::to_value(config.speaker_pipeline)?
                .as_str()
                .unwrap()
                .to_owned();
            if let Some(e) = c["engines"]
                .as_array()
                .and_then(|a| a.iter().find(|e| e["id"] == engine))
            {
                r.speech_device = e["device"].as_str().map(str::to_owned);
                r.compute_type = e["compute_type"].as_str().map(str::to_owned);
                if e["verified"] == true && e["model"] != config.speech_model {
                    r.issues.push(
                        "This exact model/device combination has not passed its execution check"
                            .into(),
                    );
                }
            }
            if let Some(s) = c["speaker_pipelines"]
                .as_array()
                .and_then(|a| a.iter().find(|e| e["id"] == speaker))
            {
                r.speaker_device = s["device"].as_str().map(str::to_owned);
            }
        }
        Err(e) => r.issues.push(e.to_string()),
    }
    if let Some(model) = additional_identification(config) {
        progress("Checking speaker identification…", None);
        match check_identification_readiness(pool, &config.target_id, model).await {
            Ok(identification) => include_identification_readiness(&mut r, model, &identification),
            Err(error) => r.issues.push(format!(
                "Prepare the selected independent identification model in target setup: {error}"
            )),
        }
    }
    r.ready = r.issues.is_empty();
    Ok(r)
}

pub(crate) fn additional_identification(config: &PipelineConfig) -> Option<Identification> {
    match (config.speaker_pipeline, config.identification) {
        (SpeakerPipeline::Sherpa, Identification::Wespeaker) => Some(Identification::Wespeaker),
        (SpeakerPipeline::Community1, Identification::Campplus) => Some(Identification::Campplus),
        _ => None,
    }
}
pub(crate) fn include_identification_readiness(
    report: &mut Readiness,
    selected: Identification,
    identification: &Readiness,
) {
    report.capabilities["identification"] = serde_json::json!({"model":selected,"ready":identification.ready,"device":identification.speaker_device,"capabilities":identification.capabilities});
    if !identification.ready {
        report.ready = false;
        if identification.issues.is_empty() {
            report.issues.push(
                "Prepare the selected independent identification model in target setup".into(),
            );
        } else {
            report.issues.extend(
                identification
                    .issues
                    .iter()
                    .map(|issue| format!("Independent identification: {issue}")),
            );
        }
    }
}
fn capability_installed(v: &serde_json::Value, key: &str, id: &str) -> bool {
    capability_available(v, key, id)
        || v[key]
            .as_array()
            .is_some_and(|a| a.iter().any(|e| e["id"] == id && e["installed"] == true))
}
fn capability_available(v: &serde_json::Value, key: &str, id: &str) -> bool {
    v[key].as_array().is_some_and(|a| {
        a.iter().any(|e| {
            e.as_str() == Some(id)
                || (e["id"] == id && e["available"] == true && e["verified"] == true)
        })
    }) || v[key][id] == true
        || (v[key][id]["available"] == true && v[key][id]["verified"] == true)
}
async fn audio_path(pool: &SqlitePool, meeting: &str) -> Result<PathBuf> {
    let folder: Option<String> = sqlx::query_scalar("SELECT folder_path FROM meetings WHERE id=?")
        .bind(meeting)
        .fetch_optional(pool)
        .await?
        .ok_or_else(|| anyhow!("Meeting no longer exists"))?;
    crate::audio::retranscription::find_audio_file(&PathBuf::from(
        folder.ok_or_else(|| anyhow!("This meeting has no saved recording"))?,
    ))
}
async fn execute_recording(
    pool: &SqlitePool,
    meeting: &str,
    options: &ProcessingOptions,
    operation: &str,
    request_id: &str,
    target_override: Option<&ProcessingTarget>,
    cancelled: &AtomicBool,
    progress: impl Fn(&str, u32, &str) + Send + Sync,
) -> Result<Analysis> {
    if operation == "process" {
        validate_config(&options.config)?;
    } else if options.config.version != 1 || options.config.target_id.trim().is_empty() {
        bail!("Invalid speaker analysis configuration");
    }
    if crate::audio::recording_service::current_meeting_id().as_deref() == Some(meeting) {
        bail!("Stop recording before processing this meeting");
    }
    if options.num_speakers < 0 || options.num_speakers > 64 {
        bail!("Expected speaker count must be between 0 and 64");
    }
    let target = match target_override {
        Some(original) => original.clone(),
        None => get_target_for_config(pool, &options.config).await?,
    };
    let path = audio_path(pool, meeting).await?;
    let hash = super::runs::audio_hash_with_progress(path.clone(), |percent| {
        progress("audio_integrity", percent.into(), "Reading the saved recording to verify its contents");
    }).await?;
    let segments:Vec<serde_json::Value>=sqlx::query("SELECT transcript,audio_start_time,audio_end_time,speaker FROM transcripts WHERE meeting_id=? ORDER BY sequence_id,audio_start_time").bind(meeting).fetch_all(pool).await?.into_iter().map(|r|serde_json::json!({"text":r.get::<String,_>("transcript"),"start":r.get::<Option<f64>,_>("audio_start_time").unwrap_or(0.),"end":r.get::<Option<f64>,_>("audio_end_time").unwrap_or(0.),"speaker":r.get::<Option<String>,_>("speaker")})).collect();
    let started_at = chrono::Utc::now().to_rfc3339();
    super::recovery::retain(
        pool,
        meeting,
        request_id,
        &target,
        options,
        operation,
        &started_at,
    )
    .await?;
    if cancelled.load(Ordering::SeqCst) {
        bail!("Processing cancelled; the saved recording was kept");
    }
    let request = serde_json::json!({"version":1,"operation":operation,"request_id":request_id,"config":options.config,"language":options.language,"num_speakers":options.num_speakers,"audio_path":path,"audio_sha256":hash,"segments":segments});
    let value = execute_audio(&target, &options.config, request, cancelled, &progress).await?;
    if cancelled.load(Ordering::SeqCst) {
        bail!("Processing cancelled; existing transcript kept");
    }
    let mut a: Analysis = serde_json::from_value(value.get("analysis").unwrap_or(&value).clone())?;
    validate_fresh_analysis(&a)?;
    if runs::audio_hash_with_progress(path.clone(), |percent| {
        progress("verifying", percent.into(), "Verifying the source recording has not changed");
    }).await? != hash {
        bail!("Source recording changed during processing; existing result was kept");
    }
    if a.audio_sha256
        .as_ref()
        .is_some_and(|returned| returned != &hash)
    {
        bail!("Worker result refers to a different source recording");
    }
    snapshot_request(
        &mut a,
        options,
        request_id,
        operation,
        &started_at,
        &hash,
        &target,
    )?;
    Ok(a)
}
async fn process_recording_inner(
    pool: &SqlitePool,
    meeting: &str,
    options: ProcessingOptions,
    request_id: &str,
    target_override: Option<&ProcessingTarget>,
    cancelled: &AtomicBool,
    progress: impl Fn(&str, u32, &str) + Send + Sync,
) -> Result<ProcessingOutcome> {
    let a = execute_recording(
        pool,
        meeting,
        &options,
        "process",
        request_id,
        target_override,
        cancelled,
        &progress,
    )
    .await?;
    progress("saving", 0, "Saving transcript and speaker groups");
    let segments = a.transcript_segments(None).await?;
    let count = segments.len();
    crate::database::repositories::transcript::TranscriptsRepository::replace_transcripts_with_analysis_cancel(pool,meeting,&segments,Some(&a),Some(cancelled)).await?;
    if options.config.identification != Identification::Off {
        progress("identifying", 0, "Matching speakers against enrolled voices");
    }
    let identification_error =
        post_publication_identification(pool, meeting, request_id, &options.config, cancelled, &progress)
            .await;
    if let Some(error) = &identification_error {
        progress("identification_failed", 0, error);
    }
    progress("finishing", 0, "Cleaning up worker files and exporting the transcript");
    if let Ok(context) = super::recovery::load(pool, meeting, request_id).await {
        let target = context.target;
        if target.kind == TargetKind::Ssh
            || options.config.speech_engine != SpeechEngine::WhisperCpp
            || options.config.speaker_pipeline != SpeakerPipeline::Sherpa
        {
            let _ = super::executor::cleanup(&target, request_id).await;
        }
    }
    if let Err(error) = runs::refresh_exports(pool, meeting).await {
        log::warn!("Committed transcript export could not be refreshed: {error}");
    }
    Ok(ProcessingOutcome {
        run_id: a.settings["request_id"].as_str().unwrap().into(),
        segments_count: count,
        identification_error,
    })
}
async fn post_publication_identification(
    pool: &SqlitePool,
    meeting: &str,
    request_id: &str,
    config: &PipelineConfig,
    cancelled: &AtomicBool,
    progress: &(impl Fn(&str, u32, &str) + Send + Sync),
) -> Option<String> {
    if config.identification == Identification::Off {
        return None;
    }
    let context = match super::recovery::load(pool, meeting, request_id).await {
        Ok(context) => context,
        Err(error) => return Some(error.to_string()),
    };
    identify_speakers_on_target(
        pool,
        meeting,
        IdentificationOptions {
            identification: config.identification,
            target_id: config.target_id.clone(),
            replace_confirmed: false,
        },
        Some(&context.target),
        cancelled,
        |_, percent, message| progress("identifying", percent, message),
    )
    .await
    .err()
    .map(|e| e.to_string())
}

async fn redetect_speakers_inner(
    pool: &SqlitePool,
    meeting: &str,
    options: ProcessingOptions,
    request_id: &str,
    target_override: Option<&ProcessingTarget>,
    cancelled: &AtomicBool,
    progress: impl Fn(&str, u32, &str) + Send + Sync,
) -> Result<ProcessingOutcome> {
    let a = execute_recording(
        pool,
        meeting,
        &options,
        "speaker_analysis",
        request_id,
        target_override,
        cancelled,
        progress,
    )
    .await?;
    publish_speaker_analysis(pool, meeting, a, cancelled).await
}
pub(crate) async fn publish_speaker_analysis(
    pool: &SqlitePool,
    meeting: &str,
    mut a: Analysis,
    cancelled: &AtomicBool,
) -> Result<ProcessingOutcome> {
    let mut tx = pool.begin().await?;
    let rows=sqlx::query("SELECT id,transcript,audio_start_time,audio_end_time,speaker,voice_profile_id,identity_origin FROM transcripts WHERE meeting_id=? ORDER BY sequence_id,audio_start_time").bind(meeting).fetch_all(&mut *tx).await?;
    // Apply new groups to the exact existing transcript rows. Recognition,
    // timing, row identifiers and overlap-capable turns remain unchanged.
    a.segments = rows
        .iter()
        .filter_map(|r| {
            Some(Segment {
                start: r.get::<Option<f64>, _>("audio_start_time")?,
                end: r.get::<Option<f64>, _>("audio_end_time")?,
                text: r.get("transcript"),
                speaker: None,
                overlapping_speakers: vec![],
            })
        })
        .collect();
    let previous: Option<String> = sqlx::query_scalar(
        "SELECT analysis_json FROM meeting_transcription_analysis WHERE meeting_id=?",
    )
    .bind(meeting)
    .fetch_optional(&mut *tx)
    .await?;
    let previous: Option<Analysis> = previous
        .map(|json| serde_json::from_str(&json))
        .transpose()?;
    let words = previous
        .as_ref()
        .map(|p| p.words.clone())
        .unwrap_or_default();
    a.words.clear();
    super::native::annotate(&mut a);
    a.words = words
        .into_iter()
        .map(|mut w| {
            w.speaker = None;
            w.overlapping_speakers.clear();
            w
        })
        .collect();
    for r in &rows {
        let start: Option<f64> = r.get("audio_start_time");
        let end: Option<f64> = r.get("audio_end_time");
        let label: Option<String> = r.get("speaker");
        let profile: Option<String> = r.get("voice_profile_id");
        // Legacy named rows may be human decisions with unknown provenance.
        // Preserve them until the user explicitly reconciles the new groups.
        if r.get::<String, _>("identity_origin") != MACHINE
            && (profile.is_some() || is_named(label.as_deref()))
        {
            continue;
        }
        if let Some(s) = a
            .segments
            .iter()
            .find(|s| Some(s.start) == start && Some(s.end) == end)
        {
            sqlx::query("UPDATE transcripts SET speaker=?,voice_profile_id=NULL,identity_origin='machine' WHERE id=?")
                .bind(&s.speaker)
                .bind(r.get::<String, _>("id"))
                .execute(&mut *tx)
                .await?;
        }
    }
    if cancelled.load(Ordering::SeqCst) {
        bail!("Speaker analysis cancelled; existing transcript kept");
    }
    runs::store_tx(&mut tx, meeting, Some(&a)).await?;
    tx.commit().await?;
    if let Err(error) = runs::refresh_exports(pool, meeting).await {
        log::warn!("Committed transcript export could not be refreshed: {error}");
    }
    Ok(ProcessingOutcome {
        run_id: a.settings["request_id"].as_str().unwrap().into(),
        segments_count: a.segments.len(),
        identification_error: None,
    })
}

async fn identify_speakers_inner(
    pool: &SqlitePool,
    meeting: &str,
    options: IdentificationOptions,
    target_override: Option<&ProcessingTarget>,
    cancelled: &AtomicBool,
    progress: impl Fn(&str, u32, &str) + Send + Sync,
) -> Result<IdentificationOutcome> {
    let mut analysis = runs::load(pool, meeting).await?.ok_or_else(|| {
        anyhow!("Re-detect speakers to create speaker groups before identification")
    })?;
    let processing_run = runs::current_run(pool, meeting).await?;
    let group_version = processing_run
        .clone()
        .unwrap_or_else(|| "legacy_unknown".into());
    let mut embedding_origins: std::collections::BTreeMap<(String, String), serde_json::Value> =
        std::collections::BTreeMap::new();
    for e in &analysis.embeddings {
        if let Some(space) = &e.embedding_space {
            embedding_origins.insert((e.speaker.clone(),space.id.clone()),serde_json::json!({"processing_run_id":processing_run,"runtime":analysis.provenance,"worker_settings":runs::redact_metadata(&analysis.settings),"target_snapshot":analysis.settings["target_snapshot"]}));
        }
    }
    let cached:Vec<(String,String)>=sqlx::query_as("SELECT embedding_json,provenance_json FROM speaker_group_embeddings WHERE meeting_id=? AND group_version=?").bind(meeting).bind(&group_version).fetch_all(pool).await?;
    for (json, metadata) in cached {
        let e: SpeakerEmbedding = serde_json::from_str(&json)?;
        if let Some(space) = &e.embedding_space {
            embedding_origins.insert((e.speaker.clone(),space.id.clone()),serde_json::json!({"processing_run_id":processing_run,"cached_extraction":serde_json::from_str::<serde_json::Value>(&metadata)?}));
        }
        if !analysis.embeddings.iter().any(|old| {
            old.speaker == e.speaker
                && old.embedding_space.as_ref().map(|s| &s.id)
                    == e.embedding_space.as_ref().map(|s| &s.id)
        }) {
            analysis.embeddings.push(e);
        }
    }
    let requested = options.identification;
    let pipeline = analysis.settings["config"]["speaker_pipeline"]
        .as_str()
        .unwrap_or(if analysis.settings["diarization"] == "community" {
            "community1"
        } else {
            "sherpa"
        });
    let model = match requested {
        Identification::Off => None,
        Identification::Campplus => Some("campplus"),
        Identification::Wespeaker => Some("wespeaker"),
        Identification::Same => Some(if pipeline == "community1" {
            "wespeaker"
        } else {
            "campplus"
        }),
    };
    let compatible =
        |e: &SpeakerEmbedding| model.is_some_and(|model| reference_embedding(e, model));
    let groups: std::collections::BTreeSet<String> =
        analysis.turns.iter().map(|t| t.speaker.clone()).collect();
    let needed = model
        .map(|model| missing_embedding_intervals(&analysis, model))
        .unwrap_or_default();
    let mut extraction_cleanup: Option<(ProcessingTarget, String)> = None;
    let mut extracted = false;
    let mut extraction_provenance = serde_json::Value::Null;
    let mut extraction_hash = analysis.settings["audio_sha256"]
        .as_str()
        .map(str::to_owned);
    if model.is_some() {
        let intervals = needed;
        if !intervals.is_empty() {
            let target = match target_override {
                Some(original) => original.clone(),
                None => get_target(pool, &options.target_id).await?,
            };
            let path = audio_path(pool, meeting).await?;
            let current_hash = super::runs::audio_hash(path.clone()).await?;
            if extraction_hash
                .as_ref()
                .is_some_and(|old| old != &current_hash)
            {
                bail!("Source recording has changed since speaker detection; re-detect speakers before identification");
            }
            extraction_hash = Some(current_hash.clone());
            let extraction_id = uuid::Uuid::new_v4().to_string();
            let request = serde_json::json!({"version":1,"operation":"extract_embeddings","request_id":extraction_id,"audio_path":path,"audio_sha256":current_hash,"duration":analysis.duration,"source_intervals":intervals,"embedding_model":model,"config":{"speaker_pipeline":pipeline,"identification":model,"target_id":options.target_id}});
            let value = extract_embeddings(&target, request, cancelled, progress).await?;
            let value = value.get("analysis").unwrap_or(&value);
            extraction_provenance = serde_json::json!({"provenance":value["provenance"],"worker_settings":value["settings"],"target":{"id":target.id,"name":target.name,"kind":target.kind},"request_id":extraction_id});
            let embeddings: Vec<SpeakerEmbedding> =
                serde_json::from_value(value["embeddings"].clone())?;
            for e in &embeddings {
                super::identification::validate_embedding(e)?;
                if let Some(space) = &e.embedding_space {
                    embedding_origins.insert(
                        (e.speaker.clone(), space.id.clone()),
                        extraction_provenance.clone(),
                    );
                }
                if !intervals.iter().any(|i| i.speaker == e.speaker)
                    || e.source_intervals.iter().any(|source| {
                        !intervals.iter().any(|allowed| {
                            source.speaker == allowed.speaker
                                && source.channel == allowed.channel
                                && source.start >= allowed.start - 0.0001
                                && source.end <= allowed.end + 0.0001
                        })
                    })
                {
                    bail!("Embedding extractor returned evidence outside the requested clean speaker intervals");
                }
            }
            if target.kind == TargetKind::Ssh || model != Some("campplus") {
                extraction_cleanup = Some((target, extraction_id));
            }
            analysis.embeddings.extend(embeddings);
            extracted = true;
        }
    }
    let candidates: Vec<_> = analysis
        .embeddings
        .iter()
        .filter(|e| compatible(e))
        .cloned()
        .collect();
    let enrollment_revisions:Vec<(String,String,String)>=sqlx::query_as("SELECT profile_id,space_id,revision FROM profile_enrollments WHERE status='ready' ORDER BY profile_id,space_id").fetch_all(pool).await?;
    let matching = super::identification::match_with_policy(pool, meeting, &candidates).await?;
    let matches = matching.matches;
    let run_id = uuid::Uuid::new_v4().to_string();
    let mut suggestions = vec![];
    let mut tx = pool.begin().await?;
    let current: Option<String> = sqlx::query_scalar(
        "SELECT current_run_id FROM meeting_processing_state WHERE meeting_id=?",
    )
    .bind(meeting)
    .fetch_optional(&mut *tx)
    .await?
    .flatten();
    if current != processing_run {
        bail!("Speaker groups changed during identification; run identification again");
    }
    let current_revisions:Vec<(String,String,String)>=sqlx::query_as("SELECT profile_id,space_id,revision FROM profile_enrollments WHERE status='ready' ORDER BY profile_id,space_id").fetch_all(&mut *tx).await?;
    if current_revisions != enrollment_revisions {
        bail!("Voice enrollment evidence changed during matching; identify speakers again");
    }
    if extracted {
        for e in &candidates {
            if let Some(space) = &e.embedding_space {
                sqlx::query("INSERT INTO speaker_group_embeddings VALUES(?,?,?,?,?,?) ON CONFLICT(meeting_id,group_version,speaker,space_id) DO UPDATE SET embedding_json=excluded.embedding_json,provenance_json=excluded.provenance_json").bind(meeting).bind(&group_version).bind(&e.speaker).bind(&space.id).bind(serde_json::to_string(e)?).bind(extraction_provenance.to_string()).execute(&mut *tx).await?;
            }
        }
    }
    for speaker in groups {
        let active = active_identity(&mut tx, meeting, &group_version, &speaker, &analysis).await?;
        let found = matches.get(&speaker);
        let alternative = options.replace_confirmed
            && active.protected
            && !active.ambiguous
            && found.is_some_and(|m| Some(m.0.as_str()) != active.assignment.profile_id.as_deref());
        let preserve = active.protected && !alternative;
        let (profile, name) = if preserve {
            (
                active.assignment.profile_id.clone(),
                active.assignment.name.clone(),
            )
        } else {
            (found.map(|m| m.0.clone()), found.map(|m| m.1.clone()))
        };
        let reason = if active.ambiguous {
            Some("This group contains different existing assignments; split or re-detect it before identifying".into())
        } else if alternative {
            Some(
                "Alternative proposal; the current identity remains until you confirm replacement"
                    .into(),
            )
        } else if preserve {
            Some("Existing confirmed or named assignment preserved".into())
        } else if model.is_none() {
            Some("Speaker identification is off".into())
        } else if found.is_none() {
            Some("No sufficiently strong, unambiguous evidence in this embedding space".into())
        } else {
            Some("Machine proposal; confirm after reviewing the source audio".into())
        };
        suggestions.push(IdentitySuggestion {
            speaker: speaker.clone(),
            profile_id: profile.clone(),
            name,
            score: None,
            reason,
            confirmed: preserve,
            current_profile_id: active.assignment.profile_id.clone(),
            current_name: active.assignment.name.clone(),
            replacement: alternative,
        });
        // Proposals have no authority to rewrite words, groups, displayed
        // labels, or an existing human decision. Acceptance is separate.
        if !active.protected {
            sqlx::query("INSERT INTO speaker_identity_decisions VALUES(?,?,?,?,0,?,datetime('now')) ON CONFLICT(meeting_id,group_version,speaker) DO UPDATE SET profile_id=excluded.profile_id,identification_run_id=excluded.identification_run_id,updated_at=excluded.updated_at WHERE speaker_identity_decisions.confirmed=0").bind(meeting).bind(&group_version).bind(&speaker).bind(profile).bind(&run_id).execute(&mut *tx).await?;
        }
    }
    let spaces: Vec<_> = candidates
        .iter()
        .filter_map(|e| e.embedding_space.as_ref().map(|s| s.id.clone()))
        .collect();
    let evidence_sources:Vec<_>=candidates.iter().filter_map(|e|e.embedding_space.as_ref().map(|space|serde_json::json!({"speaker":e.speaker,"embedding_space":space,"source_intervals":e.source_intervals,"provenance":embedding_origins.get(&(e.speaker.clone(),space.id.clone()))}))).collect();
    let provenance = serde_json::json!({"suggestions":suggestions,"embedding_spaces":spaces,"extracted":extracted,"extraction_provenance":extraction_provenance,"audio_sha256":extraction_hash,"extraction_target_id":if extracted{Some(&options.target_id)}else{None},"evidence_sources":evidence_sources,"enrollment_revisions":enrollment_revisions,"policy_qualification":"conservative heuristic; held-out calibration required","resolved_policy":matching.resolved_policy});
    sqlx::query("INSERT INTO identification_runs VALUES(?,?,?,?,?,?,?,?)")
        .bind(&run_id)
        .bind(meeting)
        .bind(&processing_run)
        .bind(&group_version)
        .bind(chrono::Utc::now().to_rfc3339())
        .bind(IDENTIFICATION_POLICY)
        .bind(serde_json::to_string(&options)?)
        .bind(provenance.to_string())
        .execute(&mut *tx)
        .await?;
    if cancelled.load(Ordering::SeqCst) {
        bail!("Speaker identification cancelled; existing assignments kept");
    }
    tx.commit().await?;
    if let Some((target, id)) = extraction_cleanup {
        let _ = super::executor::cleanup(&target, &id).await;
    }
    if let Err(error) = runs::refresh_exports(pool, meeting).await {
        log::warn!("Committed identity export could not be refreshed: {error}");
    }
    Ok(IdentificationOutcome {
        run_id,
        suggestions,
    })
}

pub fn clean_intervals(a: &Analysis) -> Vec<SourceInterval> {
    let turns: Vec<_> = a
        .turns
        .iter()
        .map(|t| SourceInterval {
            speaker: t.speaker.clone(),
            start: t.start,
            end: t.end,
            channel: 0,
        })
        .collect();
    let overlaps: Vec<_> = a
        .overlaps
        .iter()
        .map(|o| speaker_core::OverlapInterval {
            start: o.start,
            end: o.end,
            speakers: o.speakers.clone(),
        })
        .collect();
    speaker_core::clean_intervals(&turns, &overlaps)
}

/// Identification readiness deliberately does not depend on ASR availability.
/// `same` is resolved using the meeting's diarization provenance at execution.
pub async fn check_identification_readiness(
    pool: &SqlitePool,
    target_id: &str,
    identification: Identification,
) -> Result<Readiness> {
    let mut report = Readiness {
        identifications: vec![
            Identification::Same,
            Identification::Off,
            Identification::Campplus,
            Identification::Wespeaker,
        ],
        ..Default::default()
    };
    if identification == Identification::Off {
        report.ready = true;
        return Ok(report);
    }
    if identification == Identification::Same {
        report.issues.push("Resolve Same from the meeting's current speaker analysis before checking identification".into());
        return Ok(report);
    }
    let target = get_target(pool, target_id).await?;
    if target.kind == TargetKind::Local && identification == Identification::Campplus {
        report.ready = super::native::probe_embedding_local().await?;
        report.speaker_device = Some("cpu".into());
        report.capabilities =
            serde_json::json!({"native_in_process":true,"embedding_model":"campplus"});
        if !report.ready {
            report
                .issues
                .push("Download the local CAMPPlus embedding model before identification".into());
        }
        return Ok(report);
    }
    let model = if identification == Identification::Campplus {
        "campplus"
    } else {
        "wespeaker"
    };
    match super::executor::probe_identification(&target, model).await {
        Ok(capabilities) => {
            let selected = capabilities["embedding_models"].as_array()
                .and_then(|models| models.iter().find(|entry| entry["id"] == model));
            report.ready = selected.is_some_and(|entry| entry["available"] == true && entry["verified"] == true);
            report.speaker_device = selected.and_then(|entry| entry["device"].as_str()).map(str::to_owned);
            if !report.ready {
                report.issues.push(if selected.is_some_and(|entry| entry["installed"] == true) {
                    "Download or verify the selected embedding model in target setup"
                } else {
                    "Install or update the selected embedding extractor in target setup"
                }.into());
            }
            report.capabilities = capabilities;
        }
        Err(error) => report.issues.push(format!("Install or update the optional runtime and selected embedding model in target setup: {error}")),
    }
    Ok(report)
}

use super::attribution::{is_named, HUMAN, LEGACY_UNKNOWN, MACHINE};

pub async fn resume_processing(
    pool: &SqlitePool,
    meeting: &str,
    failed_run_id: &str,
    cancelled: &AtomicBool,
    progress: impl Fn(&str, u32, &str) + Send + Sync,
) -> Result<ProcessingOutcome> {
    let (requested, operation): (String, String) = sqlx::query_as(
        "SELECT requested_json,operation FROM processing_runs WHERE id=? AND meeting_id=?",
    )
    .bind(failed_run_id)
    .bind(meeting)
    .fetch_one(pool)
    .await?;
    let requested: serde_json::Value = serde_json::from_str(&requested)?;
    let request_id = requested["request_id"]
        .as_str()
        .ok_or_else(|| anyhow!("This historical run has no reconnectable worker request"))?;
    let published: bool = sqlx::query_scalar(
        "SELECT EXISTS(SELECT 1 FROM processing_runs WHERE id=? AND outcome='succeeded')",
    )
    .bind(request_id)
    .fetch_one(pool)
    .await?;
    if published {
        let n: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM transcripts WHERE meeting_id=?")
            .bind(meeting)
            .fetch_one(pool)
            .await?;
        return Ok(ProcessingOutcome {
            run_id: request_id.into(),
            segments_count: n as usize,
            identification_error: None,
        });
    }
    if requested["reconnectable"] == false {
        bail!("This run did not create a reconnectable worker job; start a new processing run using the saved recording");
    }
    let context = super::recovery::load(pool, meeting, request_id).await?;
    if context.operation != operation {
        bail!("Recovery operation differs from the original processing request");
    }
    let target = context.target;
    let options = context.options;
    let source = audio_path(pool, meeting).await?;
    let v = super::executor::resume_with_audio(&target, request_id, &source, cancelled, &progress)
        .await?;
    let mut a: Analysis = serde_json::from_value(v.get("analysis").unwrap_or(&v).clone())?;
    validate_fresh_analysis(&a)?;
    if a.request_id.as_deref() != Some(request_id) {
        bail!("Recovered result belongs to a different worker request");
    }
    let expected_hash = a.audio_sha256.as_deref().ok_or_else(|| {
        anyhow!("Recovered result has no source checksum; result was not imported")
    })?;
    let current_hash = runs::audio_hash(audio_path(pool, meeting).await?).await?;
    if expected_hash != current_hash {
        bail!("The source recording changed; recovered result was not imported");
    }
    snapshot_request(
        &mut a,
        &options,
        request_id,
        &operation,
        &context.started_at,
        &current_hash,
        &target,
    )?;
    if needs_local_sherpa_completion(&target, &options.config, &a) {
        complete_local_sherpa(&source, &mut a, options.num_speakers, cancelled, &progress).await?;
    }
    if operation == "speaker_analysis" {
        let outcome = publish_speaker_analysis(pool, meeting, a, cancelled).await?;
        let _ = super::executor::cleanup(&target, request_id).await;
        let _ = super::recovery::forget(pool, request_id).await;
        return Ok(outcome);
    }
    let segments = a.transcript_segments(None).await?;
    crate::database::repositories::transcript::TranscriptsRepository::replace_transcripts_with_analysis_cancel(pool,meeting,&segments,Some(&a),Some(cancelled)).await?;
    let identification_error =
        post_publication_identification(pool, meeting, request_id, &options.config, cancelled, &progress)
            .await;
    let _ = super::executor::cleanup(&target, request_id).await;
    let _ = super::recovery::forget(pool, request_id).await;
    if let Err(error) = runs::refresh_exports(pool, meeting).await {
        log::warn!("Committed transcript export could not be refreshed: {error}");
    }
    Ok(ProcessingOutcome {
        run_id: request_id.into(),
        segments_count: segments.len(),
        identification_error,
    })
}

pub async fn identify_speakers(
    pool: &SqlitePool,
    meeting: &str,
    options: IdentificationOptions,
    cancelled: &AtomicBool,
    progress: impl Fn(&str, u32, &str) + Send + Sync,
) -> Result<IdentificationOutcome> {
    identify_speakers_on_target(pool, meeting, options, None, cancelled, progress).await
}

async fn identify_speakers_on_target(
    pool: &SqlitePool,
    meeting: &str,
    options: IdentificationOptions,
    target_override: Option<&ProcessingTarget>,
    cancelled: &AtomicBool,
    progress: impl Fn(&str, u32, &str) + Send + Sync,
) -> Result<IdentificationOutcome> {
    let request = serde_json::to_string(&options)?;
    match identify_speakers_inner(pool, meeting, options, target_override, cancelled, progress)
        .await
    {
        Ok(r) => Ok(r),
        Err(e) => {
            let current = runs::current_run(pool, meeting).await?;
            sqlx::query("INSERT INTO identification_runs VALUES(?,?,?,?,?,?,?,?)").bind(uuid::Uuid::new_v4().to_string()).bind(meeting).bind(&current).bind(&current).bind(chrono::Utc::now().to_rfc3339()).bind(IDENTIFICATION_POLICY).bind(request).bind(serde_json::json!({"outcome":if cancelled.load(Ordering::SeqCst){"cancelled"}else{"failed"},"error":e.to_string(),"resolved_policy":super::identification::policy_snapshot()}).to_string()).execute(pool).await?;
            Err(e)
        }
    }
}

/// Reconcile human labels within the same SQLite snapshot that publishes the
/// replacement. A concurrent human decision either appears here or makes a
/// stale read transaction fail its write upgrade; it cannot be silently lost.
pub(crate) async fn preserve_confirmed_labels_tx(
    connection: &mut SqliteConnection,
    meeting: &str,
    segments: &mut [crate::audio::common::TranscriptSegment],
) -> Result<Vec<String>> {
    let previous: Option<String> = sqlx::query_scalar(
        "SELECT analysis_json FROM meeting_transcription_analysis WHERE meeting_id=?",
    )
    .bind(meeting)
    .fetch_optional(&mut *connection)
    .await?;
    let previous: Option<Analysis> = previous
        .map(|json| serde_json::from_str(&json))
        .transpose()?;
    let current: Option<String> = sqlx::query_scalar(
        "SELECT current_run_id FROM meeting_processing_state WHERE meeting_id=?",
    )
    .bind(meeting)
    .fetch_optional(&mut *connection)
    .await?
    .flatten();
    let rows: Vec<(Option<f64>,Option<f64>,Option<String>,Option<String>,String)> = sqlx::query_as("SELECT audio_start_time,audio_end_time,speaker,voice_profile_id,identity_origin FROM transcripts WHERE meeting_id=?")
        .bind(meeting).fetch_all(&mut *connection).await?;
    let decisions: Vec<(String,Option<String>,bool)> = sqlx::query_as("SELECT speaker,profile_id,confirmed FROM speaker_identity_decisions WHERE meeting_id=? AND group_version=?")
        .bind(meeting).bind(current).fetch_all(&mut *connection).await?;
    let mut protected = vec![];
    for (start, end, label, id, origin) in rows {
        if origin == MACHINE {
            continue;
        }
        if id.is_none() && !is_named(label.as_deref()) {
            continue;
        }
        let group = previous
            .as_ref()
            .and_then(|a| {
                a.segments
                    .iter()
                    .find(|s| Some(s.start) == start && Some(s.end) == end)
            })
            .and_then(|s| s.speaker.as_ref());
        if origin != HUMAN
            && group.is_some_and(|group| {
                decisions.iter().any(|(speaker, profile, confirmed)| {
                    speaker == group && !*confirmed && profile == &id
                })
            })
        {
            continue;
        }
        let (Some(start), Some(end)) = (start, end) else {
            bail!("An existing human speaker assignment has no audio timing; reconcile that assignment before reprocessing");
        };
        if !start.is_finite() || !end.is_finite() || end <= start {
            bail!("An existing human speaker assignment has invalid audio timing; reconcile that assignment before reprocessing");
        }
        protected.push((start, end, label, id, origin));
    }
    let mut origins = vec![MACHINE.to_owned(); segments.len()];
    for (index, segment) in segments.iter_mut().enumerate() {
        let (Some(start), Some(end)) = (segment.audio_start_time, segment.audio_end_time) else {
            if !protected.is_empty() {
                bail!("The replacement has no audio timing to preserve existing human speaker assignments");
            }
            continue;
        };
        let overlapping: Vec<_> = protected
            .iter()
            .filter(|(s, e, _, _, _)| *s < end && *e > start)
            .collect();
        if let Some(first) = overlapping.first() {
            if overlapping.iter().any(|r| r.2 != first.2 || r.3 != first.3) {
                bail!("The new transcript spans multiple human-confirmed speakers; reconcile or split those assignments before replacing this result");
            }
            segment.speaker = first.2.clone();
            segment.voice_profile_id = first.3.clone();
            origins[index] = if overlapping.iter().any(|row| row.4 == HUMAN) {
                HUMAN
            } else {
                LEGACY_UNKNOWN
            }
            .into();
        }
    }
    Ok(origins)
}

/// Accept an ordinary proposal. Replacing an existing protected assignment
/// requires the replacement API with its current-identity snapshot.
pub async fn confirm_identity_suggestion(
    pool: &SqlitePool,
    meeting: &str,
    expected_group_version: &str,
    speaker: &str,
    profile: &str,
) -> Result<()> {
    confirm_identity(
        pool,
        meeting,
        expected_group_version,
        speaker,
        profile,
        None,
    )
    .await
}
pub async fn confirm_identity_replacement(
    pool: &SqlitePool,
    meeting: &str,
    expected_group_version: &str,
    speaker: &str,
    profile: &str,
    expected_current: IdentityAssignment,
) -> Result<()> {
    confirm_identity(
        pool,
        meeting,
        expected_group_version,
        speaker,
        profile,
        Some(expected_current),
    )
    .await
}
async fn confirm_identity(
    pool: &SqlitePool,
    meeting: &str,
    expected_group_version: &str,
    speaker: &str,
    profile: &str,
    expected_current: Option<IdentityAssignment>,
) -> Result<()> {
    let mut tx = pool.begin().await?;
    let current: Option<String> = sqlx::query_scalar(
        "SELECT current_run_id FROM meeting_processing_state WHERE meeting_id=?",
    )
    .bind(meeting)
    .fetch_optional(&mut *tx)
    .await?
    .flatten();
    if current.as_deref() != Some(expected_group_version) {
        bail!("Speaker groups changed; reload this suggestion before confirmation");
    }
    let json: Option<String> = sqlx::query_scalar(
        "SELECT analysis_json FROM meeting_transcription_analysis WHERE meeting_id=?",
    )
    .bind(meeting)
    .fetch_optional(&mut *tx)
    .await?;
    let analysis: Analysis =
        serde_json::from_str(&json.ok_or_else(|| anyhow!("No current speaker analysis"))?)?;
    if !analysis.turns.iter().any(|t| t.speaker == speaker) {
        bail!("This speaker group no longer exists");
    }
    let active =
        active_identity(&mut tx, meeting, expected_group_version, speaker, &analysis).await?;
    if active.ambiguous {
        bail!("This group has different existing identities; split or re-detect it before assigning one identity");
    }
    if let Some(expected) = &expected_current {
        if expected != &active.assignment {
            bail!("The current speaker identity changed; review the replacement again");
        }
    } else if active.protected && active.assignment.profile_id.as_deref() != Some(profile) {
        bail!("This speaker already has a protected identity; review and confirm an explicit replacement");
    }
    let name: String = sqlx::query_scalar(
        "SELECT CASE WHEN is_self=1 THEN 'Me' ELSE name END FROM voice_profiles WHERE id=?",
    )
    .bind(profile)
    .fetch_optional(&mut *tx)
    .await?
    .ok_or_else(|| anyhow!("Voice profile no longer exists"))?;
    if let Some(previous) = active.assignment.profile_id.as_deref().filter(|id| *id != profile) {
        let intervals: Vec<_> = analysis.segments.iter()
            .filter(|s| s.speaker.as_deref() == Some(speaker))
            .map(|s| (s.start, s.end)).collect();
        super::enrollment::revoke_evidence_tx(&mut tx, previous, meeting, &intervals).await?;
    }
    sqlx::query("INSERT INTO speaker_identity_decisions VALUES(?,?,?,?,1,NULL,datetime('now')) ON CONFLICT(meeting_id,group_version,speaker) DO UPDATE SET profile_id=excluded.profile_id,confirmed=1,updated_at=excluded.updated_at").bind(meeting).bind(expected_group_version).bind(speaker).bind(profile).execute(&mut *tx).await?;
    for s in analysis
        .segments
        .iter()
        .filter(|s| s.speaker.as_deref() == Some(speaker))
    {
        sqlx::query("UPDATE transcripts SET speaker=?,voice_profile_id=?,identity_origin='human' WHERE meeting_id=? AND audio_start_time=? AND audio_end_time=?").bind(&name).bind(profile).bind(meeting).bind(s.start).bind(s.end).execute(&mut *tx).await?;
    }
    sqlx::query("INSERT INTO identity_decision_events VALUES(?,?,?,?,?,'confirmed',datetime('now'),?)").bind(uuid::Uuid::new_v4().to_string()).bind(meeting).bind(expected_group_version).bind(speaker).bind(profile).bind(serde_json::json!({"source":if expected_current.is_some(){"explicit_identity_replacement"}else{"explicit_suggestion_acceptance"},"profile_name_snapshot":name,"previous_assignment":active.assignment}).to_string()).execute(&mut *tx).await?;
    // Label and learn atomically, using only current human-confirmed clean evidence.
    super::enrollment::recover_confirmed_evidence_tx(&mut tx, profile).await?;
    tx.commit().await?;
    if let Err(error) = runs::refresh_exports(pool, meeting).await {
        log::warn!("Committed identity export could not be refreshed: {error}");
    }
    Ok(())
}

/// Return reviewable proposals alongside the current protected assignment.
/// The immutable run stays unchanged when a human later accepts a proposal.
pub async fn current_identity_suggestions(
    pool: &SqlitePool,
    meeting: &str,
) -> Result<Vec<IdentitySuggestion>> {
    let mut tx = pool.begin().await?;
    let version: Option<String> = sqlx::query_scalar(
        "SELECT current_run_id FROM meeting_processing_state WHERE meeting_id=?",
    )
    .bind(meeting)
    .fetch_optional(&mut *tx)
    .await?
    .flatten();
    let Some(version) = version else {
        return Ok(vec![]);
    };
    let analysis: Option<String> = sqlx::query_scalar(
        "SELECT analysis_json FROM meeting_transcription_analysis WHERE meeting_id=?",
    )
    .bind(meeting)
    .fetch_optional(&mut *tx)
    .await?;
    let Some(analysis) = analysis else {
        return Ok(vec![]);
    };
    let analysis: Analysis = serde_json::from_str(&analysis)?;
    let json:Option<String>=sqlx::query_scalar("SELECT result_json FROM identification_runs WHERE meeting_id=? AND group_version=? AND json_type(result_json,'$.suggestions')='array' ORDER BY created_at DESC,id DESC LIMIT 1").bind(meeting).bind(&version).fetch_optional(&mut *tx).await?;
    let mut suggestions: Vec<IdentitySuggestion> = if let Some(json) = json {
        let v: serde_json::Value = serde_json::from_str(&json)?;
        serde_json::from_value(v["suggestions"].clone())?
    } else {
        vec![]
    };
    let groups: std::collections::BTreeSet<_> =
        analysis.turns.iter().map(|t| t.speaker.clone()).collect();
    for speaker in groups {
        let active = active_identity(&mut tx, meeting, &version, &speaker, &analysis).await?;
        if !active.protected {
            continue;
        }
        if let Some(proposal) = suggestions.iter_mut().find(|s| s.speaker == speaker) {
            let expected = IdentityAssignment {
                profile_id: proposal.current_profile_id.clone(),
                name: proposal.current_name.clone(),
            };
            if proposal.replacement
                && expected == active.assignment
                && !active.ambiguous
                && proposal.profile_id != active.assignment.profile_id
            {
                continue;
            }
            *proposal = confirmed_view(speaker, active);
        } else {
            suggestions.push(confirmed_view(speaker, active));
        }
    }
    Ok(suggestions)
}
fn confirmed_view(speaker: String, active: ActiveIdentity) -> IdentitySuggestion {
    IdentitySuggestion {
        speaker,
        profile_id: active.assignment.profile_id.clone(),
        name: active.assignment.name.clone(),
        score: None,
        reason: Some(
            if active.ambiguous {
                "This group contains different assignments; split or re-detect it"
            } else {
                "Existing confirmed or named assignment"
            }
            .into(),
        ),
        confirmed: true,
        current_profile_id: active.assignment.profile_id,
        current_name: active.assignment.name,
        replacement: false,
    }
}
async fn active_identity(
    conn: &mut SqliteConnection,
    meeting: &str,
    version: &str,
    speaker: &str,
    analysis: &Analysis,
) -> Result<ActiveIdentity> {
    let decision:Option<(Option<String>,bool)>=sqlx::query_as("SELECT profile_id,confirmed FROM speaker_identity_decisions WHERE meeting_id=? AND group_version=? AND speaker=?").bind(meeting).bind(version).bind(speaker).fetch_optional(&mut *conn).await?;
    let mut named = vec![];
    let mut human_named = false;
    let mut any_machine = false;
    let mut any_authoritative = false;
    for s in analysis
        .segments
        .iter()
        .filter(|s| s.speaker.as_deref() == Some(speaker))
    {
        let rows:Vec<(Option<String>,Option<String>,String)>=sqlx::query_as("SELECT speaker,voice_profile_id,identity_origin FROM transcripts WHERE meeting_id=? AND audio_start_time=? AND audio_end_time=?").bind(meeting).bind(s.start).bind(s.end).fetch_all(&mut *conn).await?;
        for (name, id, origin) in rows {
            any_machine |= origin == MACHINE;
            any_authoritative |= origin != MACHINE;
            if origin != MACHINE && (id.is_some() || is_named(name.as_deref())) {
                human_named |= origin == HUMAN;
                named.push((name, id));
            }
        }
    }
    let identities: std::collections::BTreeSet<_> = named
        .iter()
        .map(|(name, id)| (id.clone(), if id.is_some() { None } else { name.clone() }))
        .collect();
    if let Some((profile, true)) = decision.clone().filter(|_| any_authoritative) {
        let fallback = named.first().and_then(|(name, _)| name.clone());
        let name = if let Some(profile) = &profile {
            sqlx::query_scalar(
                "SELECT CASE WHEN is_self=1 THEN 'Me' ELSE name END FROM voice_profiles WHERE id=?",
            )
            .bind(profile)
            .fetch_optional(&mut *conn)
            .await?
            .or(fallback)
        } else {
            fallback
        };
        let ambiguous =
            identities.len() > 1 || named.iter().any(|(_, id)| id != &profile) || any_machine;
        return Ok(ActiveIdentity {
            assignment: IdentityAssignment {
                profile_id: profile,
                name,
            },
            protected: true,
            ambiguous,
        });
    }
    if decision.is_none() || human_named {
        if let Some((name, profile)) = named.first() {
            return Ok(ActiveIdentity {
                assignment: IdentityAssignment {
                    profile_id: profile.clone(),
                    name: name.clone(),
                },
                protected: true,
                ambiguous: identities.len() > 1 || any_machine,
            });
        }
    }
    Ok(ActiveIdentity::default())
}

/// Native local jobs use the linked engine directly. Optional stages still
/// execute on the selected target, with no remote streaming or model fallback.
pub(crate) async fn execute_audio(
    target: &ProcessingTarget,
    config: &PipelineConfig,
    mut request: serde_json::Value,
    cancelled: &AtomicBool,
    progress: impl Fn(&str, u32, &str) + Send + Sync,
) -> Result<serde_json::Value> {
    progress("runtime", 0, "Checking the selected processing devices");
    snapshot_stage_devices(target, config, &mut request).await?;
    let operation = request["operation"].as_str().unwrap_or("process");
    if operation == "process"
        && target.kind == TargetKind::Local
        && config.speech_engine != SpeechEngine::WhisperCpp
        && config.speaker_pipeline == SpeakerPipeline::Sherpa
    {
        // Protocol v1 is additive: old immutable worker snapshots may lack
        // this operation. Reject before creating/uploading an unsupported job.
        let capabilities = super::executor::probe_config(target, config).await?;
        require_speech_only_operation(&capabilities)?;
        let source = PathBuf::from(
            request["audio_path"]
                .as_str()
                .ok_or_else(|| anyhow!("Missing source recording"))?,
        );
        let count = request["num_speakers"].as_i64().unwrap_or(0) as i32;
        let mut speech_request = request;
        speech_request["operation"] = "speech_recognition".into();
        let v = super::executor::execute(target, speech_request, cancelled, &progress).await?;
        let mut a: Analysis = serde_json::from_value(v.get("analysis").unwrap_or(&v).clone())?;
        complete_local_sherpa(&source, &mut a, count, cancelled, &progress).await?;
        return Ok(serde_json::to_value(a)?);
    }
    if target.kind != TargetKind::Local
        || (operation != "speaker_analysis" && config.speech_engine != SpeechEngine::WhisperCpp)
        || (operation == "speaker_analysis" && config.speaker_pipeline != SpeakerPipeline::Sherpa)
    {
        return super::executor::execute(target, request, cancelled, progress).await;
    }
    progress("queued", 0, "Waiting for the local processing engine");
    let _guard = tokio::select! {guard=super::native::NATIVE_GATE.lock()=>guard,_=async{while !cancelled.load(Ordering::SeqCst){tokio::time::sleep(std::time::Duration::from_millis(100)).await;}}=>bail!("Processing cancelled while waiting for the native engine")};
    if operation == "process" && crate::whisper_engine::LIVE_ENGINE_LEASE.is_live_leased() {
        bail!("Local live preview is active; process saved audio after recording stops");
    }
    let path = std::path::Path::new(
        request["audio_path"]
            .as_str()
            .ok_or_else(|| anyhow!("Missing source recording"))?,
    );
    let language = request["language"].as_str().map(str::to_owned);
    let count = request["num_speakers"].as_i64().unwrap_or(0) as i32;
    let mut native = if operation == "process" {
        super::native::recognize_local(
            path,
            &config.speech_model,
            language,
            request["speech_device"].as_str().unwrap_or(&target.device),
            cancelled,
            &progress,
        )
        .await?
    } else {
        let mut a = super::native::local_analysis(path, language).await?;
        a.segments = serde_json::from_value(request["segments"].clone())?;
        a
    };
    if cancelled.load(Ordering::SeqCst) {
        bail!("Processing cancelled; existing transcript kept");
    }
    if config.speaker_pipeline == SpeakerPipeline::Sherpa {
        progress(
            "speakers",
            80,
            "Detecting speakers with the local Sherpa pipeline",
        );
        crate::speaker_diarization::model::set_models_dir();
        super::native::native_speakers(path, &mut native, count).await?;
        native.provenance["speaker_pipeline"] =
            serde_json::json!({"id":"sherpa","version":"parley-sherpa-v1","device":"cpu"});
        Ok(serde_json::to_value(native)?)
    } else {
        let mut speaker_request = request;
        speaker_request["operation"] = "speaker_analysis".into();
        speaker_request["segments"] = serde_json::to_value(&native.segments)?;
        speaker_request["words"] = serde_json::to_value(&native.words)?;
        speaker_request["duration"] = native.duration.into();
        speaker_request["upstream_provenance"] = native.provenance.clone();
        speaker_request["upstream_recognition_device"] =
            serde_json::to_value(&native.recognition_device)?;
        speaker_request["upstream_compute_type"] = serde_json::to_value(&native.compute_type)?;
        speaker_request["upstream_recognition_seconds"] = native
            .recognition_seconds
            .unwrap_or(native.elapsed_seconds)
            .into();
        let mut worker_target = target.clone();
        if worker_target.device == "auto" {
            worker_target.device = super::native::resolve_device("auto")?;
        }
        progress("speakers", 0, "Preparing speaker detection on the selected target");
        let v =
            super::executor::execute(&worker_target, speaker_request, cancelled, progress).await?;
        let mut a: Analysis = serde_json::from_value(v.get("analysis").unwrap_or(&v).clone())?;
        a.recognition_device = native.recognition_device;
        a.compute_type = native.compute_type;
        a.provenance["speech"] = native.provenance;
        // Worker includes upstream timing once, including recovery results.
        Ok(serde_json::to_value(a)?)
    }
}

pub(crate) async fn extract_embeddings(
    target: &ProcessingTarget,
    mut request: serde_json::Value,
    cancelled: &AtomicBool,
    progress: impl Fn(&str, u32, &str) + Send + Sync,
) -> Result<serde_json::Value> {
    if request["embedding_model"] == "campplus" {
        // Independent identification resolves the extractor's device, not
        // the grouping pipeline's device (e.g. CUDA Community -> CPU CAMPPlus).
        request["speaker_device"] = "cpu".into();
    }
    if target.kind == TargetKind::Local && request["embedding_model"] == "campplus" {
        let _guard = tokio::select! {guard=super::native::NATIVE_GATE.lock()=>guard,_=async{while !cancelled.load(Ordering::SeqCst){tokio::time::sleep(std::time::Duration::from_millis(100)).await;}}=>bail!("Embedding extraction cancelled")};
        if cancelled.load(Ordering::SeqCst) {
            bail!("Embedding extraction cancelled");
        }
        crate::speaker_diarization::model::set_models_dir();
        let model = crate::speaker_diarization::model::default_model_path()
            .ok_or_else(|| anyhow!("Local speaker model unavailable"))?;
        let intervals: Vec<SourceInterval> =
            serde_json::from_value(request["source_intervals"].clone())?;
        let path = std::path::Path::new(
            request["audio_path"]
                .as_str()
                .ok_or_else(|| anyhow!("Missing source recording"))?,
        );
        progress("embedding", 10, "Extracting local CAMPPlus voice evidence");
        let embeddings = super::native::extract_native_embeddings(path, &model, &intervals).await?;
        if cancelled.load(Ordering::SeqCst) {
            bail!("Embedding extraction cancelled");
        }
        return Ok(
            serde_json::json!({"embeddings":embeddings,"provenance":{"native_runtime":speaker_core::runtime_info(),"device":"cpu","extraction_version":"parley-native-v2","reference":"native package embedder"}}),
        );
    }
    let mut resolved = target.clone();
    if request["embedding_model"] == "wespeaker" {
        let probe = super::executor::probe_identification(target, "wespeaker").await?;
        let speaker = verified_embedding_device(&probe, "wespeaker")?;
        request["speaker_device"] = speaker.clone().into();
        if resolved.device == "auto" {
            resolved.device = if speaker == "mps" {
                "metal".into()
            } else {
                speaker
            };
        }
    } else if resolved.device == "auto" {
        resolved.device = "cpu".into();
    }
    super::executor::execute(&resolved, request, cancelled, progress).await
}

pub async fn process_recording(
    pool: &SqlitePool,
    meeting: &str,
    options: ProcessingOptions,
    cancelled: &AtomicBool,
    progress: impl Fn(&str, u32, &str) + Send + Sync,
) -> Result<ProcessingOutcome> {
    processing_operation(pool, meeting, options, "process", None, cancelled, progress).await
}
/// Import selects its execution target before copying potentially large media.
pub(crate) async fn process_recording_on_target(
    pool: &SqlitePool,
    meeting: &str,
    options: ProcessingOptions,
    target: ProcessingTarget,
    cancelled: &AtomicBool,
    progress: impl Fn(&str, u32, &str) + Send + Sync,
) -> Result<ProcessingOutcome> {
    processing_operation(
        pool,
        meeting,
        options,
        "process",
        Some(target),
        cancelled,
        progress,
    )
    .await
}

pub async fn redetect_speakers(
    pool: &SqlitePool,
    meeting: &str,
    options: ProcessingOptions,
    cancelled: &AtomicBool,
    progress: impl Fn(&str, u32, &str) + Send + Sync,
) -> Result<ProcessingOutcome> {
    processing_operation(
        pool,
        meeting,
        options,
        "speaker_analysis",
        None,
        cancelled,
        progress,
    )
    .await
}
async fn processing_operation(
    pool: &SqlitePool,
    meeting: &str,
    options: ProcessingOptions,
    operation: &str,
    target_override: Option<ProcessingTarget>,
    cancelled: &AtomicBool,
    progress: impl Fn(&str, u32, &str) + Send + Sync,
) -> Result<ProcessingOutcome> {
    let request_id = uuid::Uuid::new_v4().to_string();
    let started = chrono::Utc::now().to_rfc3339();
    let mut requested =
        serde_json::json!({"options":options,"request_id":request_id,"started_at":started});
    let worker_created = AtomicBool::new(false);
    let observed_progress = |stage: &str, percent: u32, message: &str| {
        // Saved-audio jobs announce upload only after the owned job's Create
        // response was acknowledged. Native processing and setup probes do
        // not announce this through the main job's progress callback.
        if stage == "uploading" {
            worker_created.store(true, Ordering::SeqCst);
        }
        progress(stage, percent, message);
    };
    let result = if operation == "process" {
        process_recording_inner(
            pool,
            meeting,
            options,
            &request_id,
            target_override.as_ref(),
            cancelled,
            observed_progress,
        )
        .await
    } else {
        redetect_speakers_inner(
            pool,
            meeting,
            options,
            &request_id,
            target_override.as_ref(),
            cancelled,
            observed_progress,
        )
        .await
    };
    requested["reconnectable"] = worker_created.load(Ordering::SeqCst).into();
    if let Err(e) = &result {
        runs::record_failure(
            pool,
            meeting,
            operation,
            &requested,
            &e.to_string(),
            cancelled.load(Ordering::SeqCst),
        )
        .await?;
    }
    if result.is_ok() {
        let _ = super::recovery::forget(pool, &request_id).await;
    }
    result
}

pub async fn check_identification_for_meeting(
    pool: &SqlitePool,
    meeting: &str,
    target_id: &str,
    choice: Identification,
) -> Result<Readiness> {
    let analysis = runs::load(pool, meeting).await?;
    if analysis.is_none() {
        return Ok(Readiness {
            ready: false,
            issues: vec![
                "Re-detect speakers to create current groups before identification".into(),
            ],
            identifications: vec![choice],
            ..Default::default()
        });
    }

    let resolved = if choice == Identification::Same {
        if analysis.as_ref().is_some_and(|a| {
            a.settings["config"]["speaker_pipeline"] == "community1"
                || a.settings["diarization"] == "community"
        }) {
            Identification::Wespeaker
        } else {
            Identification::Campplus
        }
    } else {
        choice
    };
    let model = match resolved {
        Identification::Campplus => "campplus",
        Identification::Wespeaker => "wespeaker",
        _ => "",
    };
    if let Some(mut a) = analysis {
        let cached:Vec<String>=sqlx::query_scalar("SELECT embedding_json FROM speaker_group_embeddings e JOIN meeting_processing_state s ON s.meeting_id=e.meeting_id AND s.group_version=e.group_version WHERE e.meeting_id=?").bind(meeting).fetch_all(pool).await?;
        a.embeddings.extend(
            cached
                .iter()
                .filter_map(|j| serde_json::from_str::<SpeakerEmbedding>(j).ok()),
        );
        if choice == Identification::Off || missing_embedding_intervals(&a, model).is_empty() {
            return Ok(Readiness {
                ready: true,
                identifications: vec![choice],
                capabilities: serde_json::json!({"inference_required":false}),
                ..Default::default()
            });
        }
    }
    check_identification_readiness(pool, target_id, resolved).await
}

fn reference_embedding(e: &SpeakerEmbedding, model: &str) -> bool {
    let Some(s) = &e.embedding_space else {
        return false;
    };
    if e.source_intervals.is_empty() || s.dimension != e.vector.len() {
        return false;
    }
    match model {
        "campplus" => {
            s.model == "campplus"
                && s.id.starts_with("campplus:sha256:")
                && s.preprocessing == "pcm16k-mono-ffmpeg-v1"
                && s.normalization == "l2"
                && s.aggregation == "longest-three-trimmed-10s-v1"
                && s.extraction_version == "parley-native-v2"
        }
        "wespeaker" => {
            (s.model == "wespeaker" || s.model == "community1/bundled-wespeaker")
                && s.id.starts_with("wespeaker:")
                && s.preprocessing == "pcm16k-mono:package-reference-mask-v1"
                && s.normalization == "l2"
                && s.aggregation == "longest-three-clean-10s-duration-mean-v1"
                && s.extraction_version == "pyannote-4.0.4-clean-v1"
        }
        _ => false,
    }
}

/// Analyze an external saved recording without opening a meeting or profile
/// database. Import owns its atomic publication after this adapter succeeds.
pub async fn analyze_audio(
    target: &ProcessingTarget,
    options: &ProcessingOptions,
    path: &std::path::Path,
    cancelled: &AtomicBool,
    progress: impl Fn(&str, u32, &str) + Send + Sync,
) -> Result<Analysis> {
    validate_config(&options.config)?;
    if !(0..=64).contains(&options.num_speakers) {
        bail!("Expected speaker count must be Auto or 1–64");
    }
    let request_id = uuid::Uuid::new_v4().to_string();
    let started = chrono::Utc::now().to_rfc3339();
    let hash = runs::audio_hash(path.to_owned()).await?;
    let request = serde_json::json!({"version":1,"operation":"process","request_id":request_id,"config":options.config,"language":options.language,"num_speakers":options.num_speakers,"audio_path":path,"audio_sha256":hash,"segments":[]});
    let value = execute_audio(target, &options.config, request, cancelled, progress).await?;
    if cancelled.load(Ordering::SeqCst) {
        bail!("Import cancelled; no result was published");
    }
    let mut a: Analysis = serde_json::from_value(value.get("analysis").unwrap_or(&value).clone())?;
    validate_fresh_analysis(&a)?;
    if runs::audio_hash(path.to_owned()).await? != hash {
        bail!("Imported source changed during processing; no result was published");
    }
    if a.audio_sha256
        .as_ref()
        .is_some_and(|returned| returned != &hash)
        || a.request_id
            .as_ref()
            .is_some_and(|returned| returned != &request_id)
    {
        bail!("Worker result belongs to a different recording or request");
    }
    snapshot_request(
        &mut a,
        options,
        &request_id,
        "process",
        &started,
        &hash,
        target,
    )?;
    Ok(a)
}

pub(crate) fn snapshot_request(
    a: &mut Analysis,
    options: &ProcessingOptions,
    request_id: &str,
    operation: &str,
    started_at: &str,
    audio_sha256: &str,
    target: &ProcessingTarget,
) -> Result<()> {
    a.audio_sha256 = Some(audio_sha256.into());
    a.request_id = Some(request_id.into());
    a.settings["config"] = serde_json::to_value(&options.config)?;
    a.settings["requested_language"] = serde_json::to_value(&options.language)?;
    a.settings["num_speakers"] = options.num_speakers.into();
    a.settings["request_id"] = request_id.into();
    a.settings["started_at"] = started_at.into();
    a.settings["operation"] = operation.into();
    a.settings["audio_sha256"] = audio_sha256.into();
    a.settings["target_snapshot"] =
        serde_json::json!({"id":target.id,"name":target.name,"kind":target.kind});
    Ok(())
}

fn verified_embedding_device(probe: &serde_json::Value, model: &str) -> Result<String> {
    probe["embedding_models"].as_array()
        .and_then(|entries| entries.iter().find(|entry| entry["id"] == model && entry["available"] == true && entry["verified"] == true))
        .and_then(|entry| entry["device"].as_str()).filter(|device| !["auto", ""].contains(device)).map(str::to_owned)
        .ok_or_else(|| anyhow!("Install or update the selected embedding adapter and verify its model/device in target setup"))
}

fn verified_speaker_device(probe: &serde_json::Value, pipeline: &str) -> Result<String> {
    probe["speaker_pipelines"]
        .as_array()
        .and_then(|entries| {
            entries
                .iter()
                .find(|e| e["id"] == pipeline && e["available"] == true && e["verified"] == true)
        })
        .and_then(|e| e["device"].as_str())
        .map(str::to_owned)
        .ok_or_else(|| {
            anyhow!(
                "The selected speaker pipeline has not passed its execution check on this target"
            )
        })
}
async fn snapshot_stage_devices(
    target: &ProcessingTarget,
    config: &PipelineConfig,
    request: &mut serde_json::Value,
) -> Result<()> {
    let speech = if request["operation"] == "speaker_analysis" {
        if target.device == "auto" {
            "cpu".to_owned()
        } else {
            target.device.clone()
        }
    } else if target.kind == TargetKind::Local && config.speech_engine == SpeechEngine::WhisperCpp {
        super::native::resolve_device(&target.device)?
    } else if target.device == "auto" {
        let probe = super::executor::probe_config(target, config).await?;
        verified_speech_device(&probe, config)?
    } else {
        target.device.clone()
    };
    request["speech_device"] = speech.clone().into();
    let speaker = match config.speaker_pipeline {
        SpeakerPipeline::Sherpa => "cpu".to_owned(),
        SpeakerPipeline::Community1 if speech == "metal" => {
            let probe = super::executor::probe_config(target, config).await?;
            verified_speaker_device(&probe, "community1")?
        }
        SpeakerPipeline::Community1 if speech == "cuda" => "cuda".to_owned(),
        SpeakerPipeline::Community1 => "cpu".to_owned(),
    };
    request["speaker_device"] = speaker.into();
    Ok(())
}

pub(crate) fn missing_embedding_intervals(a: &Analysis, model: &str) -> Vec<SourceInterval> {
    let covered: std::collections::BTreeSet<_> = a
        .embeddings
        .iter()
        .filter(|e| reference_embedding(e, model))
        .map(|e| e.speaker.as_str())
        .collect();
    clean_intervals(a)
        .into_iter()
        .filter(|i| !covered.contains(i.speaker.as_str()))
        .collect()
}

fn validate_fresh_analysis(a: &Analysis) -> Result<()> {
    a.validate()?;
    if a.embeddings
        .iter()
        .any(|e| e.embedding_space.is_none() || e.source_intervals.is_empty())
    {
        bail!(
            "A new processing result must identify the exact embedding space and source evidence"
        );
    }
    Ok(())
}

pub async fn check_speaker_analysis_readiness(
    pool: &SqlitePool,
    target_id: &str,
    pipeline: SpeakerPipeline,
    _identification: Identification,
) -> Result<Readiness> {
    let mut target = get_target(pool, target_id).await?;
    if target.kind == TargetKind::Local && pipeline == SpeakerPipeline::Sherpa {
        let ready = super::native::probe_speakers_local().await?;
        return Ok(Readiness {
            ready,
            issues: if ready {
                vec![]
            } else {
                vec!["Download or verify the local Sherpa speaker package".into()]
            },
            speaker_pipelines: vec![pipeline],
            identifications: vec![Identification::Off],
            speaker_device: Some("cpu".into()),
            capabilities: serde_json::json!({"native_in_process":true,"operation":"speaker_analysis"}),
            ..Default::default()
        });
    }
    if target.kind == TargetKind::Local && target.device == "auto" {
        target.device = "cpu".into();
    }
    let probe = super::executor::probe(&target).await?;
    let id = match pipeline {
        SpeakerPipeline::Sherpa => "sherpa",
        SpeakerPipeline::Community1 => "community1",
    };
    let ready = capability_available(&probe, "speaker_pipelines", id);
    let device = verified_speaker_device(&probe, id).ok();
    Ok(Readiness {
        ready,
        issues: if ready {
            vec![]
        } else {
            vec!["Install, download or verify the selected speaker package in target setup".into()]
        },
        speaker_pipelines: if capability_installed(&probe, "speaker_pipelines", id) {
            vec![pipeline]
        } else {
            vec![]
        },
        identifications: vec![Identification::Off],
        speaker_device: device,
        capabilities: probe,
        ..Default::default()
    })
}

pub(crate) fn verified_speech_device(
    probe: &serde_json::Value,
    config: &PipelineConfig,
) -> Result<String> {
    let engine = serde_json::to_value(config.speech_engine)?
        .as_str()
        .unwrap()
        .to_owned();
    let device = probe["engines"]
        .as_array()
        .and_then(|entries| {
            entries.iter().find(|e| {
                e["id"] == engine
                    && e["model"] == config.speech_model
                    && e["available"] == true
                    && e["verified"] == true
            })
        })
        .and_then(|e| e["device"].as_str())
        .filter(|d| !["auto", ""].contains(d))
        .ok_or_else(|| {
            anyhow!("The selected speech model/device has not passed target setup verification")
        })?;
    Ok(device.into())
}

pub(crate) fn require_speech_only_operation(capabilities: &serde_json::Value) -> Result<()> {
    if !capabilities["operations"]
        .as_array()
        .is_some_and(|ops| ops.iter().any(|op| op == "speech_recognition"))
    {
        bail!("Install or update this target's worker to support speech-only local processing");
    }
    Ok(())
}

pub(crate) fn needs_local_sherpa_completion(
    target: &ProcessingTarget,
    config: &PipelineConfig,
    analysis: &Analysis,
) -> bool {
    target.kind == TargetKind::Local
        && config.speech_engine != SpeechEngine::WhisperCpp
        && config.speaker_pipeline == SpeakerPipeline::Sherpa
        && analysis.settings["worker_operation"] == "speech_recognition"
        && analysis.settings["local_sherpa_completed"] != true
}

async fn complete_local_sherpa(
    source: &std::path::Path,
    a: &mut Analysis,
    count: i32,
    cancelled: &AtomicBool,
    progress: &(impl Fn(&str, u32, &str) + Send + Sync),
) -> Result<()> {
    let _guard = tokio::select! {guard=super::native::NATIVE_GATE.lock()=>guard,_=async{while !cancelled.load(Ordering::SeqCst){tokio::time::sleep(std::time::Duration::from_millis(100)).await;}}=>bail!("Speaker analysis cancelled")};
    let started = std::time::Instant::now();
    progress(
        "speakers",
        80,
        "Detecting speakers with the linked local Sherpa package",
    );
    crate::speaker_diarization::model::set_models_dir();
    super::native::native_speakers(source, a, count).await?;
    if cancelled.load(Ordering::SeqCst) {
        bail!("Speaker analysis cancelled; existing result kept");
    }
    let seconds = started.elapsed().as_secs_f64();
    a.diarization_seconds = Some(seconds);
    a.elapsed_seconds += seconds;
    a.settings["local_sherpa_completed"] = true.into();
    Ok(())
}
