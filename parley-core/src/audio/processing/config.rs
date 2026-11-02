//! Stable pipeline choices, target setup, and explicit legacy mapping.
use crate::database::repositories::setting::SettingsRepository;
use anyhow::{bail, Result};
use serde::{Deserialize, Serialize};
use sqlx::SqlitePool;
static TARGET_EDIT_GATE: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

pub const NATIVE_EMBEDDING: &str = "campplus:3dspeaker_speech_campplus_sv_en_voxceleb_16k.onnx";
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SpeechEngine {
    WhisperCpp,
    FasterWhisper,
    MlxWhisper,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SpeakerPipeline {
    Sherpa,
    Community1,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Identification {
    Same,
    Off,
    Campplus,
    Wespeaker,
}
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(default)]
pub struct PipelineConfig {
    pub version: u32,
    pub target_id: String,
    pub speech_engine: SpeechEngine,
    pub speech_model: String,
    pub speaker_pipeline: SpeakerPipeline,
    pub identification: Identification,
}
impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            version: 1,
            target_id: "local".into(),
            speech_engine: SpeechEngine::WhisperCpp,
            speech_model: "large-v3-turbo-q5_0".into(),
            speaker_pipeline: SpeakerPipeline::Sherpa,
            identification: Identification::Same,
        }
    }
}
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TargetKind {
    Local,
    Ssh,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ProcessingTarget {
    pub id: String,
    pub name: String,
    pub kind: TargetKind,
    pub ssh_host: Option<String>,
    pub worker_path: Option<String>,
    pub python: Option<String>,
    pub device: String,
    /// Historical optional-runtime device. It never overrides native GPU selection.
    pub legacy_python_device: Option<String>,
    pub assets_dir: Option<String>,
}
impl Default for ProcessingTarget {
    fn default() -> Self {
        Self {
            id: "local".into(),
            name: "Local".into(),
            kind: TargetKind::Local,
            ssh_host: None,
            worker_path: None,
            python: None,
            device: "auto".into(),
            legacy_python_device: None,
            assets_dir: None,
        }
    }
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeechModel {
    pub id: String,
    pub label: String,
    pub engine: SpeechEngine,
    pub checkpoint: String,
    pub stored_weights: String,
    pub experimental: bool,
}
pub fn speech_models() -> Vec<SpeechModel> {
    [
        (
            SpeechEngine::WhisperCpp,
            "large-v3-turbo",
            "Floating-point weights",
            false,
        ),
        (
            SpeechEngine::WhisperCpp,
            "large-v3-turbo-q5_0",
            "Q5_0",
            false,
        ),
        (
            SpeechEngine::FasterWhisper,
            "large-v3-turbo-fp16",
            "FP16 weights",
            false,
        ),
        (
            SpeechEngine::FasterWhisper,
            "large-v3-turbo-int8",
            "INT8 weights",
            false,
        ),
        (
            SpeechEngine::MlxWhisper,
            "large-v3-turbo-mlx-fp16",
            "FP16 weights",
            false,
        ),
        (
            SpeechEngine::MlxWhisper,
            "large-v3-turbo-mlx-q4",
            "Q4 weights — qualification required",
            true,
        ),
    ]
    .into_iter()
    .map(|(engine, id, weights, experimental)| SpeechModel {
        id: id.into(),
        label: format!("Whisper large-v3-turbo — {weights}"),
        engine,
        checkpoint: "large-v3-turbo".into(),
        stored_weights: weights.into(),
        experimental,
    })
    .collect()
}
/// Architecture-compatible choices, including optional packages not installed yet.
/// Actual model/device qualification belongs to Readiness.capabilities.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PipelineChoices {
    pub engines: Vec<SpeechEngine>,
    pub speaker_pipelines: Vec<SpeakerPipeline>,
    pub identifications: Vec<Identification>,
}
pub fn local_choices() -> PipelineChoices {
    let system = match std::env::consts::OS {
        "macos" => "Darwin",
        "linux" => "Linux",
        other => other,
    };
    choices_for_platform(
        &serde_json::json!({"system":system,"architecture":std::env::consts::ARCH}),
    )
}
pub fn choices_for_platform(platform: &serde_json::Value) -> PipelineChoices {
    let system = platform["system"].as_str().unwrap_or_default();
    let architecture = platform["architecture"].as_str().unwrap_or_default();
    if !matches!(system, "Linux" | "linux" | "Darwin" | "darwin" | "macos")
        || !matches!(architecture, "x86_64" | "amd64" | "arm64" | "aarch64")
    {
        return PipelineChoices::default();
    }
    let mut engines = vec![SpeechEngine::WhisperCpp, SpeechEngine::FasterWhisper];
    if matches!(system, "Darwin" | "darwin" | "macos")
        && matches!(architecture, "arm64" | "aarch64")
    {
        engines.push(SpeechEngine::MlxWhisper);
    }
    PipelineChoices {
        engines,
        speaker_pipelines: vec![SpeakerPipeline::Sherpa, SpeakerPipeline::Community1],
        identifications: vec![
            Identification::Same,
            Identification::Off,
            Identification::Campplus,
            Identification::Wespeaker,
        ],
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Readiness {
    pub ready: bool,
    pub issues: Vec<String>,
    pub engines: Vec<SpeechEngine>,
    pub models: Vec<SpeechModel>,
    pub speaker_pipelines: Vec<SpeakerPipeline>,
    pub identifications: Vec<Identification>,
    pub speech_device: Option<String>,
    pub speaker_device: Option<String>,
    pub compute_type: Option<String>,
    pub capabilities: serde_json::Value,
}
pub async fn load_config(pool: &SqlitePool) -> Result<PipelineConfig> {
    if let Some(v) = SettingsRepository::get_setting_json(pool, "processing_pipeline_v1").await? {
        return Ok(serde_json::from_str(&v)?);
    }
    let mut c = PipelineConfig::default();
    if let Some(legacy) = SettingsRepository::get_transcript_config(pool).await? {
        if legacy.provider == "localWhisper" && !legacy.model.is_empty() {
            c.speech_model = legacy.model;
        }
    }
    if let Some(v) = SettingsRepository::get_setting_json(pool, "ui_config").await? {
        let v: serde_json::Value = serde_json::from_str(&v)?;
        if v["offlineRecognizer"] == "faster" {
            c.speech_engine = SpeechEngine::FasterWhisper;
            c.speech_model = if v["offlineModel"] == "fast" {
                "large-v3-turbo-int8"
            } else {
                "large-v3-turbo-fp16"
            }
            .into();
        }
        if v["offlineDiarization"] == "community" {
            c.speaker_pipeline = SpeakerPipeline::Community1;
        }
    }
    Ok(c)
}
pub async fn save_config(pool: &SqlitePool, c: &PipelineConfig) -> Result<()> {
    validate_config(c)?;
    set(pool, "processing_pipeline_v1", c).await
}
pub fn validate_config(c: &PipelineConfig) -> Result<()> {
    if c.version != 1 || c.target_id.trim().is_empty() {
        bail!("Unsupported pipeline version or missing target");
    }
    if !speech_models()
        .iter()
        .any(|m| m.engine == c.speech_engine && m.id == c.speech_model && !m.experimental)
    {
        bail!("Choose a qualified model compatible with the speech engine");
    }
    Ok(())
}
pub async fn list_targets(pool: &SqlitePool) -> Result<Vec<ProcessingTarget>> {
    let mut targets: Vec<ProcessingTarget> =
        SettingsRepository::get_setting_json(pool, "processing_targets_v1")
            .await?
            .map(|v| serde_json::from_str(&v))
            .transpose()?
            .unwrap_or_default();
    if !targets.iter().any(|t| t.id == "local") {
        let mut local = ProcessingTarget::default();
        if let Some(v) = SettingsRepository::get_setting_json(pool, "ui_config").await? {
            let v: serde_json::Value = serde_json::from_str(&v)?;
            local.python = v["offlinePython"]
                .as_str()
                .filter(|s| !s.is_empty())
                .map(str::to_owned);
            local.assets_dir = v["offlineAssets"]
                .as_str()
                .filter(|s| !s.is_empty())
                .map(str::to_owned);
            if let Some(device) = v["offlineDevice"].as_str().filter(|s| !s.is_empty()) {
                local.legacy_python_device = Some(device.into());
            }
        }
        targets.insert(0, local);
    }
    Ok(targets)
}
pub async fn get_target(pool: &SqlitePool, id: &str) -> Result<ProcessingTarget> {
    list_targets(pool)
        .await?
        .into_iter()
        .find(|t| t.id == id)
        .ok_or_else(|| anyhow::anyhow!("Processing target is no longer available"))
}
pub async fn save_target(pool: &SqlitePool, target: &ProcessingTarget) -> Result<()> {
    let _guard = TARGET_EDIT_GATE.lock().await;
    if target.id.trim().is_empty() || target.name.trim().is_empty() {
        bail!("Target needs an identifier and display name");
    }
    if target.kind == TargetKind::Ssh
        && target.ssh_host.as_ref().is_none_or(|h| {
            h.trim().is_empty() || h.starts_with('-') || h.contains(char::is_whitespace)
        })
    {
        bail!("Enter one OpenSSH host alias");
    }
    if target.id == "local" && target.kind != TargetKind::Local {
        bail!("Local target cannot become a remote target");
    }
    let mut targets = list_targets(pool).await?;
    targets.retain(|t| t.id != target.id);
    targets.push(target.clone());
    set(pool, "processing_targets_v1", &targets).await
}
pub async fn delete_target(pool: &SqlitePool, id: &str) -> Result<()> {
    let _guard = TARGET_EDIT_GATE.lock().await;
    if id == "local" {
        bail!("The local target cannot be removed");
    }
    let mut ts = list_targets(pool).await?;
    ts.retain(|t| t.id != id);
    set(pool, "processing_targets_v1", &ts).await
}
async fn set<T: Serialize>(pool: &SqlitePool, key: &str, value: &T) -> Result<()> {
    sqlx::query("INSERT INTO app_settings(key,value,updated_at) VALUES(?,?,datetime('now')) ON CONFLICT(key) DO UPDATE SET value=excluded.value,updated_at=excluded.updated_at").bind(key).bind(serde_json::to_string(value)?).execute(pool).await?;
    Ok(())
}

pub async fn get_target_for_config(
    pool: &SqlitePool,
    config: &PipelineConfig,
) -> Result<ProcessingTarget> {
    let mut target = get_target(pool, &config.target_id).await?;
    if target.kind == TargetKind::Local
        && target.device == "auto"
        && config.speech_engine != SpeechEngine::WhisperCpp
    {
        if let Some(device) = &target.legacy_python_device {
            target.device = device.clone();
        }
    }
    Ok(target)
}
