//! Shared pipeline editor. Defaults and meeting overrides use the same validated choices.
use crate::{app_state::AppServices, runtime::Io};
use gpui_kit::base::StyledExt as _;
use gpui_kit::component::{
    button::{Button, ButtonVariants as _},
    h_flex,
    input::{Input, InputState},
    menu::{DropdownMenu as _, PopupMenuItem},
    progress::Progress,
    spinner::Spinner,
    v_flex, ActiveTheme, Disableable as _, Sizable as _,
};
use gpui_kit::prelude::FluentBuilder as _;
use gpui_kit::*;
use parley_core::audio::processing::config::{
    self, Identification, PipelineConfig, ProcessingTarget, Readiness, SpeakerPipeline,
    SpeechEngine, TargetKind,
};

static NEXT_TARGET_OPERATION_ID: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(1);

/// A view of the existing default-pipeline check, not a separate startup check.
pub struct PipelineStatus {
    pub target: String,
    pub message: String,
    pub busy: bool,
    pub ready: bool,
    pub percent: Option<u8>,
}

pub fn engine_label(value: SpeechEngine) -> &'static str {
    match value {
        SpeechEngine::WhisperCpp => "whisper.cpp",
        SpeechEngine::FasterWhisper => "faster-whisper",
        SpeechEngine::MlxWhisper => "MLX Whisper",
    }
}
pub fn speaker_label(value: SpeakerPipeline) -> &'static str {
    match value {
        SpeakerPipeline::Sherpa => "Sherpa-ONNX",
        SpeakerPipeline::Community1 => "Community-1",
    }
}
pub fn identification_label(value: Identification) -> &'static str {
    match value {
        Identification::Same => "Same as speaker pipeline",
        Identification::Off => "Off",
        Identification::Campplus => "CAMPPlus (Sherpa)",
        Identification::Wespeaker => "WeSpeaker (Community-1)",
    }
}

pub struct PipelineEditor {
    pub config: PipelineConfig,
    pub readiness: Option<Readiness>,
    pub targets: Vec<ProcessingTarget>,
    pub loading: bool,
    pub error: Option<String>,
    readiness_progress: Option<ReadinessProgress>,
    choices: Option<(String, config::PipelineChoices)>,
    defaults: bool,
    identification_only: bool,
    speaker_analysis_only: bool,
    meeting_id: Option<String>,
    generation: u64,
    save_revision: std::sync::Arc<std::sync::atomic::AtomicU64>,
    save_gate: std::sync::Arc<tokio::sync::Mutex<()>>,
    managing: bool,
    editing_target: Option<ProcessingTarget>,
    target_name: Entity<InputState>,
    target_host: Entity<InputState>,
    target_worker: Entity<InputState>,
    target_python: Entity<InputState>,
    target_assets: Entity<InputState>,
    target_device: Entity<InputState>,
    native_bundle: Entity<InputState>,
    target_busy: bool,
    target_message: Option<String>,
    target_operation_id: Option<String>,
    target_retry: Option<(ProcessingTarget, &'static str)>,
}
impl PipelineEditor {
    pub fn status(&self) -> PipelineStatus {
        let busy = self.loading || self.target_busy;
        let message = if self.target_busy {
            self.target_message.clone().unwrap_or_else(|| "Preparing target…".into())
        } else if self.loading {
            self.readiness_progress.as_ref().map(|progress| {
                readiness_progress_status(
                    &progress.stage, progress.percent, progress.started_at.elapsed().as_secs(),
                )
            }).unwrap_or_else(|| "Loading pipeline settings…".into())
        } else if let Some(error) = &self.error {
            format!("Needs attention: {error}")
        } else if self.ready() {
            "Ready to transcribe".into()
        } else if let Some(report) = &self.readiness {
            format!("Needs attention: {}", report.issues.join(" · "))
        } else {
            "Not checked — open Transcription settings".into()
        };
        PipelineStatus {
            target: self.targets.iter().find(|target| target.id == self.config.target_id)
                .map(|target| target.name.clone()).unwrap_or_else(|| "Selected target".into()),
            message,
            busy,
            ready: self.ready(),
            percent: if self.loading && !self.target_busy {
                self.readiness_progress.as_ref().and_then(|progress| progress.percent)
            } else {
                None
            },
        }
    }

    pub fn new(defaults: bool, window: &mut Window, cx: &mut Context<Self>) -> Self {
        let mut this = Self {
            config: PipelineConfig::default(),
            readiness: None,
            targets: vec![],
            loading: true,
            error: None,
            readiness_progress: None,
            choices: None,
            defaults,
            identification_only: false,
            speaker_analysis_only: false,
            meeting_id: None,
            generation: 0,
            save_revision: std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0)),
            save_gate: std::sync::Arc::new(tokio::sync::Mutex::new(())),
            managing: false,
            editing_target: None,
            target_name: cx.new(|cx| InputState::new(window, cx).placeholder("Display name")),
            target_host: cx.new(|cx| InputState::new(window, cx).placeholder("SSH host alias")),
            target_worker: cx.new(|cx| InputState::new(window, cx).placeholder("Worker location")),
            target_python: cx
                .new(|cx| InputState::new(window, cx).placeholder("Python executable (optional)")),
            target_assets: cx.new(|cx| {
                InputState::new(window, cx).placeholder("Model assets folder (optional)")
            }),
            target_device: cx.new(|cx| {
                InputState::new(window, cx).placeholder("Device: cpu, vulkan, cuda or metal")
            }),
            native_bundle: cx.new(|cx| {
                InputState::new(window, cx)
                    .placeholder("Optional local .tar or .tar.gz helper bundle")
            }),
            target_busy: false,
            target_message: None,
            target_operation_id: None,
            target_retry: None,
        };
        let events = AppServices::global(cx).core_events.clone();
        cx.subscribe(&events, |this, _, event, cx| {
            if event.name == "processing-readiness-progress"
                && readiness_progress_is_current(
                    this.loading,
                    this.generation,
                    this.readiness_progress.as_ref().map(|progress| progress.operation_id.as_str()),
                    event.payload["generation"].as_u64(),
                    event.payload["operation_id"].as_str(),
                )
            {
                if let Some(progress) = &mut this.readiness_progress {
                    progress.stage = event.payload["message"]
                        .as_str()
                        .unwrap_or("Checking selected pipeline availability…")
                        .to_owned();
                    progress.percent = event.payload["percent"]
                        .as_u64()
                        .map(|percent| percent.min(100) as u8);
                    cx.notify();
                }
            }
            if event.name == "processing-target-progress"
                && this.target_busy
                && event.payload["operation_id"].as_str() == this.target_operation_id.as_deref()
            {
                let percent = event.payload["percent"].as_u64().unwrap_or(0).min(100);
                let message = event.payload["message"]
                    .as_str()
                    .unwrap_or("Preparing target");
                this.target_message = Some(format!("{percent}% · {message}"));
                cx.notify();
            }
        })
        .detach();
        this.reload(cx);
        this
    }
    pub fn set_identification_only(&mut self, cx: &mut Context<Self>) {
        self.identification_only = true;
        cx.notify();
    }
    pub fn set_speaker_analysis_only(&mut self, value: bool, cx: &mut Context<Self>) {
        self.speaker_analysis_only = value;
        cx.notify();
    }
    pub fn set_meeting(&mut self, meeting: Option<String>, cx: &mut Context<Self>) {
        self.meeting_id = meeting;
        self.generation += 1;
        self.readiness = None;
        self.readiness_progress = None;
        self.loading = false;
        cx.notify();
    }
    pub fn ready(&self) -> bool {
        !self.loading
            && !self.target_busy
            && self.error.is_none()
            && self.readiness.as_ref().is_some_and(|r| r.ready)
    }
    pub fn snapshot(&self) -> Result<PipelineConfig, String> {
        if self.ready() {
            Ok(self.config.clone())
        } else {
            Err(self.error.clone().unwrap_or_else(|| {
                self.readiness
                    .as_ref()
                    .map(|r| r.issues.join(". "))
                    .filter(|s| !s.is_empty())
                    .unwrap_or_else(|| "Check the selected pipeline before processing".into())
            }))
        }
    }
    pub fn reload(&mut self, cx: &mut Context<Self>) {
        let Some(pool) = AppServices::global(cx).pool() else {
            self.loading = false;
            self.readiness_progress = None;
            self.error = Some("The library is unavailable".into());
            return;
        };
        self.loading = true;
        self.readiness = None;
        self.choices = None;
        self.generation += 1;
        let generation = self.generation;
        self.begin_availability_progress("Loading saved pipeline choices…", cx);
        let io = Io::global(cx);
        cx.spawn(async move |this, cx| {
            let result = io
                .spawn(async move {
                    let config = config::load_config(&pool).await?;
                    let targets = config::list_targets(&pool).await?;
                    Ok::<_, anyhow::Error>((config, targets))
                })
                .await;
            let _ = this.update(cx, |this, cx| {
                if this.generation != generation {
                    return;
                }
                match result {
                    Ok(Ok((config, targets))) => {
                        this.config = config;
                        this.targets = targets;
                        this.error = None;
                        this.check(cx);
                    }
                    Ok(Err(e)) => {
                        this.loading = false;
                        this.readiness_progress = None;
                        this.error = Some(e.to_string());
                    }
                    Err(e) => {
                        this.loading = false;
                        this.readiness_progress = None;
                        this.error = Some(e.to_string());
                    }
                }
                cx.notify();
            });
        })
        .detach();
    }
    fn change(&mut self, field: &'static str, value: String, cx: &mut Context<Self>) {
        let previous = self.config.clone();
        if let Err(error) = apply_choice(&mut self.config, field, value) {
            self.error = Some(error);
            cx.notify();
            return;
        }
        if self.config == previous { return; }
        let revision = self
            .save_revision
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst)
            + 1;
        // Preserve downstream selections. The readiness report explains incompatible choices.
        self.error = None;
        self.check(cx);
        if self.defaults {
            if let Some(pool) = AppServices::global(cx).pool() {
                let config = self.config.clone();
                let save_revision = self.save_revision.clone();
                let save_gate = self.save_gate.clone();
                let io = Io::global(cx);
                cx.spawn(async move |this, cx| {
                    let r = io
                        .spawn(async move {
                            let _guard = save_gate.lock().await;
                            if save_revision.load(std::sync::atomic::Ordering::SeqCst) != revision {
                                return Ok(());
                            }
                            config::save_config(&pool, &config).await
                        })
                        .await;
                    let _ = this.update(cx, |this, cx| {
                        if this.save_revision.load(std::sync::atomic::Ordering::SeqCst) != revision
                        {
                            return;
                        }
                        match r {
                            Ok(Ok(())) => {}
                            Ok(Err(error)) => this.error = Some(error.to_string()),
                            Err(error) => this.error = Some(error.to_string()),
                        }
                        cx.notify();
                    });
                })
                .detach();
            }
        }
        cx.notify();
    }
    fn begin_availability_progress(&mut self, stage: &str, cx: &mut Context<Self>) -> String {
        let operation_id = format!(
            "readiness-check-{}",
            NEXT_TARGET_OPERATION_ID.fetch_add(1, std::sync::atomic::Ordering::SeqCst)
        );
        self.readiness_progress = Some(ReadinessProgress {
            operation_id: operation_id.clone(),
            stage: stage.into(),
            percent: None,
            started_at: std::time::Instant::now(),
        });
        let generation = self.generation;
        let timer_operation_id = operation_id.clone();
        cx.spawn(async move |this, cx| loop {
            cx.background_executor()
                .timer(std::time::Duration::from_secs(1))
                .await;
            let active = this.update(cx, |this, cx| {
                let active = readiness_progress_is_current(
                    this.loading,
                    this.generation,
                    this.readiness_progress.as_ref().map(|progress| progress.operation_id.as_str()),
                    Some(generation),
                    Some(&timer_operation_id),
                );
                if active {
                    cx.notify();
                }
                active
            });
            if !matches!(active, Ok(true)) {
                break;
            }
        })
        .detach();
        operation_id
    }

    fn check(&mut self, cx: &mut Context<Self>) {
        self.error = None;
        self.readiness_progress = None;
        if self.target_retry.take().is_some() {
            self.target_message = None;
        }
        let Some(pool) = AppServices::global(cx).pool() else {
            self.loading = false;
            self.readiness = None;
            self.error = Some("The library is unavailable".into());
            cx.notify();
            return;
        };
        self.generation += 1;
        let generation = self.generation;
        let config = self.config.clone();
        if self
            .targets
            .iter()
            .any(|target| target.id == config.target_id && target.kind == TargetKind::Local)
        {
            self.choices = Some((config.target_id.clone(), config::local_choices()));
        } else if !self.identification_only
            && !self
                .choices
                .as_ref()
                .is_some_and(|(id, _)| id == &config.target_id)
        {
            let choice_pool = pool.clone();
            let choice_target = config.target_id.clone();
            let choice_io = Io::global(cx);
            cx.spawn(async move |this, cx| {
                let target = choice_target.clone();
                let result = choice_io
                    .spawn(async move {
                        parley_core::audio::processing::service::target_choices(
                            &choice_pool,
                            &choice_target,
                        )
                        .await
                    })
                    .await;
                let _ = this.update(cx, |this, cx| {
                    if this.generation != generation || this.config.target_id != target {
                        return;
                    }
                    if let Ok(Ok(choices)) = result {
                        this.choices = Some((target, choices));
                        cx.notify();
                    }
                });
            })
            .detach();
        }
        let identification_only = self.identification_only;
        let speaker_analysis_only = self.speaker_analysis_only;
        let meeting_id = self.meeting_id.clone();
        let io = Io::global(cx);
        self.loading = true;
        self.readiness = None;
        let stage = if identification_only && meeting_id.is_some() {
            "Checking stored speaker evidence and identification model availability…"
        } else if identification_only {
            "Checking the speaker identification model and target runtime…"
        } else if speaker_analysis_only {
            "Checking the selected speaker package and target runtime…"
        } else {
            "Connecting to the selected target and checking its worker setup…"
        };
        let operation_id = self.begin_availability_progress(stage, cx);
        let sink = AppServices::global(cx).sink.clone();
        cx.notify();
        cx.spawn(async move |this, cx| {
            let event_operation_id = operation_id.clone();
            let r = io
                .spawn(async move {
                    if identification_only && meeting_id.is_some() {
                        parley_core::audio::processing::service::check_identification_for_meeting(
                            &pool,
                            meeting_id.as_deref().unwrap(),
                            &config.target_id,
                            config.identification,
                        )
                        .await
                    } else if identification_only {
                        parley_core::audio::processing::service::check_identification_readiness(
                            &pool,
                            &config.target_id,
                            config.identification,
                        )
                        .await
                    } else if speaker_analysis_only {
                        parley_core::audio::processing::service::check_speaker_analysis_readiness(
                            &pool,
                            &config.target_id,
                            config.speaker_pipeline,
                            Identification::Off,
                        )
                        .await
                    } else {
                        parley_core::audio::processing::service::check_readiness_with_progress(
                            &pool,
                            &config,
                            move |message, percent| {
                                let _ = sink.emit_value(
                                    "processing-readiness-progress",
                                    serde_json::json!({
                                        "operation_id": event_operation_id,
                                        "generation": generation,
                                        "message": message,
                                        "percent": percent,
                                    }),
                                );
                            },
                        )
                        .await
                    }
                })
                .await;
            let _ = this.update(cx, |this, cx| {
                if !readiness_progress_is_current(
                    this.loading,
                    this.generation,
                    this.readiness_progress.as_ref().map(|progress| progress.operation_id.as_str()),
                    Some(generation),
                    Some(&operation_id),
                ) {
                    return;
                }
                this.loading = false;
                this.readiness_progress = None;
                match r {
                    Ok(Ok(report)) => {
                        this.readiness = Some(report);
                    }
                    Ok(Err(e)) => this.error = Some(e.to_string()),
                    Err(e) => this.error = Some(e.to_string()),
                }
                cx.notify();
            });
        })
        .detach();
    }
    fn picker(
        &self,
        field: &'static str,
        title: &'static str,
        current: String,
        options: Vec<(String, String)>,
        cx: &mut Context<Self>,
    ) -> AnyElement {
        let view = cx.entity().downgrade();
        h_flex()
            .w_full()
            .gap_3()
            .justify_between()
            .items_center()
            .child(div().text_sm().child(title))
            .child(
                Button::new(format!("pipeline-{field}"))
                    .outline()
                    .label(current)
                    .disabled(options.is_empty() || self.targets.is_empty() || self.target_busy)
                    .dropdown_menu(move |menu, _, _| {
                        let mut menu = menu;
                        for (value, label) in &options {
                            let value = value.clone();
                            let view = view.clone();
                            menu = menu.item(PopupMenuItem::new(label.clone()).on_click(
                                move |_, _, cx| {
                                    let _ = view.update(cx, |this, cx| {
                                        this.change(field, value.clone(), cx)
                                    });
                                },
                            ));
                        }
                        menu
                    }),
            )
            .into_any_element()
    }
    fn edit_target(
        &mut self,
        target: ProcessingTarget,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        if self.target_busy {
            return;
        }
        for (input, value) in [
            (&self.target_name, target.name.clone()),
            (
                &self.target_host,
                target.ssh_host.clone().unwrap_or_default(),
            ),
            (
                &self.target_worker,
                target.worker_path.clone().unwrap_or_default(),
            ),
            (
                &self.target_python,
                target.python.clone().unwrap_or_default(),
            ),
            (
                &self.target_assets,
                target.assets_dir.clone().unwrap_or_default(),
            ),
            (&self.target_device, target.device.clone()),
        ] {
            input.update(cx, |s, cx| s.set_value(value, window, cx));
        }
        self.editing_target = Some(target);
        if self.target_retry.is_none() {
            self.target_message = None;
        }
        cx.notify();
    }
    fn save_target(&mut self, cx: &mut Context<Self>) {
        if self.target_busy {
            return;
        }
        let Some(mut target) = self.editing_target.clone() else {
            return;
        };
        let Some(pool) = AppServices::global(cx).pool() else {
            return;
        };
        let optional = |input: &Entity<InputState>| {
            let value = input.read(cx).value().trim().to_string();
            if value.is_empty() {
                None
            } else {
                Some(value)
            }
        };
        target.name = self.target_name.read(cx).value().trim().into();
        target.ssh_host = optional(&self.target_host);
        target.worker_path = optional(&self.target_worker);
        target.python = optional(&self.target_python);
        target.assets_dir = optional(&self.target_assets);
        target.device = self.target_device.read(cx).value().trim().into();
        self.target_busy = true;
        self.target_operation_id = None;
        self.generation += 1;
        self.loading = false;
        self.readiness_progress = None;
        cx.notify();
        let io = Io::global(cx);
        cx.spawn(async move |this, cx| {
            let r = io
                .spawn(async move {
                    config::save_target(&pool, &target).await?;
                    config::list_targets(&pool).await
                })
                .await;
            let _ = this.update(cx, |this, cx| {
                this.target_busy = false;
                match r {
                    Ok(Ok(targets)) => {
                        this.targets = targets;
                        this.choices = None;
                        this.editing_target = None;
                        this.target_message =
                            Some("Target saved. Check its setup before processing.".into());
                        this.check(cx);
                    }
                    Ok(Err(e)) => this.target_message = Some(e.to_string()),
                    Err(e) => this.target_message = Some(e.to_string()),
                }
                cx.notify();
            });
        })
        .detach();
    }
    fn choose_native_bundle(&mut self, cx: &mut Context<Self>) {
        if self.target_busy {
            return;
        }
        let paths = cx.prompt_for_paths(PathPromptOptions {
            files: true,
            directories: false,
            multiple: false,
            prompt: Some("Select a native helper bundle (.tar, .tar.gz or .tgz)".into()),
        });
        cx.spawn(async move |this, cx| {
            if let Ok(Ok(Some(paths))) = paths.await {
                if let Some(path) = paths.into_iter().next() {
                    let _ = this.update_in(cx, |this, window, cx| {
                        this.native_bundle.update(cx, |input, cx| {
                            input.set_value(path.to_string_lossy().to_string(), window, cx)
                        });
                        cx.notify();
                    });
                }
            }
        })
        .detach();
    }

    fn target_operation(
        &mut self,
        target: ProcessingTarget,
        operation: &'static str,
        cx: &mut Context<Self>,
    ) {
        if self.target_busy {
            return;
        }
        if operation == "prepare" && target.id != self.config.target_id {
            self.target_message = Some("Select this target before downloading its models".into());
            cx.notify();
            return;
        }
        let io = Io::global(cx);
        let config = self.config.clone();
        let speaker_only = self.speaker_analysis_only;
        let bundle = std::path::PathBuf::from(self.native_bundle.read(cx).value().trim());
        let sink = AppServices::global(cx).sink.clone();
        let target_id = target.id.clone();
        let operation_id = format!(
            "target-operation-{}",
            NEXT_TARGET_OPERATION_ID.fetch_add(1, std::sync::atomic::Ordering::SeqCst)
        );
        self.target_operation_id = Some(operation_id.clone());
        self.target_busy = true;
        self.target_retry = None;
        self.error = None;
        self.readiness = None;
        // Discard any availability check that started before this operation.
        self.generation += 1;
        self.loading = false;
        self.readiness_progress = None;
        self.target_message = Some(
            match operation {
                "install" => "Installing or updating the isolated worker…",
                "bundle" => "Installing the native helpers…",
                "rollback" => "Restoring the previous worker…",
                "prepare" if speaker_only => "Downloading the selected speaker package…",
                "prepare" => "Downloading the selected pipeline models…",
                _ => "Checking target setup…",
            }
            .into(),
        );
        cx.notify();
        let retry_target = target.clone();
        parley_core::audio::processing::service::invalidate_readiness(&target.id);
        cx.spawn(async move |this, cx| {
            let result = io
                .spawn(async move {
                    use parley_core::audio::processing::{executor, service as processing};
                    let progress = |stage: &str, percent: u32, message: &str| {
                        let _ = sink.emit_value(
                            "processing-target-progress",
                            serde_json::json!({
                                "target_id": target_id,
                                "operation_id": operation_id,
                                "stage": stage,
                                "percent": percent,
                                "message": message,
                            }),
                        );
                    };
                    match operation {
                        "rollback" => executor::rollback(&target, progress).await,
                        "bundle" => executor::install_bundle(&target, &bundle, progress).await,
                        "install" => executor::install(&target, progress).await,
                        "prepare" if speaker_only => {
                            processing::prepare_speaker_pipeline(
                                &target,
                                config.speaker_pipeline,
                                Identification::Off,
                                progress,
                            )
                            .await
                        }
                        "prepare" => processing::prepare_pipeline(&target, &config, progress).await,
                        _ => executor::probe_config(&target, &config).await,
                    }
                })
                .await;
            let result = match result {
                Ok(result) => result.map_err(|error| error.to_string()),
                Err(error) => Err(error.to_string()),
            };
            let outcome = target_operation_outcome(operation, result);
            let _ = this.update(cx, |this, cx| {
                this.target_busy = false;
                this.target_operation_id = None;
                match outcome {
                    TargetOperationOutcome::Succeeded(message) => {
                        this.target_message = Some(message);
                        this.check(cx);
                    }
                    TargetOperationOutcome::Failed(message) => {
                        this.target_message = Some(message.clone());
                        this.error = Some(message);
                        this.readiness = None;
                        this.loading = false;
                        this.target_retry = Some((retry_target, operation));
                    }
                }
                cx.notify();
            });
        })
        .detach();
    }

    fn remove_target(&mut self, id: String, cx: &mut Context<Self>) {
        if self.target_busy {
            return;
        }
        let Some(pool) = AppServices::global(cx).pool() else {
            return;
        };
        self.target_busy = true;
        self.target_operation_id = None;
        self.generation += 1;
        self.loading = false;
        self.readiness_progress = None;
        cx.notify();
        let io = Io::global(cx);
        cx.spawn(async move |this, cx| {
            let r = io
                .spawn(async move {
                    config::delete_target(&pool, &id).await?;
                    config::list_targets(&pool).await
                })
                .await;
            let _ = this.update(cx, |this, cx| {
                this.target_busy = false;
                match r {
                    Ok(Ok(targets)) => {
                        this.targets = targets;
                        this.choices = None;
                        if !this
                            .targets
                            .iter()
                            .any(|target| target.id == this.config.target_id)
                        {
                            this.target_message = Some(
                                "The selected target was removed. Choose another target above."
                                    .into(),
                            );
                        }
                        this.check(cx);
                    }
                    Ok(Err(e)) => this.target_message = Some(e.to_string()),
                    Err(e) => this.target_message = Some(e.to_string()),
                }
                cx.notify();
            });
        })
        .detach();
    }
}
impl Render for PipelineEditor {
    fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let engines = self
            .choices
            .as_ref()
            .filter(|(id, _)| id == &self.config.target_id)
            .map(|(_, choices)| choices.engines.clone())
            .or_else(|| self.readiness.as_ref().map(|report| report.engines.clone()))
            .unwrap_or_default();
        let models = self
            .readiness
            .as_ref()
            .map(|r| r.models.clone())
            .unwrap_or_else(config::speech_models);
        let speakers = self
            .choices
            .as_ref()
            .filter(|(id, _)| id == &self.config.target_id)
            .map(|(_, choices)| choices.speaker_pipelines.clone())
            .or_else(|| {
                self.readiness
                    .as_ref()
                    .map(|report| report.speaker_pipelines.clone())
            })
            .unwrap_or_default();
        let mut identities = self
            .choices
            .as_ref()
            .filter(|(id, _)| id == &self.config.target_id)
            .map(|(_, choices)| choices.identifications.clone())
            .or_else(|| {
                self.readiness
                    .as_ref()
                    .map(|report| report.identifications.clone())
            })
            .unwrap_or_default();
        for choice in if self.identification_only {
            vec![
                Identification::Same,
                Identification::Off,
                Identification::Campplus,
                Identification::Wespeaker,
            ]
        } else {
            vec![Identification::Same, Identification::Off]
        } {
            if !identities.contains(&choice) {
                identities.push(choice);
            }
        }
        let target_label = self
            .targets
            .iter()
            .find(|t| t.id == self.config.target_id)
            .map(|t| t.name.clone())
            .unwrap_or_else(|| "Choose a target".into());
        let model_label = config::speech_models()
            .into_iter()
            .find(|m| m.id == self.config.speech_model && m.engine == self.config.speech_engine)
            .map(|m| m.label)
            .unwrap_or_else(|| "Choose a compatible model".into());
        let artifact_ready = self
            .readiness
            .as_ref()
            .and_then(|report| report.capabilities.get("artifacts"))
            .and_then(|items| items.as_array())
            .and_then(|items| {
                items
                    .iter()
                    .find(|item| item["id"] == self.config.speech_model)
            })
            .and_then(|item| item["ready"].as_bool());
        let model_label = match artifact_ready {
            Some(true) => format!("{model_label} · Ready"),
            Some(false) => format!("{model_label} · Download required"),
            None => model_label,
        };
        let encode = |v: serde_json::Value| v.as_str().unwrap_or_default().to_string();
        let inference_required = !(self.identification_only
            && (self.config.identification == Identification::Off
                || self
                    .readiness
                    .as_ref()
                    .is_some_and(|report| report.capabilities["inference_required"] == false)));
        let mut body = v_flex().w_full().gap_3();
        if inference_required {
            body = body.child(
                self.picker(
                    "target",
                    "Target",
                    target_label,
                    self.targets
                        .iter()
                        .map(|target| (target.id.clone(), target.name.clone()))
                        .collect(),
                    cx,
                ),
            );
        }
        if !self.identification_only && !self.speaker_analysis_only {
            body = body
                .child(
                    self.picker(
                        "engine",
                        "Speech engine",
                        engine_label(self.config.speech_engine).into(),
                        engines
                            .iter()
                            .map(|engine| {
                                (
                                    encode(serde_json::to_value(engine).unwrap()),
                                    engine_label(*engine).into(),
                                )
                            })
                            .collect(),
                        cx,
                    ),
                )
                .child(
                    self.picker(
                        "model",
                        "Speech model",
                        model_label,
                        models
                            .into_iter()
                            .filter(|model| {
                                model.engine == self.config.speech_engine && !model.experimental
                            })
                            .map(|model| (model.id, model.label))
                            .collect(),
                        cx,
                    ),
                );
        }
        if !self.identification_only {
            let mut speakers = speakers;
            if self.speaker_analysis_only {
                for pipeline in [SpeakerPipeline::Sherpa, SpeakerPipeline::Community1] {
                    if !speakers.contains(&pipeline) {
                        speakers.push(pipeline);
                    }
                }
            }
            body = body.child(
                self.picker(
                    "speaker",
                    "Speaker pipeline",
                    speaker_label(self.config.speaker_pipeline).into(),
                    speakers
                        .iter()
                        .map(|pipeline| {
                            (
                                encode(serde_json::to_value(pipeline).unwrap()),
                                speaker_label(*pipeline).into(),
                            )
                        })
                        .collect(),
                    cx,
                ),
            );
        }
        if !self.speaker_analysis_only {
            body = body.child(
                self.picker(
                    "identification",
                    "Speaker identification",
                    identification_label(self.config.identification).into(),
                    identities
                        .iter()
                        .map(|choice| {
                            (
                                encode(serde_json::to_value(choice).unwrap()),
                                identification_label(*choice).into(),
                            )
                        })
                        .collect(),
                    cx,
                ),
            );
        }
        let status = if self.target_busy {
            self.target_message
                .clone()
                .unwrap_or_else(|| "Target setup is in progress…".into())
        } else if self.loading {
            self.readiness_progress
                .as_ref()
                .map(|progress| {
                    readiness_progress_status(
                        &progress.stage,
                        progress.percent,
                        progress.started_at.elapsed().as_secs(),
                    )
                })
                .unwrap_or_else(|| "Checking selected pipeline availability…".into())
        } else if let Some(e) = &self.error {
            e.clone()
        } else if let Some(r) = &self.readiness {
            if r.ready && self.identification_only && !inference_required {
                if self.config.identification == Identification::Off {
                    "Speaker identification is off".into()
                } else {
                    "Ready · No new speaker extraction needed; no inference target required".into()
                }
            } else if r.ready && self.identification_only {
                format!(
                    "Ready · Speaker identification: {}",
                    r.speaker_device.as_deref().unwrap_or("Verified target")
                )
            } else if r.ready && self.speaker_analysis_only {
                format!(
                    "Ready · Speaker analysis: {}",
                    r.speaker_device.as_deref().unwrap_or("Verified target")
                )
            } else if r.ready {
                format!(
                    "Ready · Speech: {} · Speakers: {} · Compute: {}",
                    r.speech_device.as_deref().unwrap_or("Unknown"),
                    r.speaker_device.as_deref().unwrap_or("Unknown"),
                    r.compute_type.as_deref().unwrap_or("Unknown")
                )
            } else {
                r.issues.join(" · ")
            }
        } else {
            "Check setup to continue".into()
        };
        let mut status_view = v_flex().w_full().gap_2().child(
            h_flex()
                .w_full()
                .gap_2()
                .items_start()
                .when(self.loading && !self.target_busy, |row| {
                    row.child(Spinner::new().small())
                })
                .child(div()
                    .flex_1()
                    .min_w_0()
                    .text_xs()
                    .text_color(if self.ready() {
                        cx.theme().muted_foreground
                    } else {
                        cx.theme().warning
                    })
                    .child(multiline_status(&status))),
        );
        if self.loading && !self.target_busy {
            status_view = status_view.child(
                div()
                    .text_xs()
                    .text_color(cx.theme().muted_foreground)
                    .child("Large models can take a minute or more to check."),
            );
            if let Some(percent) = self.readiness_progress.as_ref().and_then(|progress| progress.percent) {
                status_view = status_view.child(
                    Progress::new("pipeline-readiness-stage-progress").value(f32::from(percent)),
                );
            }
        }
        body = body
            .child(status_view)
            .child(
                h_flex()
                    .gap_2()
                    .child(
                        Button::new("check-pipeline")
                            .outline()
                            .label("Check availability")
                            .disabled(self.loading || self.target_busy)
                            .on_click(cx.listener(|this, _, _, cx| {
                                if this.loading || this.target_busy {
                                    return;
                                }
                                this.error = None;
                                parley_core::audio::processing::service::invalidate_readiness(&this.config.target_id);
                                this.check(cx);
                            })),
                    )
                    .when(self.defaults, |row| {
                        row.child(
                            Button::new("manage-targets")
                                .outline()
                                .label("Manage targets")
                                .on_click(cx.listener(|this, _, _, cx| {
                                    this.managing = !this.managing;
                                    cx.notify();
                                })),
                        )
                    }),
            );
        if let Some((target, operation)) = &self.target_retry {
            let target = target.clone();
            let operation = *operation;
            body = body.child(
                Button::new("retry-target-operation")
                    .outline()
                    .label("Retry failed setup")
                    .disabled(self.target_busy || self.loading)
                    .on_click(cx.listener(move |this, _, _, cx| {
                        this.target_operation(target.clone(), operation, cx)
                    })),
            );
        }
        if artifact_ready == Some(false) && !self.identification_only && !self.speaker_analysis_only
        {
            if let Some(target) = self
                .targets
                .iter()
                .find(|target| target.id == self.config.target_id)
                .cloned()
            {
                body = body.child(
                    Button::new("download-required-pipeline-models")
                        .outline()
                        .label("Download required models")
                        .disabled(self.target_busy)
                        .on_click(cx.listener(move |this, _, _, cx| {
                            this.target_operation(target.clone(), "prepare", cx)
                        })),
                );
            }
        }
        if self.speaker_analysis_only && self.readiness.as_ref().is_some_and(|report| !report.ready)
        {
            if let Some(target) = self
                .targets
                .iter()
                .find(|target| target.id == self.config.target_id)
                .cloned()
            {
                body = body.child(
                    Button::new("download-required-speaker-package")
                        .outline()
                        .label("Download required speaker models")
                        .disabled(self.target_busy || self.loading)
                        .on_click(cx.listener(move |this, _, _, cx| {
                            this.target_operation(target.clone(), "prepare", cx)
                        })),
                );
            }
        }
        if !self.defaults && self.target_retry.is_none() {
            if let Some(message) = &self.target_message {
                body = body.child(div().text_xs().child(multiline_status(message)));
            }
        }
        if self.defaults {
            body=body.child(div().text_xs().text_color(cx.theme().muted_foreground).child("Defaults apply to future jobs. Live transcription is a local preview; saved recordings use this final pipeline."));
        }
        if self.managing {
            body = body.child(self.render_targets(cx));
        }
        body
    }
}
impl PipelineEditor {
    fn render_targets(&self, cx: &mut Context<Self>) -> AnyElement {
        let mut body = v_flex()
            .w_full()
            .gap_3()
            .p_3()
            .border_1()
            .border_color(cx.theme().border)
            .rounded_md()
            .child(div().text_sm().font_semibold().child("Processing targets"));
        if self.target_retry.is_some() {
            if let Some(message) = &self.target_message {
                body = body.child(
                    div()
                        .text_xs()
                        .text_color(cx.theme().warning)
                        .child(multiline_status(message)),
                );
            }
        }
        body=body.child(div().text_xs().child("Native helper bundle (optional)")).child(h_flex().w_full().gap_2().child(div().flex_1().child(Input::new(&self.native_bundle))).child(Button::new("choose-native-helper-bundle").outline().label("Choose bundle…").disabled(self.target_busy).on_click(cx.listener(|this,_,_,cx|this.choose_native_bundle(cx))))).child(div().text_xs().text_color(cx.theme().muted_foreground).child("Use a local bundle built for the target computer. The bundle path stays on this computer."));
        for target in &self.targets {
            let edit = target.clone();
            let delete = target.id.clone();
            let probe = target.clone();
            let install = target.clone();
            let prepare = target.clone();
            let native = target.clone();
            let rollback = target.clone();
            body = body.child(
                h_flex()
                    .w_full()
                    .flex_wrap()
                    .gap_2()
                    .items_center()
                    .child(div().flex_1().text_sm().child(target.name.clone()))
                    .child(
                        Button::new(format!("edit-target-{}", target.id))
                            .outline()
                            .label("Setup")
                            .disabled(self.target_busy)
                            .on_click(cx.listener(move |this, _, window, cx| {
                                this.edit_target(edit.clone(), window, cx)
                            })),
                    )
                    .child(
                        Button::new(format!("probe-target-{}", target.id))
                            .outline()
                            .label("Check setup")
                            .disabled(self.target_busy)
                            .on_click(cx.listener(move |this, _, _, cx| {
                                this.target_operation(probe.clone(), "probe", cx)
                            })),
                    )
                    .child(
                        Button::new(format!("install-target-{}", target.id))
                            .outline()
                            .label("Install / update worker")
                            .disabled(self.target_busy)
                            .on_click(cx.listener(move |this, _, _, cx| {
                                this.target_operation(install.clone(), "install", cx)
                            })),
                    )
                    .child(
                        Button::new(format!("prepare-target-{}", target.id))
                            .outline()
                            .label("Download models")
                            .disabled(self.target_busy || target.id != self.config.target_id)
                            .on_click(cx.listener(move |this, _, _, cx| {
                                this.target_operation(prepare.clone(), "prepare", cx)
                            })),
                    )
                    .child(
                        Button::new(format!("install-native-helpers-{}", target.id))
                            .outline()
                            .label("Install native helpers")
                            .disabled(
                                self.target_busy
                                    || self.native_bundle.read(cx).value().trim().is_empty(),
                            )
                            .on_click(cx.listener(move |this, _, _, cx| {
                                this.target_operation(native.clone(), "bundle", cx)
                            })),
                    )
                    .child(
                        Button::new(format!("restore-worker-{}", target.id))
                            .outline()
                            .label("Restore previous worker")
                            .disabled(self.target_busy)
                            .on_click(cx.listener(move |this, _, _, cx| {
                                this.target_operation(rollback.clone(), "rollback", cx)
                            })),
                    )
                    .when(target.kind != TargetKind::Local, |row| {
                        row.child(
                            Button::new(format!("delete-target-{}", target.id))
                                .ghost()
                                .label("Remove")
                                .disabled(self.target_busy)
                                .on_click(cx.listener(move |this, _, _, cx| {
                                    this.remove_target(delete.clone(), cx)
                                })),
                        )
                    }),
            );
        }
        body=body.child(div().text_xs().text_color(cx.theme().muted_foreground).child("Restore previous worker switches code, dependencies and native helpers to the previous validated release. Model downloads are kept. An unavailable previous release is reported explicitly."));
        body = body.child(
            Button::new("add-processing-target")
                .outline()
                .label("Add remote target")
                .disabled(self.target_busy)
                .on_click(cx.listener(|this, _, window, cx| {
                    let mut target = ProcessingTarget::default();
                    target.id = format!("target-{}", chrono::Utc::now().timestamp_micros());
                    target.kind = TargetKind::Ssh;
                    target.name = String::new();
                    this.edit_target(target, window, cx);
                })),
        );
        if let Some(target) = &self.editing_target {
            body=body.child(div().text_xs().child("Display name")).child(Input::new(&self.target_name)).when(target.kind==TargetKind::Ssh,|b|b.child(div().text_xs().child("SSH host alias")).child(Input::new(&self.target_host))).child(div().text_xs().child("Worker folder (optional)")).child(Input::new(&self.target_worker)).child(div().text_xs().child("Python executable (optional)")).child(Input::new(&self.target_python)).child(div().text_xs().child("Model assets folder (optional)")).child(Input::new(&self.target_assets)).child(div().text_xs().child("Device")).child(Input::new(&self.target_device)).child(div().text_xs().text_color(cx.theme().muted_foreground).child(if target.kind == TargetKind::Ssh { "Uses your existing SSH setup. Choose a dedicated private worker folder. Runtime paths and device are checked on the selected computer." } else { "The default worker folder is private to this app. To use another location, choose a dedicated private folder. Runtime paths and device are checked on this computer." })).child(h_flex().gap_2().child(Button::new("save-processing-target").primary().label("Save target").disabled(self.target_busy).on_click(cx.listener(|this,_,_,cx|this.save_target(cx)))).child(Button::new("cancel-target-setup").ghost().label("Cancel").on_click(cx.listener(|this,_,_,cx|{this.editing_target=None;cx.notify();}))));
        }
        if self.target_retry.is_none() {
            if let Some(message) = &self.target_message {
                body = body.child(div().text_xs().child(multiline_status(message)));
            }
        }
        body.into_any_element()
    }
}

struct ReadinessProgress {
    operation_id: String,
    stage: String,
    percent: Option<u8>,
    started_at: std::time::Instant,
}

fn readiness_progress_is_current(
    loading: bool,
    generation: u64,
    operation_id: Option<&str>,
    event_generation: Option<u64>,
    event_operation_id: Option<&str>,
) -> bool {
    loading
        && event_generation == Some(generation)
        && operation_id.is_some()
        && operation_id == event_operation_id
}

fn readiness_progress_status(stage: &str, percent: Option<u8>, elapsed_secs: u64) -> String {
    let stage_percent = percent.map(|percent| format!(" {percent}%")).unwrap_or_default();
    format!("{stage}{stage_percent} · {elapsed_secs}s elapsed")
}

fn multiline_status(message: &str) -> impl IntoElement {
    v_flex().w_full().gap_1().children(
        message
            .lines()
            .map(|line| div().w_full().whitespace_normal().child(line.to_owned()))
            .collect::<Vec<_>>(),
    )
}

#[derive(Debug, PartialEq, Eq)]
enum TargetOperationOutcome {
    Succeeded(String),
    Failed(String),
}

fn target_operation_outcome(
    operation: &str,
    result: Result<serde_json::Value, String>,
) -> TargetOperationOutcome {
    let report = match result {
        Ok(report) => report,
        Err(message) => return TargetOperationOutcome::Failed(message),
    };
    let issues = report
        .get("issues")
        .and_then(|value| value.as_array())
        .map(|items| {
            items
                .iter()
                .filter_map(|value| value.as_str())
                .collect::<Vec<_>>()
                .join(" · ")
        })
        .unwrap_or_default();
    let message = if issues.is_empty() {
        match operation {
            "rollback" => "Previous worker restored. Model downloads are kept; checking availability…",
            "bundle" => "Native helper installation finished. Checking availability…",
            "install" => "Worker installation finished. Checking availability…",
            "prepare" => "Model preparation finished. Checking availability…",
            _ => "Setup check finished.",
        }
        .into()
    } else {
        issues
    };
    TargetOperationOutcome::Succeeded(message)
}

fn apply_choice(config: &mut PipelineConfig, field: &str, value: String) -> Result<(), String> {
    let decode = |value: String| serde_json::Value::String(value);
    match field {
        "target" => config.target_id = value,
        "engine" => {
            config.speech_engine = serde_json::from_value(decode(value))
                .map_err(|_| "Choose a listed speech engine")?
        }
        "model" => config.speech_model = value,
        "speaker" => {
            config.speaker_pipeline = serde_json::from_value(decode(value))
                .map_err(|_| "Choose a listed speaker pipeline")?
        }
        "identification" => {
            config.identification = serde_json::from_value(decode(value))
                .map_err(|_| "Choose a listed identification model")?
        }
        _ => return Err("Unknown pipeline field".into()),
    }
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::{
        apply_choice, readiness_progress_is_current, readiness_progress_status,
        target_operation_outcome, TargetOperationOutcome,
    };
    use parley_core::audio::processing::config::{self, PipelineConfig};
    #[test]
    fn readiness_progress_formats_elapsed_time_without_inventing_percent() {
        assert_eq!(
            readiness_progress_status("Checking worker setup…", None, 12),
            "Checking worker setup… · 12s elapsed"
        );
        assert_eq!(
            readiness_progress_status("Checking speech model file…", Some(63), 18),
            "Checking speech model file… 63% · 18s elapsed"
        );
    }

    #[test]
    fn readiness_progress_routes_only_to_current_running_operation() {
        assert!(readiness_progress_is_current(true, 4, Some("check-2"), Some(4), Some("check-2")));
        // Two editors may share a generation; globally unique IDs isolate their events.
        assert!(!readiness_progress_is_current(true, 4, Some("check-2"), Some(4), Some("check-1")));
        assert!(!readiness_progress_is_current(true, 5, Some("check-2"), Some(4), Some("check-2")));
        // Completion or a setup failure stops both late events and the elapsed-time ticker.
        assert!(!readiness_progress_is_current(false, 4, Some("check-2"), Some(4), Some("check-2")));
        assert!(!readiness_progress_is_current(true, 4, None, Some(4), None));
        assert!(!readiness_progress_is_current(true, 4, Some("check-2"), None, Some("check-2")));
    }

    #[test]
    fn target_operation_failure_preserves_actionable_message_without_refresh() {
        let message = "Community-1 requires model access.\nAccept the model terms, then retry.";
        for operation in ["prepare", "install"] {
            assert_eq!(
                target_operation_outcome(operation, Err(message.into())),
                TargetOperationOutcome::Failed(message.into())
            );
        }
    }

    #[test]
    fn target_operation_success_allows_readiness_refresh() {
        assert!(matches!(
            target_operation_outcome("prepare", Ok(serde_json::json!({ "issues": [] }))),
            TargetOperationOutcome::Succeeded(_)
        ));
    }
    #[test]
    fn changing_engine_keeps_quantization_until_explicit_reselection() {
        let mut config = PipelineConfig::default();
        let old_model = config.speech_model.clone();
        apply_choice(&mut config, "engine", "faster_whisper".into()).unwrap();
        assert_eq!(config.speech_model, old_model);
        assert!(config::validate_config(&config).is_err());
        apply_choice(&mut config, "model", "large-v3-turbo-int8".into()).unwrap();
        assert!(config::validate_config(&config).is_ok());
    }
    #[test]
    fn target_changes_do_not_replace_pipeline_choices() {
        let mut config = PipelineConfig::default();
        let before = config.clone();
        apply_choice(&mut config, "target", "a-different-target".into()).unwrap();
        assert_eq!(config.speech_engine, before.speech_engine);
        assert_eq!(config.speech_model, before.speech_model);
        assert_eq!(config.speaker_pipeline, before.speaker_pipeline);
        assert_eq!(config.identification, before.identification);
    }
    #[test]
    fn invalid_ids_leave_existing_choices_intact() {
        let mut config = PipelineConfig::default();
        let before = config.clone();
        assert!(apply_choice(&mut config, "engine", "unrecognized".into()).is_err());
        assert_eq!(config, before);
    }
}
