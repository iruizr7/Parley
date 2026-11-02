//! Import Audio dialog: pick/drop an audio file, validate it, choose a
//! title/language/model/speaker-count, then run
//! `parley_core::audio::import::start_import_with` and track its progress.
//! Progress arrives as the `import-progress`/`import-complete`/
//! `import-error`/`import-warning` core events.

mod logic;
pub(crate) mod timeline;

use gpui_kit::component::scroll::ScrollableElement as _;

use std::path::PathBuf;

use gpui_kit::component::{
    button::{Button, ButtonVariants as _},
    dialog::Dialog,
    h_flex,
    input::{Input, InputEvent, InputState},
    label::Label,
    notification::{Notification, NotificationType},
    progress::Progress,
    spinner::Spinner,
    Sizable as _,
    select::{Select, SelectItem, SelectState},
    v_flex, ActiveTheme, Disableable as _, IndexPath, StyledExt as _, WindowExt as _,
};
use gpui_kit::*;

use parley_core::audio::import::{
    self, AudioFileInfo, ImportError, ImportProgress, ImportResult, ImportWarning,
};

use crate::app_state::AppServices;
use crate::core_events::CoreEvent;
use crate::shell::{self, Route};

pub use logic::pick_dropped_audio_file;
pub(crate) use logic::{language_label, LANGUAGES};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Status {
    Idle,
    Validating,
    Processing,
    Complete,
    Error,
}

#[derive(Debug, Clone, PartialEq)]
struct LangOption {
    code: SharedString,
    label: SharedString,
}

impl SelectItem for LangOption {
    type Value = SharedString;
    fn title(&self) -> SharedString {
        self.label.clone()
    }
    fn value(&self) -> &Self::Value {
        &self.code
    }
}

#[derive(Debug, Clone, PartialEq)]
struct SpeakerOption {
    count: i32,
    label: SharedString,
}

impl SelectItem for SpeakerOption {
    type Value = i32;
    fn title(&self) -> SharedString {
        self.label.clone()
    }
    fn value(&self) -> &Self::Value {
        &self.count
    }
}

/// Everything the dialog needs; lives only while the dialog is open (created
/// fresh in [`open`]/[`open_with_file`], kept alive by the `Dialog`'s
/// content-builder closure holding a clone of the `Entity`).
struct ImportState {
    status: Status,
    file_info: Option<AudioFileInfo>,
    error: Option<String>,
    progress: Option<ImportProgress>,
    timeline: timeline::Timeline,
    estimate: timeline::Estimate,
    started_at: Option<std::time::Instant>,
    saved_meeting: Option<String>,
    title: Entity<InputState>,
    title_edited: bool,
    show_advanced: bool,
    language: Entity<SelectState<Vec<LangOption>>>,
    speakers: Entity<SelectState<Vec<SpeakerOption>>>,
    pipeline: Entity<crate::views::processing::PipelineEditor>,
    _subscriptions: Vec<Subscription>,
}

impl ImportState {
    fn new(window: &mut Window, cx: &mut Context<Self>) -> Self {
        let title = cx.new(|cx| InputState::new(window, cx).placeholder("Meeting title"));
        let mut subscriptions = vec![cx.subscribe(&title, |this, _, event, _cx| {
            if matches!(event, InputEvent::Change) {
                this.title_edited = true;
            }
        })];

        let lang_items: Vec<LangOption> = logic::LANGUAGES
            .iter()
            .map(|(code, _)| LangOption {
                code: (*code).into(),
                label: logic::language_label(code).into(),
            })
            .collect();
        let language =
            cx.new(|cx| SelectState::new(lang_items, Some(IndexPath::default()), window, cx));

        let speaker_items: Vec<SpeakerOption> = std::iter::once(SpeakerOption {
            count: 0,
            label: logic::speaker_count_label(0).into(),
        })
        .chain((1..=8).map(|n| SpeakerOption {
            count: n,
            label: logic::speaker_count_label(n).into(),
        }))
        .collect();
        let speakers =
            cx.new(|cx| SelectState::new(speaker_items, Some(IndexPath::default()), window, cx));

        let core_events = AppServices::global(cx).core_events.clone();
        subscriptions.push(
            cx.subscribe(&core_events, |this, _, event: &CoreEvent, cx| {
                this.on_core_event(event, cx);
            }),
        );

        let this = Self {
            status: Status::Idle,
            file_info: None,
            error: None,
            progress: None,
            timeline: timeline::Timeline::new(false, true),
            estimate: timeline::Estimate::default(),
            started_at: None,
            saved_meeting: None,
            title,
            title_edited: false,
            show_advanced: true,
            language,
            speakers,
            pipeline: cx.new(|cx| crate::views::processing::PipelineEditor::new(false, window, cx)),
            _subscriptions: subscriptions,
        };
        cx.observe(&this.pipeline, |_, _, cx| cx.notify()).detach();
        this
    }

    fn on_core_event(&mut self, event: &CoreEvent, cx: &mut Context<Self>) {
        match event.name.as_str() {
            "import-progress" => {
                if let Some(progress) = event.decode::<ImportProgress>() {
                    if self.status != Status::Processing { return; }
                    let elapsed = self.started_at.map(|t| t.elapsed()).unwrap_or_default();
                    self.timeline.update(&progress.stage, elapsed);
                    self.estimate.update(&progress.stage, &progress.message, progress.stage_percentage, elapsed);
                    self.progress = Some(progress);
                    cx.notify();
                }
            }
            "import-complete" => {
                if let Some(result) = event.decode::<ImportResult>() {
                    self.finish(result, cx);
                }
            }
            "import-error" => {
                if let Some(err) = event.decode::<ImportError>() {
                    self.timeline.finish(self.started_at.map(|t| t.elapsed()).unwrap_or_default(), false);
                    if err.retained_recording {
                        self.saved_meeting = err.meeting_id.clone();
                        shell::refresh_meetings(cx);
                    }
                    if err.cancelled {
                        self.status = Status::Idle;
                        self.progress = None;
                        with_main_window(cx, |window, cx| window.close_dialog(cx));
                        return;
                    }
                    self.status = Status::Error;
                    self.error = Some(err.error);
                    cx.notify();
                }
            }
            "import-warning" => {
                if let Some(warning) = event.decode::<ImportWarning>() {
                    toast(cx, NotificationType::Warning, warning_message(warning));
                }
            }
            _ => {}
        }
    }

    /// Keep the completed timings available until the user opens the meeting.
    fn finish(&mut self, result: ImportResult, cx: &mut Context<Self>) {
        self.timeline.finish(self.started_at.map(|t| t.elapsed()).unwrap_or_default(), true);
        self.status = Status::Complete;
        self.saved_meeting = Some(result.meeting_id);
        shell::refresh_meetings(cx);
        cx.notify();
    }

    fn begin_validate(&mut self, path: PathBuf, cx: &mut Context<Self>) {
        self.status = Status::Validating;
        self.error = None;
        cx.notify();

        let io = AppServices::global(cx).io.clone();
        cx.spawn(async move |this, cx| {
            let handle = io.spawn(async move {
                tokio::task::spawn_blocking(move || import::validate_audio_file(&path)).await
            });
            let outcome: Result<AudioFileInfo, String> = match handle.await {
                Ok(Ok(Ok(info))) => Ok(info),
                Ok(Ok(Err(e))) => Err(e.to_string()),
                Ok(Err(e)) => Err(format!("Validation task panicked: {e}")),
                Err(e) => Err(format!("Validation task panicked: {e}")),
            };
            let _ = this.update_in(cx, |this, window, cx| match outcome {
                Ok(info) => this.apply_file_info(info, window, cx),
                Err(e) => {
                    this.status = Status::Error;
                    this.error = Some(e);
                    cx.notify();
                }
            });
        })
        .detach();
    }

    fn apply_file_info(
        &mut self,
        info: AudioFileInfo,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        if !self.title_edited {
            let filename = info.filename.clone();
            self.title
                .update(cx, |state, cx| state.set_value(filename, window, cx));
        }
        self.file_info = Some(info);
        self.status = Status::Idle;
        self.error = None;
        cx.notify();
    }

    fn selected_language(&self, cx: &App) -> Option<String> {
        self.language
            .read(cx)
            .selected_value()
            .map(|v| v.to_string())
            .filter(|v| v != "auto")
    }

    fn selected_speakers(&self, cx: &App) -> i32 {
        self.speakers
            .read(cx)
            .selected_value()
            .copied()
            .unwrap_or(0)
    }

    fn start_import(&mut self, cx: &mut Context<Self>) {
        if matches!(self.status, Status::Processing | Status::Validating) {
            return;
        }
        let Some(info) = self.file_info.clone() else {
            return;
        };
        let typed_title = self.title.read(cx).value().to_string();
        let title = if typed_title.trim().is_empty() {
            info.filename.clone()
        } else {
            typed_title
        };
        let language = self.selected_language(cx);
        let config = match self.pipeline.read(cx).snapshot() {
            Ok(config) => config,
            Err(error) => {
                self.error = Some(error);
                cx.notify();
                return;
            }
        };
        let num_speakers = self.selected_speakers(cx);

        use parley_core::audio::processing::config::{Identification, SpeechEngine, SpeakerPipeline, TargetKind};
        let local = self.pipeline.read(cx).targets.iter()
            .find(|t| t.id == config.target_id).is_some_and(|t| t.kind == TargetKind::Local);
        let speech_first = local && (config.speech_engine == SpeechEngine::WhisperCpp
            || config.speaker_pipeline == SpeakerPipeline::Sherpa);
        self.timeline = timeline::Timeline::new(!speech_first, config.identification != Identification::Off);
        self.estimate = timeline::Estimate::default();
        self.started_at = Some(std::time::Instant::now());
        self.saved_meeting = None;
        self.status = Status::Processing;
        let timer_started_at = self.started_at;
        cx.spawn(async move |this, cx| loop {
            cx.background_executor().timer(std::time::Duration::from_secs(1)).await;
            let active = this.update(cx, |this, cx| {
                if this.status != Status::Processing || this.started_at != timer_started_at { return false; }
                cx.notify(); true
            });
            if !matches!(active, Ok(true)) { break; }
        }).detach();
        self.error = None;
        self.progress = None;
        cx.notify();

        let services = AppServices::global(cx);
        let sink = services.sink.clone();
        let pool = services.pool();
        let io = services.io.clone();
        io.spawn(async move {
            // `start_import_with` emits `import-error` itself on failure —
            // this task's only job is to drive it to completion.
            if let Err(e) = import::start_import_with_options(
                sink,
                pool,
                info.path,
                title,
                parley_core::audio::processing::service::ProcessingOptions {
                    config,
                    language,
                    num_speakers,
                },
            )
            .await
            {
                log::warn!("import: start_import_with failed: {e}");
            }
        });
    }

    fn cancel(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        if self.status == Status::Processing {
            import::cancel_import();
            // Wait for the terminal event so a durably saved recording remains
            // discoverable even when processing was cancelled.
            return;
        }
        window.close_dialog(cx);
    }
}

fn warning_message(warning: ImportWarning) -> String {
    match warning.details {
        Some(details) => format!("{}: {}", warning.warning, details),
        None => warning.warning,
    }
}

/// Push a toast on the main window from a context that only has `&mut App`
/// (a core-event subscription has no `Window`) — the same
/// `tray::MainWindow` global the tray menu uses to reach the window.
fn with_main_window(cx: &mut App, f: impl FnOnce(&mut Window, &mut App)) {
    crate::ui::with_main_window(cx, f);
}

fn toast(cx: &mut App, kind: NotificationType, message: String) {
    with_main_window(cx, move |window, cx| {
        window.push_notification(Notification::new().message(message).with_type(kind), cx);
    });
}

/// Open the Import Audio dialog empty — the user picks a file from inside
/// it. Used by the sidebar's "Import audio" item.
pub fn open(window: &mut Window, cx: &mut App) {
    open_with(window, cx, None);
}

/// Open the Import Audio dialog with `path` already selected and validation
/// kicked off. Used for drag-and-drop.
pub fn open_with_file(window: &mut Window, cx: &mut App, path: PathBuf) {
    open_with(window, cx, Some(path));
}

fn open_with(window: &mut Window, cx: &mut App, preselected: Option<PathBuf>) {
    if window.has_active_dialog(cx) {
        return;
    }

    let state = cx.new(|cx| ImportState::new(window, cx));
    if let Some(path) = preselected {
        state.update(cx, |state, cx| state.begin_validate(path, cx));
    }

    window.open_dialog(cx, move |dialog, window, cx| {
        render_dialog(dialog, state.clone(), window, cx)
    });
}

fn render_dialog(
    dialog: Dialog,
    state: Entity<ImportState>,
    _window: &mut Window,
    cx: &mut App,
) -> Dialog {
    let status = state.read(cx).status;

    let content_state = state.clone();
    let footer_state = state.clone();

    let title_text = match status {
        Status::Processing => "Importing Audio…",
        Status::Error => "Import Failed",
        Status::Complete => "Import Complete",
        _ => "Import Audio File",
    };

    let mut dialog = dialog
        .title(Label::new(title_text).text_lg().font_semibold())
        .width(px(620.))
        .close_button(status != Status::Processing)
        .overlay_closable(status != Status::Processing)
        .keyboard(status != Status::Processing)
        .content(move |content, window, cx| render_body(content, &content_state, window, cx));

    let has_file = state.read(cx).file_info.is_some() && state.read(cx).pipeline.read(cx).ready();
    dialog = dialog.footer(render_footer(footer_state, status, has_file, state.read(cx).saved_meeting.is_some()));
    dialog
}

fn render_body<'a>(
    content: gpui_kit::component::dialog::DialogContent,
    state: &Entity<ImportState>,
    window: &mut Window,
    cx: &mut App,
) -> gpui_kit::component::dialog::DialogContent {
    let s = state.read(cx);
    let status = s.status;
    let file_info = s.file_info.clone();
    let error = s.error.clone();
    let progress = s.progress.clone();
    let estimate = s.estimate.clone();
    let show_advanced = s.show_advanced;
    let started_at = s.started_at;
    let timeline = s.timeline.clone();

    let mut body = v_flex().gap_3().w_full();

    match status {
        Status::Error => {
            body = body.child(
                Label::new(error.unwrap_or_default())
                    .text_sm()
                    .text_color(cx.theme().danger),
            );
        }
        Status::Processing | Status::Complete => {}
        Status::Idle | Status::Validating => {
            if let Some(error) = error {
                body = body.child(Label::new(error).text_sm().text_color(cx.theme().danger));
            }
            body = body.child(file_picker_section(
                state,
                file_info.as_ref(),
                status == Status::Validating,
                window,
                cx,
            ));

            if file_info.is_some() {
                body = body.child(advanced_options_section(state, show_advanced, cx));
            }
        }
    }

    if started_at.is_some() {
        let now = timeline.finished_at.unwrap_or_else(|| started_at.unwrap().elapsed());
        body = body.child(Label::new(format!("Elapsed: {}", logic::format_duration(now.as_secs_f64()))).text_sm());
        let mut steps = v_flex().id("import-steps").w_full().max_h(px(360.)).overflow_y_scrollbar().gap_2();
        for step in &timeline.steps {
            let elapsed = step.elapsed(timeline.active_since, now);
            let marker = match step.state {
                timeline::State::Done => "✓", timeline::State::Failed => "!",
                timeline::State::Skipped => "—", _ => "○",
            };
            let mut row = h_flex().w_full().gap_2().items_center();
            row = if step.state == timeline::State::Active { row.child(Spinner::new().small()) }
                else { row.child(Label::new(marker).text_color(match step.state {
                    timeline::State::Done => cx.theme().success,
                    timeline::State::Failed => cx.theme().danger,
                    _ => cx.theme().muted_foreground,
                })) };
            let timing = match step.state {
                timeline::State::Pending => "Pending".into(),
                timeline::State::Skipped => "Not run".into(),
                timeline::State::Active => match progress.as_ref().and_then(|p| p.stage_percentage) {
                    Some(percent) => format!("Subtask {percent}% · {} elapsed", logic::format_duration(elapsed.as_secs_f64())),
                    None => format!("{} elapsed", logic::format_duration(elapsed.as_secs_f64())),
                },
                _ => logic::format_duration(elapsed.as_secs_f64()),
            };
            steps = steps.child(row.child(div().flex_1().child(step.phase.label()))
                .child(Label::new(timing).text_xs().text_color(cx.theme().muted_foreground)));
            if step.state == timeline::State::Active {
                if let Some(p) = &progress {
                    steps = steps.child(Label::new(p.message.clone()).text_sm().text_color(cx.theme().muted_foreground));
                    if let Some(percent) = p.stage_percentage {
                        steps = steps.child(Progress::new("import-subtask-progress").value(logic::progress_percent(percent)))
                            .child(Label::new(match estimate.remaining(now) {
                                Some(remaining) => format!("{percent}% · About {} remaining in this subtask", logic::format_duration(remaining.as_secs_f64())),
                                None if percent == 0 => "Waiting for the first progress measurement…".into(),
                                None if percent < 100 => format!("{percent}% · Updating the remaining-time estimate…"),
                                None => format!("{percent}% of this subtask"),
                            }).text_xs());
                    } else {
                        steps = steps.child(Label::new("Percentage and remaining time are not available for this subtask.")
                            .text_xs().text_color(cx.theme().muted_foreground));
                    }
                }
            }
        }
        body = body.child(steps);
    }

    content.child(body)
}

fn render_footer(state: Entity<ImportState>, status: Status, has_file: bool, has_saved_meeting: bool) -> AnyElement {
    // Keep completed timings visible until the user opens the saved meeting.
    if status == Status::Complete || (status == Status::Error && has_saved_meeting) {
        let saved = state.clone();
        return h_flex().w_full().justify_end().gap_2()
            .child(Button::new("import-close-result").outline().label("Close")
                .on_click(|_, window, cx| window.close_dialog(cx)))
            .child(Button::new("import-open-result").label("Open meeting").on_click(move |_, window, cx| {
                if let Some(id) = saved.read(cx).saved_meeting.clone() {
                    window.close_dialog(cx);
                    shell::navigate(Route::Meeting(id), cx);
                }
            })).into_any_element();
    }
    match status {
        Status::Error => h_flex()
            .w_full()
            .justify_end()
            .gap_2()
            .child(
                Button::new("import-close")
                    .outline()
                    .label("Close")
                    .on_click(move |_, window, cx| window.close_dialog(cx)),
            )
            .child(Button::new("import-retry").label("Try Again").on_click({
                let state = state.clone();
                move |_, _, cx| {
                    state.update(cx, |state, cx| {
                        state.status = Status::Idle;
                        state.started_at = None;
                        state.progress = None;
                        state.error = None;
                        cx.notify();
                    });
                }
            }))
            .into_any_element(),
        Status::Complete => unreachable!(),
        Status::Processing => h_flex()
            .w_full()
            .justify_end()
            .child(
                Button::new("import-cancel")
                    .outline()
                    .label("Cancel")
                    .on_click({
                        let state = state.clone();
                        move |_, window, cx| {
                            state.update(cx, |state, cx| state.cancel(window, cx));
                        }
                    }),
            )
            .into_any_element(),
        Status::Idle | Status::Validating => h_flex()
            .w_full()
            .justify_end()
            .gap_2()
            .child(
                Button::new("import-dismiss")
                    .outline()
                    .label("Cancel")
                    .on_click(move |_, window, cx| window.close_dialog(cx)),
            )
            .child(
                Button::new("import-start")
                    .label("Import")
                    .disabled(!has_file)
                    .on_click({
                        let state = state.clone();
                        move |_, _, cx| {
                            state.update(cx, |state, cx| state.start_import(cx));
                        }
                    }),
            )
            .into_any_element(),
    }
}

fn file_picker_section(
    state: &Entity<ImportState>,
    file_info: Option<&AudioFileInfo>,
    validating: bool,
    window: &mut Window,
    cx: &mut App,
) -> impl IntoElement {
    if let Some(info) = file_info {
        let title_state = state.read(cx).title.clone();
        v_flex()
            .gap_3()
            .p_3()
            .rounded_md()
            .bg(cx.theme().muted)
            .child(
                v_flex()
                    .gap_1()
                    .child(Label::new(info.filename.clone()).text_sm().font_semibold())
                    .child(
                        h_flex()
                            .gap_3()
                            .child(
                                Label::new(logic::format_duration(info.duration_seconds))
                                    .text_xs()
                                    .text_color(cx.theme().muted_foreground),
                            )
                            .child(
                                Label::new(logic::format_file_size(info.size_bytes))
                                    .text_xs()
                                    .text_color(cx.theme().muted_foreground),
                            )
                            .child(
                                Label::new(info.format.clone())
                                    .text_xs()
                                    .text_color(cx.theme().info),
                            ),
                    ),
            )
            .child(
                v_flex()
                    .gap_1()
                    .child(Label::new("Meeting Title").text_sm().font_medium())
                    .child(Input::new(&title_state)),
            )
            .child(
                Button::new("import-choose-different")
                    .outline()
                    .label("Choose Different File")
                    .on_click({
                        let state = state.clone();
                        move |_, window, cx| pick_file(&state, window, cx)
                    }),
            )
            .into_any_element()
    } else {
        let _ = window;
        v_flex()
            .gap_2()
            .items_center()
            .p_6()
            .rounded_md()
            .border_1()
            .border_dashed()
            .border_color(cx.theme().border)
            .child(
                Button::new("import-select-file")
                    .icon(gpui_kit::assets::IconName::Upload)
                    .label(if validating {
                        "Validating…"
                    } else {
                        "Select Audio File"
                    })
                    .disabled(validating)
                    .on_click({
                        let state = state.clone();
                        move |_, window, cx| pick_file(&state, window, cx)
                    }),
            )
            .child(
                Label::new(
                    parley_core::audio::constants::AUDIO_EXTENSIONS
                        .join(", ")
                        .to_uppercase(),
                )
                .text_xs()
                .text_color(cx.theme().muted_foreground),
            )
            .into_any_element()
    }
}

fn advanced_options_section(
    state: &Entity<ImportState>,
    show_advanced: bool,
    cx: &mut App,
) -> impl IntoElement {
    let s = state.read(cx);
    let language = s.language.clone();
    let speakers = s.speakers.clone();
    let pipeline = s.pipeline.clone();

    let mut section = v_flex().gap_2().w_full().child(
        Button::new("import-toggle-advanced")
            .ghost()
            .label(if show_advanced {
                "Hide advanced options"
            } else {
                "Advanced options"
            })
            .on_click({
                let state = state.clone();
                move |_, _, cx| {
                    state.update(cx, |state, cx| {
                        state.show_advanced = !state.show_advanced;
                        cx.notify();
                    });
                }
            }),
    );

    if show_advanced {
        let mut options = v_flex().gap_3().w_full();
        options = options.child(
            v_flex()
                .gap_1()
                .child(Label::new("Language").text_sm().font_medium())
                .child(Select::new(&language)),
        );
        options = options.child(
            v_flex()
                .gap_1()
                .child(Label::new("Number of speakers").text_sm().font_medium())
                .child(Select::new(&speakers)),
        );
        options = options.child(pipeline);
        section = section.child(options);
    }

    section
}

fn pick_file(state: &Entity<ImportState>, _window: &mut Window, cx: &mut App) {
    let rx = cx.prompt_for_paths(PathPromptOptions {
        files: true,
        directories: false,
        multiple: false,
        prompt: Some("Select an audio file".into()),
    });
    let state = state.clone();
    cx.spawn(async move |cx| {
        let picked = rx.await;
        if let Ok(Ok(Some(mut paths))) = picked {
            if let Some(path) = paths.drain(..).next() {
                let _ = state.update(cx, |state, cx| state.begin_validate(path, cx));
            }
        }
    })
    .detach();
}
