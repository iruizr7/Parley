//! Meeting page: editable header (title + delete), a virtualized transcript
//! panel, and a right-hand pane with two tabs: the summary (generate/
//! regenerate, WYSIWYG editing, copy/export, retranscription), "Ask"
//! (chat with the meeting, see `ask.rs`) and "Insights" (talk-time
//! analytics, see `insights.rs`).
//!
//! Loading, generation and polling are all guarded by a `generation`
//! counter bumped on every [`MeetingView::load`] — async work started for an
//! older meeting id checks it before touching `self` so a fast
//! double-navigation can't clobber the currently-shown meeting with a
//! stale response (see `cx.spawn` closures below).

mod ask;
mod calendar_format;
mod followup;
mod format;
mod moments;
mod insights;
mod speaker_chip;
mod speaker_filter;
mod transcript_search;

use std::time::Duration;

use gpui_kit::component::{
    ActiveTheme, Disableable as _, IconName, WindowExt as _,
    button::{Button, ButtonVariants as _},
    checkbox::Checkbox,
    h_flex,
    input::{Input, InputEvent, InputState},
    select::{Select, SelectEvent, SelectItem, SelectState, SearchableVec},
    IndexPath,
    notification::Notification,
    tab::{Tab, TabBar},
    text::{TextView, TextViewState},
    v_flex,
};
use gpui_kit::base::StyledExt as _;
use gpui_kit::component::IconNameExt as _;
use gpui_kit::component::Sizable as _;
use gpui_kit::component::scroll::ScrollableElement as _;
use gpui_kit::prelude::FluentBuilder as _;
use gpui_kit::*;
use parley_core::calendar::models::CalendarEvent;
use parley_core::calendar::repository::CalendarRepository;
use parley_core::calendar::service::link_meeting_with_snapshot;
use parley_core::database::models::{ActionItem, MeetingDetails, MeetingNote, MeetingTranscript, VoiceProfile};
use parley_core::database::repositories::action_item::{
    ActionItemsRepository, NewActionItem, SOURCE_MANUAL as ACTION_ITEM_SOURCE_MANUAL, STATUS_DONE, STATUS_OPEN,
};
use parley_core::database::repositories::meeting::MeetingsRepository;
use parley_core::database::repositories::meeting_note::{MeetingNotesRepository, SOURCE_MANUAL as NOTE_SOURCE_MANUAL};
use parley_core::database::repositories::setting::{SettingsRepository, KEY_UI_CONFIG};
use parley_core::database::repositories::summary::SummaryProcessesRepository;
use parley_core::database::repositories::voice_profile::VoiceProfilesRepository;
use parley_core::speaker_diarization::service::{
    assign_segment_speaker_core, merge_cluster_into_profile_core, promote_speaker_to_profile_core,
    unassign_speaker_core,
};
use parley_core::summary::markdown_export;
use parley_core::summary::service::SummaryService;
use zorite_editor::{EditorState, SyntaxStyle};

use crate::app_state::AppServices;
use crate::runtime::Io;
use crate::shell::{self, Route};

/// ±N days around the meeting's `created_at` when fetching nearby calendar
/// events for the link picker. Mirrors `CalendarEventPicker.tsx`'s
/// `WINDOW_DAYS`.
const CALENDAR_WINDOW_DAYS: i64 = 7;

/// Extra space above/below the transcript panel's visible area that
/// `transcript_list_state` measures eagerly, so scrolling doesn't pop in
/// unmeasured rows. Mirrors `message_scroller.rs`'s `LIST_OVERDRAW`.
const TRANSCRIPT_LIST_OVERDRAW: Pixels = px(400.);

#[derive(Clone, PartialEq)]
struct SpeakerOption {
    id: String,
    label: SharedString,
}
impl SelectItem for SpeakerOption {
    type Value = String;
    fn title(&self) -> SharedString { self.label.clone() }
    fn value(&self) -> &String { &self.id }
}
type SpeakerPicker = SelectState<SearchableVec<SpeakerOption>>;

/// One open speaker-edit panel: rename a named voice profile, or promote /
/// merge an unnamed "Speaker N" cluster. Mirrors
/// `EditableSpeakerChip.tsx`'s per-chip popover state, scoped to a single
/// segment at a time (see [`MeetingView::speaker_edit`]).
#[derive(Clone)]
struct SpeakerEditState {
    /// Uniquely identifies which chip this panel belongs to (segment id +
    /// label), so a stray async response for a since-closed/reopened panel
    /// is dropped instead of applied.
    key: String,
    speaker: String,
    voice_profile_id: Option<String>,
    name_input: Entity<InputState>,
    email_input: Entity<InputState>,
    profiles: Vec<VoiceProfile>,
    profiles_loading: bool,
    picker: Option<Entity<SpeakerPicker>>,
    picker_subscription: Option<std::sync::Arc<Subscription>>,
    has_ranking: bool,
    /// `Some(profile_id)` = fold this cluster into that existing profile;
    /// `None` = create/rename via `name_input`/`email_input`. Only
    /// meaningful for either a whole-group or single-segment reassignment.
    merge_target: Option<String>,
    saving: bool,
    error: Option<String>,
    /// Where the chip was clicked, in window coordinates — the popover's
    /// anchor position (see [`render_floating_speaker_edit`]).
    click_position: Point<Pixels>,
    /// This meeting's linked-calendar-event attendees, offered as one-click
    /// name/email fills — mirrors `EditableSpeakerChip.tsx`'s `attendees`
    /// state. Computed once when the panel opens from
    /// [`MeetingView::calendar_event`], which is already loaded for the
    /// calendar row.
    attendees: Vec<speaker_chip::AttendeeSuggestion>,
    /// The clicked segment, for an "only this segment" save.
    segment_id: String,
    /// How many of this meeting's segments carry `speaker` — the scope
    /// switch is only offered when there's more than one.
    cluster_size: usize,
    /// `true` = relabel just `segment_id` (a cluster holding two voices);
    /// `false` = the whole cluster, as before.
    single_segment: bool,
}

/// The full Lucide catalog (as opposed to `gpui_kit::component::IconName`,
/// which only carries a curated default subset — most of the icons this
/// page needs aren't in it).
type Lucide = gpui_kit::assets::IconName;

/// How far along the current meeting's summary is. Distinct from "there is
/// no summary yet", which is `Idle` with empty `summary_state` text.
#[derive(Clone, PartialEq)]
enum SummaryPhase {
    Idle,
    /// Initial `summary_processes` fetch for a newly-loaded meeting.
    Loading,
    /// A generation is in flight (started here, or resumed because the row
    /// was already "processing" when the page loaded).
    Generating,
    Error(String),
}

pub struct MeetingView {
    meeting_id: Option<String>,
    /// Bumped on every `load()`; async completions compare it to the
    /// current value before applying, so a stale in-flight fetch for a
    /// meeting the user has since navigated away from is dropped.
    generation: u64,
    title: String,
    title_input: Entity<InputState>,
    editing_title: bool,
    created_at: Option<chrono::DateTime<chrono::Utc>>,
    /// Recording folder on disk, if any (needed for retranscription; absent
    /// when auto_save was off, or the meeting predates folder tracking).
    meeting_folder_path: Option<String>,
    transcripts: Vec<MeetingTranscript>,
    selected_speakers: std::collections::BTreeSet<speaker_filter::SpeakerKey>,
    transcript_search_input: Entity<InputState>,
    transcript_query: String,
    search_reset_pending: bool,
    /// Backs the transcript panel's variable-height `list()` (see
    /// `render_transcript`) — rows wrap to a different number of lines
    /// depending on segment text length and pane width, so this can't be a
    /// `uniform_list` (which assumes every row shares one measured height).
    /// Reset alongside `transcripts` every time that vector is replaced.
    transcript_list_state: ListState,
    transcripts_loading: bool,
    load_error: Option<String>,
    summary_state: Entity<TextViewState>,
    /// Source-of-truth markdown for the summary: what `summary_state`
    /// renders, what "Edit" seeds the editor with, and what "Copy summary"
    /// and export read. `TextViewState` doesn't expose a text getter.
    summary_markdown: String,
    summary_has_content: bool,
    summary_phase: SummaryPhase,
    editing_summary: bool,
    summary_editor: Entity<EditorState>,
    saving_summary: bool,
    _poll_task: Option<Task<()>>,

    /// Available summary templates (id, name, description), loaded once at
    /// startup via `summary::templates::list_templates`.
    templates: Vec<(String, String, String)>,
    selected_template: String,
    /// Set once the user manually cycles the template picker, so the async
    /// default-setting fetch (`load_default_template`) doesn't clobber their
    /// choice if it resolves afterwards.
    template_user_selected: bool,
    custom_prompt: Entity<InputState>,
    show_custom_prompt: bool,

    retranscribe_open: bool,
    retranscribe_language: String,
    retranscribe_speakers: Entity<InputState>,
    pipeline_editor: Entity<crate::views::processing::PipelineEditor>,
    identify_editor: Entity<crate::views::processing::PipelineEditor>,
    processing_history: Entity<crate::views::processing_history::ProcessingHistory>,
    history_open: bool,
    identify_open: bool,
    identify_busy: bool,
    identity_suggestions: Vec<parley_core::audio::processing::service::IdentitySuggestion>,
    identity_group_version: Option<String>,
    identity_request: u64,
    review_identity_replacements: bool,
    processing_cancel: std::sync::Arc<std::sync::atomic::AtomicBool>,
    processing_operation: &'static str,
    retranscribe_in_progress: bool,
    /// "Re-detect speakers" running for this meeting (#115).
    rediarize_in_progress: bool,
    job_progress: Entity<crate::views::task_progress::TaskProgress>,
    retranscribe_progress_pct: u32,
    retranscribe_progress_message: String,
    retranscribe_error: Option<String>,

    // -- Calendar event link --------------------------------------------
    calendar_event: Option<CalendarEvent>,
    calendar_loading: bool,
    calendar_mutating: bool,
    calendar_picker_open: bool,
    /// `None` while the picker's event fetch is in flight.
    calendar_picker_events: Option<Vec<CalendarEvent>>,
    calendar_picker_query: Entity<InputState>,
    calendar_picker_error: Option<String>,

    // -- Per-segment speaker chip edit -----------------------------------
    speaker_edit: Option<SpeakerEditState>,

    // -- Per-segment audio playback ---------------------------------------
    /// Segment id currently playing, if any.
    playing_segment: Option<String>,
    /// Segment id whose clip is being extracted (between click and playback
    /// actually starting), if any.
    loading_segment: Option<String>,

    // -- Confidence indicator ---------------------------------------------
    /// `ui_config.showConfidenceIndicator`, read once per `load()` — mirrors
    /// `ConfidenceIndicator.tsx`'s `showIndicator` gate.
    show_confidence: bool,

    // -- Meeting notes ------------------------------------------------------
    // Mirrors `MeetingNotesPanel.tsx`: append-only (no edit; delete+re-add).
    notes: Vec<MeetingNote>,
    notes_composing: bool,
    notes_draft: Entity<InputState>,
    notes_saving: bool,

    // -- Per-meeting action items --------------------------------------------
    // Mirrors `ActionItemsPanel.tsx`.
    action_items: Vec<ActionItem>,
    action_items_loading: bool,
    /// Item currently shown with an editable text field, if any.
    ai_editing_id: Option<String>,
    ai_edit_input: Entity<InputState>,
    ai_adding: bool,
    ai_add_input: Entity<InputState>,
    ai_extracting: bool,

    // -- Summary typewriter reveal --------------------------------------------
    // `on_summary_stream` buffers incoming deltas here instead of pushing
    // them straight to `summary_state`; `_summary_reveal_task` drains the
    // buffer at `format::SUMMARY_REVEAL_INTERVAL_MS`, mirroring
    // `useTranscriptStreaming.ts`'s pacing (see `format.rs`).
    summary_reveal_pending: String,
    _summary_reveal_task: Option<Task<()>>,

    // -- Right-hand pane: Summary | Ask | Insights ------------------------------
    right_tab: RightTab,
    /// "Ask Parley" chat (see `ask.rs`).
    ask_panel: Entity<ask::AskPanel>,
    /// Chapters & bookmarks strip above the transcript (see `moments.rs`).
    moments_panel: Entity<moments::MomentsPanel>,
    /// Talk-time analytics (see `insights.rs`).
    insights_panel: Entity<insights::InsightsPanel>,
    /// Transcript row briefly highlighted after a citation jump.
    highlighted_segment: Option<String>,
    _highlight_task: Option<Task<()>>,
    /// A moment to show once the transcript has loaded (set when a
    /// cross-meeting "Ask Parley" citation opens this meeting).
    pending_jump: Option<f64>,

    _subscriptions: Vec<Subscription>,
}

/// Which panel the right-hand pane shows.
#[derive(Clone, Copy, PartialEq, Eq)]
enum RightTab {
    Summary,
    Ask,
    Insights,
}

impl MeetingView {
    pub fn new(window: &mut Window, cx: &mut Context<Self>) -> Self {
        let title_input = cx.new(|cx| InputState::new(window, cx).placeholder("Meeting title"));
        let transcript_search_input = cx.new(|cx| InputState::new(window, cx).placeholder("Search transcript…"));
        let retranscribe_language = "auto".to_string();
        let retranscribe_speakers = cx.new(|cx| InputState::new(window, cx).placeholder("Auto"));
        let summary_state = cx.new(|cx| TextViewState::markdown("", cx));
        let summary_editor = cx.new(|cx| {
            let mut editor = EditorState::new(window, cx).with_placeholder("No summary yet…");
            editor.set_markdown_style(syntax_style(cx), cx);
            editor
        });

        let ask_panel = cx.new(|cx| ask::AskPanel::new(window, cx));
        let moments_panel = cx.new(|cx| moments::MomentsPanel::new(window, cx));
        let insights_panel = cx.new(|cx| insights::InsightsPanel::new(window, cx));
        let core_events = AppServices::global(cx).core_events.clone();
        let mut subscriptions = vec![cx.subscribe(&core_events, |this, _, event, cx| {
            match event.name.as_str() {
                "summary-templates-changed" => {
                    this.templates = parley_core::summary::templates::list_templates();
                    if !this.templates.iter().any(|(id, _, _)| *id == this.selected_template) {
                        this.selected_template = "standard_meeting".into();
                        this.template_user_selected = false;
                    }
                    if !this.template_user_selected && !matches!(this.summary_phase, SummaryPhase::Generating) {
                        this.load_default_template(cx);
                    }
                    cx.notify();
                }
                "summary-stream" => this.on_summary_stream(event, cx),
                "retranscription-progress" => this.on_retranscription_progress(event, cx),
                "retranscription-complete" => this.on_retranscription_complete(event, cx),
                "retranscription-error" => this.on_retranscription_error(event, cx),
                "action-items-extracted" => this.on_action_items_extracted(event, cx),
                parley_core::audio::playback::PLAYBACK_ENDED_EVENT => this.on_segment_playback_ended(cx),
                _ => {}
            }
        })];
        subscriptions.push(cx.subscribe(&ask_panel, |this, _, event: &ask::AskPanelEvent, cx| {
            let ask::AskPanelEvent::JumpTo(seconds) = event;
            this.jump_and_play(*seconds, cx);
        }));
        subscriptions.push(cx.subscribe(&moments_panel, |this, _, event: &moments::MomentsEvent, cx| {
            let moments::MomentsEvent::JumpTo(seconds) = event;
            this.jump_and_play(*seconds, cx);
        }));
        // Chapter/bookmark markers change transcript row heights.
        subscriptions.push(cx.observe(&moments_panel, |this, _, cx| {
            this.transcript_list_state.remeasure();
            cx.notify();
        }));
        subscriptions.push(cx.subscribe(&insights_panel, |this, _, event: &insights::InsightsPanelEvent, cx| {
            let insights::InsightsPanelEvent::JumpTo(seconds) = event;
            this.jump_and_play(*seconds, cx);
        }));

        subscriptions.push(cx.subscribe(&transcript_search_input, |this, input, event: &InputEvent, cx| {
            if matches!(event, InputEvent::Change) {
                let query = input.read(cx).value().to_string();
                if this.transcript_query != query {
                    this.transcript_query = query;
                    this.speaker_edit = None;
                    this.refresh_speaker_filter();
                    this.transcript_list_state.scroll_to(ListOffset { item_ix: 0, offset_in_item: px(0.) });
                    cx.notify();
                }
            }
        }));

        let templates = parley_core::summary::templates::list_templates();
        let template_ids: Vec<String> = templates.iter().map(|(id, _, _)| id.clone()).collect();
        let selected_template = format::resolve_default_template(None, &template_ids);
        let custom_prompt =
            cx.new(|cx| InputState::new(window, cx).placeholder("Custom instructions (optional)…"));
        let calendar_picker_query =
            cx.new(|cx| InputState::new(window, cx).placeholder("Search events…"));
        let notes_draft = cx.new(|cx| InputState::new(window, cx).placeholder("Add a note…"));
        let ai_edit_input = cx.new(|cx| InputState::new(window, cx).placeholder("Action item text"));
        let ai_add_input = cx.new(|cx| InputState::new(window, cx).placeholder("Add an action item…"));

        let view = Self {
            meeting_id: None,
            generation: 0,
            title: String::new(),
            title_input,
            editing_title: false,
            created_at: None,
            meeting_folder_path: None,
            transcripts: Vec::new(),
            selected_speakers: Default::default(),
            transcript_search_input,
            transcript_query: String::new(),
            search_reset_pending: false,
            transcript_list_state: ListState::new(0, ListAlignment::Top, TRANSCRIPT_LIST_OVERDRAW),
            transcripts_loading: false,
            load_error: None,
            summary_state,
            summary_markdown: String::new(),
            summary_has_content: false,
            summary_phase: SummaryPhase::Idle,
            editing_summary: false,
            summary_editor,
            saving_summary: false,
            _poll_task: None,
            templates,
            selected_template,
            template_user_selected: false,
            custom_prompt,
            show_custom_prompt: false,
            retranscribe_open: false,
            retranscribe_language,
            retranscribe_speakers,
            pipeline_editor: cx
                .new(|cx| crate::views::processing::PipelineEditor::new(false, window, cx)),
            identify_editor: cx.new(|cx| {
                let mut editor = crate::views::processing::PipelineEditor::new(false, window, cx);
                editor.set_identification_only(cx);
                editor
            }),
            processing_history: cx
                .new(|_| crate::views::processing_history::ProcessingHistory::new()),
            history_open: false,
            identify_open: false,
            identify_busy: false,
            identity_suggestions: Vec::new(),
            identity_group_version: None,
            identity_request: 0,
            review_identity_replacements: false,
            processing_cancel: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
            processing_operation: "transcribe",
            retranscribe_in_progress: false,
            rediarize_in_progress: false,
            job_progress: cx.new(crate::views::task_progress::TaskProgress::new),
            retranscribe_progress_pct: 0,
            retranscribe_progress_message: String::new(),
            retranscribe_error: None,
            calendar_event: None,
            calendar_loading: false,
            calendar_mutating: false,
            calendar_picker_open: false,
            calendar_picker_events: None,
            calendar_picker_query,
            calendar_picker_error: None,
            speaker_edit: None,
            playing_segment: None,
            loading_segment: None,
            show_confidence: false,
            notes: Vec::new(),
            notes_composing: false,
            notes_draft,
            notes_saving: false,
            action_items: Vec::new(),
            action_items_loading: false,
            ai_editing_id: None,
            ai_edit_input,
            ai_adding: false,
            ai_add_input,
            ai_extracting: false,
            summary_reveal_pending: String::new(),
            _summary_reveal_task: None,
            right_tab: RightTab::Summary,
            ask_panel,
            moments_panel,
            insights_panel,
            highlighted_segment: None,
            _highlight_task: None,
            pending_jump: None,
            _subscriptions: subscriptions,
        };
        cx.subscribe(
            &view.processing_history,
            |this, _, _: &crate::views::processing_history::ProcessingResultChanged, cx| {
                this.reload_transcript(cx)
            },
        )
        .detach();
        cx.observe(&view.pipeline_editor, |_, _, cx| cx.notify())
            .detach();
        cx.observe(&view.identify_editor, |_, _, cx| cx.notify())
            .detach();
        view.load_default_template(cx);
        view
    }

    /// Fetch the stored default-template setting (if the Summary settings
    /// page has written one — see `views/settings/summary.rs`) and apply it
    /// as the picker's initial selection. A no-op if there's no pool yet or
    /// no such setting; the picker already has a sane default from
    /// `resolve_default_template(None, ..)`.
    fn load_default_template(&self, cx: &mut Context<Self>) {
        let Some(pool) = AppServices::global(cx).pool() else {
            return;
        };
        let io = Io::global(cx);
        let template_ids: Vec<String> = self.templates.iter().map(|(id, _, _)| id.clone()).collect();
        cx.spawn(async move |this, cx| {
            let configured = io
                .spawn(async move {
                    SettingsRepository::get_setting::<String>(
                        &pool,
                        parley_core::database::repositories::setting::KEY_DEFAULT_SUMMARY_TEMPLATE,
                    ).await
                })
                .await;
            let default = match configured {
                Ok(Ok(Some(id))) => format::resolve_default_template(Some(&id), &template_ids),
                _ => format::resolve_default_template(None, &template_ids),
            };
            let _ = this.update(cx, |this, cx| {
                // Don't clobber a selection the user already made while this
                // was in flight.
                if !this.template_user_selected {
                    this.selected_template = default;
                    cx.notify();
                }
            });
        })
        .detach();
    }

    /// Show the meeting with `id` (called by the shell on navigation).
    pub fn load(&mut self, id: String, cx: &mut Context<Self>) {
        self.generation += 1;
        let generation = self.generation;

        self.meeting_id = Some(id.clone());
        self.identify_editor
            .update(cx, |editor, cx| editor.set_meeting(Some(id.clone()), cx));
        self.identify_open = false;
        self.identify_busy = false;
        self.identity_suggestions.clear();
        self.identity_group_version = None;
        self.review_identity_replacements = false;
        self.history_open = false;
        self.processing_operation = "transcribe";
        self.processing_history
            .update(cx, |history, cx| history.load(id.clone(), cx));
        self.title = String::new();
        self.editing_title = false;
        self.created_at = None;
        self.meeting_folder_path = None;
        self.transcripts = Vec::new();
        self.selected_speakers.clear();
        self.transcript_query.clear();
        self.search_reset_pending = true;
        self.transcript_list_state.reset(0);
        self.transcripts_loading = true;
        self.load_error = None;
        self.summary_phase = SummaryPhase::Loading;
        self.summary_has_content = false;
        self.summary_markdown = String::new();
        self.editing_summary = false;
        self.saving_summary = false;
        self._poll_task = None;
        self.retranscribe_open = false;
        self.retranscribe_in_progress = false;
        self.rediarize_in_progress = false;
        self.retranscribe_error = None;
        self.job_progress.update(cx, |progress, cx| progress.finish(cx));
        self.retranscribe_progress_pct = 0;
        self.retranscribe_progress_message = String::new();
        self.calendar_event = None;
        self.calendar_loading = true;
        self.calendar_picker_open = false;
        self.calendar_picker_events = None;
        self.calendar_picker_error = None;
        self.speaker_edit = None;
        self.playing_segment = None;
        self.loading_segment = None;
        self.notes = Vec::new();
        self.notes_composing = false;
        self.notes_saving = false;
        self.action_items = Vec::new();
        self.action_items_loading = true;
        self.ai_editing_id = None;
        self.ai_adding = false;
        self.ai_extracting = false;
        self.summary_reveal_pending.clear();
        self._summary_reveal_task = None;
        self.highlighted_segment = None;
        self._highlight_task = None;
        self.pending_jump = None;
        self.summary_state.update(cx, |state, cx| state.set_text("", cx));
        self.ask_panel.update(cx, |panel, cx| panel.set_meeting(id.clone(), cx));
        self.moments_panel.update(cx, |panel, cx| panel.set_meeting(id.clone(), cx));
        let insights_visible = self.right_tab == RightTab::Insights;
        self.insights_panel.update(cx, |panel, cx| panel.set_meeting(id.clone(), insights_visible, cx));
        cx.notify();

        let Some(pool) = AppServices::global(cx).pool() else {
            self.load_error = Some("No database yet — finish onboarding first.".into());
            self.transcripts_loading = false;
            return;
        };
        let io = Io::global(cx);

        // Meeting + transcripts + folder path (for retranscription).
        {
            let pool = pool.clone();
            let id = id.clone();
            let io = io.clone();
            cx.spawn(async move |this, cx| {
                let pool_meta = pool.clone();
                let id_meta = id.clone();
                let result = io
                    .spawn(async move { MeetingsRepository::get_meeting(&pool, &id).await })
                    .await;
                let metadata = io
                    .spawn(async move { MeetingsRepository::get_meeting_metadata(&pool_meta, &id_meta).await })
                    .await;
                let _ = this.update(cx, |this, cx| {
                    if this.generation != generation {
                        return;
                    }
                    this.transcripts_loading = false;
                    match result {
                        Ok(Ok(Some(details))) => this.apply_meeting_details(details, cx),
                        Ok(Ok(None)) => this.load_error = Some("Meeting not found".to_string()),
                        Ok(Err(e)) => this.load_error = Some(format!("Failed to load meeting: {e}")),
                        Err(e) => this.load_error = Some(format!("Failed to load meeting: {e}")),
                    }
                    if let Ok(Ok(Some(meta))) = metadata {
                        this.meeting_folder_path = meta.folder_path;
                        if this.load_error.is_none()
                            && this.transcripts.is_empty()
                            && this.meeting_folder_path.is_some()
                            && meta.status != "recording"
                        {
                            this.show_processing_options(cx);
                        }
                    }
                    cx.notify();
                });
            })
            .detach();
        }

        // Existing summary (if any), and resume polling if one is already
        // in flight (e.g. started before navigating away and back).
        self.refresh_summary(generation, cx);
        self.load_calendar_event(generation, cx);
        self.load_notes(generation, cx);
        self.load_action_items(generation, cx);
        self.load_confidence_setting(generation, cx);
        self.load_identity_suggestions(cx);

        // A clip from the previous meeting shouldn't keep playing once we've
        // navigated away from its transcript.
        parley_core::audio::playback::stop();
    }

    /// An "Ask Parley" citation was clicked: bring the segment said at
    /// `seconds` to the top of the transcript and highlight it briefly.
    fn jump_to_time(&mut self, seconds: f64, cx: &mut Context<Self>) {
        let Some(ix) = format::transcript_index_for_time(&self.transcripts, seconds) else {
            return;
        };
        let mut turns = self.visible_transcript_turns();
        if !turns.iter().any(|turn| turn.rows.contains(&ix)) {
            // A chapter/bookmark link must reveal its target, even when hidden.
            self.selected_speakers.clear();
            self.transcript_query.clear();
            self.search_reset_pending = true;
            turns = self.visible_transcript_turns();
            self.transcript_list_state.reset(turns.len());
        }
        self.transcript_list_state.scroll_to(ListOffset {
            item_ix: turns.iter().position(|turn| turn.rows.contains(&ix)).unwrap_or(0),
            offset_in_item: px(0.),
        });
        self.highlighted_segment = Some(self.transcripts[ix].id.clone());
        let generation = self.generation;
        self._highlight_task = Some(cx.spawn(async move |this, cx| {
            cx.background_executor().timer(Duration::from_millis(2500)).await;
            let _ = this.update(cx, |this, cx| {
                if this.generation == generation {
                    this.highlighted_segment = None;
                    cx.notify();
                }
            });
        }));
        cx.notify();
    }

    /// A chapter, bookmark, citation or monologue was clicked: show the
    /// transcript there and play the recording from that line, both
    /// channels, for up to [`JUMP_PLAY_SECS`]. The row's stop button (or
    /// any other playback) stops it.
    fn jump_and_play(&mut self, seconds: f64, cx: &mut Context<Self>) {
        self.jump_to_time(seconds, cx);
        let Some(ix) = format::transcript_index_for_time(&self.transcripts, seconds) else {
            return;
        };
        let segment = &self.transcripts[ix];
        let Some(start) = segment.audio_start_time.filter(|s| s.is_finite()) else {
            return;
        };
        let recording_end = self
            .transcripts
            .iter()
            .filter_map(|t| t.audio_end_time.or(t.audio_start_time))
            .filter(|s| s.is_finite())
            .fold(start, f64::max);
        let end = (start + JUMP_PLAY_SECS).min(recording_end);
        if end - start < 0.5 || self.meeting_folder_path.is_none() {
            return;
        }
        self.play_segment(segment.id.clone(), start, end, None, cx);
    }

    /// Show the transcript at `seconds` — now if it's loaded, otherwise as
    /// soon as it is. Call after [`Self::load`].
    pub fn jump_when_loaded(&mut self, seconds: f64, cx: &mut Context<Self>) {
        if self.transcripts_loading || self.transcripts.is_empty() {
            self.pending_jump = Some(seconds);
        } else {
            self.jump_to_time(seconds, cx);
        }
    }

    fn set_right_tab(&mut self, tab: RightTab, cx: &mut Context<Self>) {
        if self.right_tab != tab {
            self.right_tab = tab;
            if tab == RightTab::Insights {
                self.insights_panel.update(cx, |panel, cx| panel.refresh(cx));
            }
            cx.notify();
        }
    }

    /// The stored transcript or its speaker attribution changed: keep the
    /// Insights numbers in step.
    fn transcript_changed(&mut self, cx: &mut Context<Self>) {
        let visible = self.right_tab == RightTab::Insights;
        self.insights_panel.update(cx, |panel, cx| panel.invalidate(visible, cx));
    }

    fn apply_meeting_details(&mut self, details: MeetingDetails, cx: &mut Context<Self>) {
        self.title = details.title;
        self.created_at = chrono::DateTime::parse_from_rfc3339(&details.created_at)
            .ok()
            .map(|dt| dt.with_timezone(&chrono::Utc));
        self.transcripts = details.transcripts;
        self.refresh_speaker_filter();
        if let Some(seconds) = self.pending_jump.take() {
            self.jump_to_time(seconds, cx);
        }
    }

    fn refresh_summary(&mut self, generation: u64, cx: &mut Context<Self>) {
        let Some(pool) = AppServices::global(cx).pool() else {
            return;
        };
        let Some(id) = self.meeting_id.clone() else {
            return;
        };
        let io = Io::global(cx);
        cx.spawn(async move |this, cx| {
            let result = io
                .spawn(async move { SummaryProcessesRepository::get_summary_data_for_meeting(&pool, &id).await })
                .await;
            let _ = this.update(cx, |this, cx| {
                if this.generation != generation {
                    return;
                }
                match result {
                    Ok(Ok(Some(process))) => {
                        let markdown = process
                            .result
                            .as_deref()
                            .and_then(|s| serde_json::from_str::<serde_json::Value>(s).ok())
                            .and_then(|v| v.get("markdown").and_then(|m| m.as_str()).map(str::to_string));
                        let in_progress =
                            matches!(process.status.to_lowercase().as_str(), "processing" | "pending" | "summarizing");
                        // While a (re)generation runs, the row still holds the
                        // previous summary (kept so a failed run doesn't lose
                        // it). Don't paint it back over the new one streaming
                        // in — the pane shows only the live preview until the
                        // run finishes.
                        if let Some(markdown) = markdown.filter(|_| !in_progress) {
                            this.summary_has_content = !markdown.is_empty();
                            this.summary_markdown = markdown.clone();
                            this.summary_reveal_pending.clear();
                            this.summary_state.update(cx, |state, cx| state.set_text(&markdown, cx));
                        }
                        match process.status.to_lowercase().as_str() {
                            _ if in_progress => {
                                if !matches!(this.summary_phase, SummaryPhase::Generating) {
                                    this.summary_phase = SummaryPhase::Generating;
                                    this.start_poll(generation, cx);
                                }
                            }
                            "failed" | "error" => {
                                this.summary_phase =
                                    SummaryPhase::Error(process.error.unwrap_or_else(|| "Summary generation failed".into()));
                            }
                            _ => this.summary_phase = SummaryPhase::Idle,
                        }
                    }
                    Ok(Ok(None)) => this.summary_phase = SummaryPhase::Idle,
                    Ok(Err(e)) => this.summary_phase = SummaryPhase::Error(format!("Failed to load summary: {e}")),
                    Err(e) => this.summary_phase = SummaryPhase::Error(format!("Failed to load summary: {e}")),
                }
                cx.notify();
            });
        })
        .detach();
    }

    fn save_title(&mut self, cx: &mut Context<Self>) {
        let Some(id) = self.meeting_id.clone() else {
            return;
        };
        let Some(pool) = AppServices::global(cx).pool() else {
            return;
        };
        let new_title = self.title_input.read(cx).value().to_string();
        if new_title.trim().is_empty() {
            return;
        }
        self.title = new_title.clone();
        self.editing_title = false;
        cx.notify();

        let io = Io::global(cx);
        cx.spawn(async move |_this, cx| {
            let _ = io
                .spawn(async move { MeetingsRepository::update_meeting_title(&pool, &id, &new_title).await })
                .await;
            let _ = cx.update(|cx| shell::refresh_meetings(cx));
        })
        .detach();
    }

    fn confirm_delete(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let this = cx.entity();
        window.open_alert_dialog(cx, move |alert, _, _| {
            let this = this.clone();
            alert
                .title("Delete meeting")
                .description("This permanently deletes the meeting, its transcript, and its summary. This cannot be undone.")
                .show_cancel(true)
                .on_ok(move |_, _, cx| {
                    this.update(cx, |this, cx| this.delete(cx));
                    true
                })
        });
    }

    fn delete(&mut self, cx: &mut Context<Self>) {
        let Some(id) = self.meeting_id.clone() else {
            return;
        };
        let Some(pool) = AppServices::global(cx).pool() else {
            return;
        };
        let io = Io::global(cx);
        cx.spawn(async move |_this, cx| {
            let result = io
                .spawn(async move { MeetingsRepository::delete_meeting(&pool, &id).await })
                .await;
            let _ = cx.update(|cx| {
                match result {
                    Ok(Ok(true)) => {
                        shell::refresh_meetings(cx);
                        shell::navigate(Route::Recording, cx);
                    }
                    Ok(Ok(false)) => log::warn!("Meeting already deleted"),
                    Ok(Err(e)) => log::error!("Failed to delete meeting: {e}"),
                    Err(e) => log::error!("Failed to delete meeting: {e}"),
                }
            });
        })
        .detach();
    }

    fn cycle_template(&mut self, cx: &mut Context<Self>) {
        if self.templates.is_empty() {
            return;
        }
        let ids: Vec<String> = self.templates.iter().map(|(id, _, _)| id.clone()).collect();
        let current = ids.iter().position(|id| *id == self.selected_template).unwrap_or(0);
        let next = (current + 1) % ids.len();
        self.selected_template = ids[next].clone();
        self.template_user_selected = true;
        cx.notify();
    }

    fn toggle_custom_prompt(&mut self, cx: &mut Context<Self>) {
        self.show_custom_prompt = !self.show_custom_prompt;
        cx.notify();
    }

    fn generate_summary(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let Some(id) = self.meeting_id.clone() else {
            return;
        };
        let Some(pool) = AppServices::global(cx).pool() else {
            return;
        };
        if self.transcripts.is_empty() {
            window.push_notification(Notification::error("No transcript available for this meeting."), cx);
            return;
        }
        let text = format::build_transcript_text(&self.transcripts);
        let sink = AppServices::global(cx).sink.clone();
        let io = Io::global(cx);
        let generation = self.generation;
        let template_id = self.selected_template.clone();
        let custom_prompt = self.custom_prompt.read(cx).value().to_string();

        self.summary_phase = SummaryPhase::Generating;
        self.summary_has_content = false;
        self.summary_markdown = String::new();
        self.summary_reveal_pending.clear();
        self.summary_state.update(cx, |state, cx| state.set_text("", cx));
        cx.notify();

        let pool_for_config = pool.clone();
        cx.spawn(async move |this, cx| {
            let config = io
                .spawn(async move { SettingsRepository::get_model_config(&pool_for_config).await })
                .await;
            let Ok(Ok(Some(config))) = config else {
                let _ = this.update(cx, |this, cx| {
                    if this.generation != generation {
                        return;
                    }
                    this.summary_phase =
                        SummaryPhase::Error("No model configured — set one up in Settings first.".into());
                    cx.notify();
                });
                return;
            };

            let _ = io.spawn(SummaryService::process_transcript_background(
                sink,
                pool,
                id,
                text,
                config.provider,
                config.model,
                custom_prompt,
                template_id,
            ));

            let _ = this.update(cx, |this, cx| {
                if this.generation != generation {
                    return;
                }
                this.start_poll(generation, cx);
            });
        })
        .detach();
    }

    fn start_poll(&mut self, generation: u64, cx: &mut Context<Self>) {
        let task = cx.spawn(async move |this, cx| {
            loop {
                cx.background_executor().timer(Duration::from_millis(1500)).await;
                let still_relevant = this
                    .update(cx, |this, cx| {
                        if this.generation != generation {
                            return false;
                        }
                        this.refresh_summary(generation, cx);
                        !matches!(this.summary_phase, SummaryPhase::Generating)
                    })
                    .unwrap_or(true);
                if still_relevant {
                    break;
                }
            }
        });
        self._poll_task = Some(task);
    }

    /// Buffers the incoming delta instead of pushing it straight to
    /// `summary_state` — [`Self::start_summary_reveal_task`] drains the
    /// buffer at a typewriter pace (see `format::summary_reveal_chars_per_tick`).
    /// `summary_markdown` (the source-of-truth for save/copy/export) still
    /// gets the delta immediately: only the *rendered* `summary_state` text
    /// lags behind.
    fn on_summary_stream(&mut self, event: &crate::core_events::CoreEvent, cx: &mut Context<Self>) {
        #[derive(serde::Deserialize)]
        struct Delta {
            meeting_id: String,
            delta: String,
            /// Discard the preview streamed so far: the core is about to
            /// re-stream (a structured-summary retry or the Markdown
            /// fallback).
            #[serde(default)]
            reset: bool,
        }
        let Some(payload) = event.decode::<Delta>() else {
            return;
        };
        if self.meeting_id.as_deref() != Some(payload.meeting_id.as_str()) {
            return;
        }
        if payload.reset {
            self.summary_has_content = false;
            self.summary_markdown.clear();
            self.summary_reveal_pending.clear();
            self.summary_state.update(cx, |state, cx| state.set_text("", cx));
            cx.notify();
        }
        if payload.delta.is_empty() {
            return;
        }
        self.summary_has_content = true;
        self.summary_markdown.push_str(&payload.delta);

        if cx.reduce_motion() {
            // Mirrors `useTranscriptStreaming.ts`'s reduced-motion path:
            // skip the reveal animation and show text immediately.
            self.summary_state.update(cx, |state, cx| state.push_str(&payload.delta, cx));
            return;
        }

        self.summary_reveal_pending.push_str(&payload.delta);
        if self._summary_reveal_task.is_none() {
            self.start_summary_reveal_task(cx);
        }
    }

    /// Drains `summary_reveal_pending` into `summary_state` a few characters
    /// at a time on a repeating timer, mirroring `useTranscriptStreaming.ts`'s
    /// pacing. Stops itself once the buffer empties (a later delta restarts
    /// it via `on_summary_stream`), so at most one timer runs at a time.
    fn start_summary_reveal_task(&mut self, cx: &mut Context<Self>) {
        let generation = self.generation;
        let task = cx.spawn(async move |this, cx| {
            loop {
                cx.background_executor()
                    .timer(Duration::from_millis(format::SUMMARY_REVEAL_INTERVAL_MS))
                    .await;
                let done = this
                    .update(cx, |this, cx| {
                        if this.generation != generation || this.summary_reveal_pending.is_empty() {
                            return true;
                        }
                        let n = format::summary_reveal_chars_per_tick(this.summary_reveal_pending.len())
                            .min(this.summary_reveal_pending.len());
                        // Drain on a char boundary — `n` counts bytes of a
                        // UTF-8 buffer, which may land mid-codepoint.
                        let mut boundary = n;
                        while boundary < this.summary_reveal_pending.len()
                            && !this.summary_reveal_pending.is_char_boundary(boundary)
                        {
                            boundary += 1;
                        }
                        let chunk: String = this.summary_reveal_pending.drain(..boundary).collect();
                        this.summary_state.update(cx, |state, cx| state.push_str(&chunk, cx));
                        this.summary_reveal_pending.is_empty()
                    })
                    .unwrap_or(true);
                if done {
                    break;
                }
            }
            let _ = this.update(cx, |this, cx| {
                if this.generation == generation {
                    this._summary_reveal_task = None;
                    cx.notify();
                }
            });
        });
        self._summary_reveal_task = Some(task);
    }

    // ---- Summary editing (zorite) ----------------------------------------

    fn start_edit_summary(&mut self, cx: &mut Context<Self>) {
        let markdown = self.summary_markdown.clone();
        self.summary_editor.update(cx, |editor, cx| editor.set_text(markdown, cx));
        self.editing_summary = true;
        cx.notify();
    }

    fn cancel_edit_summary(&mut self, cx: &mut Context<Self>) {
        self.editing_summary = false;
        cx.notify();
    }

    /// Whether the summary editor is open with edits that differ from the
    /// saved markdown — the trigger for the navigation-away guard (see
    /// [`format::has_unsaved_summary_edits`] and `shell::AppShell::guard_navigate`).
    pub fn has_unsaved_summary_edits(&self, cx: &App) -> bool {
        format::has_unsaved_summary_edits(
            self.editing_summary,
            &self.summary_editor.read(cx).text(),
            &self.summary_markdown,
        )
    }

    /// Discard the in-progress summary edit without saving. Used by the
    /// "Discard" choice of the unsaved-changes dialog.
    pub fn discard_summary_edits(&mut self, cx: &mut Context<Self>) {
        self.cancel_edit_summary(cx);
    }

    /// Save the in-progress summary edit. Used by the "Save" choice of the
    /// unsaved-changes dialog — fire-and-forget, like the regular Save
    /// button: the save task keeps running (and updates this entity) even
    /// after the shell has already navigated away, guarded by `generation`.
    pub fn save_summary_and_leave(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        self.save_summary(window, cx);
    }

    fn save_summary(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let Some(id) = self.meeting_id.clone() else {
            return;
        };
        let Some(pool) = AppServices::global(cx).pool() else {
            return;
        };
        let markdown = self.summary_editor.read(cx).text().to_string();
        self.saving_summary = true;
        cx.notify();

        let io = Io::global(cx);
        let generation = self.generation;
        cx.spawn(async move |this, cx| {
            let value = serde_json::json!({ "markdown": markdown.clone() });
            let pool_for_md = pool.clone();
            let id_for_md = id.clone();
            let value_for_md = value.clone();
            let saved = io
                .spawn(async move { SummaryProcessesRepository::update_meeting_summary(&pool, &id, &value).await })
                .await;
            // Best-effort `summary.md` sidecar write — failures here don't
            // fail the save, the DB is the source of truth.
            let _ = io
                .spawn(async move {
                    let _ = markdown_export::write_summary_md(&pool_for_md, &id_for_md, &value_for_md).await;
                    // Keep the export-folder note (if enabled) in step with edits.
                    parley_core::export::folder_sync::sync_after_summary(&pool_for_md, &id_for_md).await;
                })
                .await;

            let _ = this.update(cx, |this, cx| {
                this.saving_summary = false;
                if this.generation != generation {
                    return;
                }
                match saved {
                    Ok(Ok(true)) => {
                        this.summary_markdown = markdown.clone();
                        this.summary_has_content = !markdown.is_empty();
                        this.summary_state.update(cx, |state, cx| state.set_text(&markdown, cx));
                        this.editing_summary = false;
                    }
                    _ => {
                        log::error!("meeting: failed to save edited summary");
                    }
                }
                cx.notify();
            });
        })
        .detach();

        window.push_notification(Notification::info("Saving summary…"), cx);
    }

    // ---- Copy / export -----------------------------------------------------

    fn copy_transcript(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        if self.transcripts.is_empty() {
            window.push_notification(Notification::error("No transcripts available to copy"), cx);
            return;
        }
        let Some(id) = self.meeting_id.clone() else {
            return;
        };
        let text = format::copy_transcript_text(&id, &self.title, self.created_at, &self.transcripts);
        cx.write_to_clipboard(ClipboardItem::new_string(text));
        window.push_notification(Notification::success("Transcript copied to clipboard"), cx);
    }

    fn copy_summary(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        if self.summary_markdown.trim().is_empty() {
            window.push_notification(Notification::error("No summary content available to copy"), cx);
            return;
        }
        let Some(id) = self.meeting_id.clone() else {
            return;
        };
        let text = format::copy_summary_text(&id, &self.title, self.created_at, &self.summary_markdown);
        cx.write_to_clipboard(ClipboardItem::new_string(text));
        window.push_notification(Notification::success("Summary copied to clipboard"), cx);
    }

    /// `format` is `"markdown"` or `"json"`, matching `export_meeting`'s
    /// core function.
    fn export_copy(&mut self, format: &'static str, cx: &mut Context<Self>) {
        let Some(id) = self.meeting_id.clone() else {
            return;
        };
        let Some(pool) = AppServices::global(cx).pool() else {
            return;
        };
        let io = Io::global(cx);
        cx.spawn(async move |_this, cx| {
            let result = io
                .spawn(async move { parley_core::export::build_export(&pool, &id, format).await })
                .await;
            let _ = cx.update(|cx| match result {
                Ok(Ok(export)) => {
                    cx.write_to_clipboard(ClipboardItem::new_string(export.content));
                    let label = if format == "json" { "JSON" } else { "Markdown" };
                    notify(cx, Notification::success(format!("Meeting copied as {label}")));
                }
                _ => notify(cx, Notification::error("Failed to export meeting")),
            });
        })
        .detach();
    }

    /// Export in any format (Markdown, JSON, SRT, VTT, DOCX, PDF), then a
    /// native save-file dialog (GPUI's `prompt_for_new_path`).
    fn export_save(&mut self, format: parley_core::export::ExportFormat, cx: &mut Context<Self>) {
        let Some(id) = self.meeting_id.clone() else {
            return;
        };
        let Some(pool) = AppServices::global(cx).pool() else {
            return;
        };
        let io = Io::global(cx);
        cx.spawn(async move |_this, cx| {
            let result = io
                .spawn(async move { parley_core::export::build_export_file(&pool, &id, format).await })
                .await;
            let export = match result {
                Ok(Ok(export)) => export,
                Ok(Err(e)) => {
                    log::warn!("meeting: {} export failed: {e}", format.name());
                    // Core errors here are written for people ("This
                    // transcript has no timestamps…"), so show them.
                    let _ = cx.update(|cx| notify(cx, Notification::error(format!("Export failed: {e}"))));
                    return;
                }
                Err(_) => {
                    let _ = cx.update(|cx| notify(cx, Notification::error("Failed to export meeting")));
                    return;
                }
            };

            let home = std::env::var("HOME").map(std::path::PathBuf::from).unwrap_or_else(|_| std::path::PathBuf::from("/"));
            let rx = cx.update(|cx| cx.prompt_for_new_path(&home, Some(&export.filename)));
            match rx.await {
                Ok(Ok(Some(path))) => {
                    let content = export.bytes;
                    let write_result = io.spawn(async move { std::fs::write(&path, content) }).await;
                    let _ = cx.update(|cx| match write_result {
                        Ok(Ok(())) => notify(cx, Notification::success("Meeting exported")),
                        _ => notify(cx, Notification::error("Failed to write export")),
                    });
                }
                Ok(Ok(None)) => {
                    // Cancelled the save dialog — not an error, no toast.
                }
                _ => {
                    let _ = cx.update(|cx| notify(cx, Notification::error("Failed to export meeting")));
                }
            }
        })
        .detach();
    }

    /// The header's "Export…" button: one menu entry per save format.
    fn render_export_menu(&self, cx: &mut Context<Self>) -> impl IntoElement {
        use gpui_kit::component::menu::{DropdownMenu as _, PopupMenuItem};
        use parley_core::export::ExportFormat;

        let view = cx.entity().downgrade();
        Button::new("export-meeting")
            .ghost()
            .icon(Lucide::Download)
            .tooltip("Export meeting…")
            .dropdown_menu(move |menu, _window, _cx| {
                let mut menu = menu;
                for (i, format) in [
                    ExportFormat::Markdown,
                    ExportFormat::Json,
                    ExportFormat::Docx,
                    ExportFormat::Pdf,
                    ExportFormat::Srt,
                    ExportFormat::Vtt,
                ]
                .into_iter()
                .enumerate()
                {
                    // Separate the documents from the transcript-only subtitles.
                    if i == 4 {
                        menu = menu.separator();
                    }
                    let view = view.clone();
                    menu = menu.item(PopupMenuItem::new(format!("{}…", format.label())).on_click(
                        move |_, _, cx| {
                            let _ = view.update(cx, |this, cx| this.export_save(format, cx));
                        },
                    ));
                }
                menu
            })
    }

    // ---- Retranscription -----------------------------------------------------

    fn open_retranscribe(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        if self.meeting_folder_path.is_none() {
            window.push_notification(Notification::error("This meeting has no saved audio to retranscribe."), cx);
            return;
        }
        self.processing_operation = "transcribe";
        self.show_processing_options(cx);
    }

    fn show_processing_options(&mut self, cx: &mut Context<Self>) {
        self.retranscribe_open = true;
        self.identify_open = false;
        self.history_open = false;
        self.identity_request += 1;
        self.retranscribe_error = None;
        let speaker_only = self.processing_operation == "speakers";
        self.pipeline_editor.update(cx, |editor, cx| {
            editor.set_speaker_analysis_only(speaker_only, cx);
            editor.reload(cx);
        });
        cx.notify();
    }

    fn close_retranscribe(&mut self, cx: &mut Context<Self>) {
        self.retranscribe_open = false;
        cx.notify();
    }

    fn start_retranscribe(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        self.start_pipeline_job(self.processing_operation, window, cx);
    }

    fn start_rediarize(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        self.processing_operation = "speakers";
        self.show_processing_options(cx);
        window.push_notification(
            Notification::info("Choose the speaker pipeline, then start re-detection."),
            cx,
        );
    }

    fn start_pipeline_job(
        &mut self,
        operation: &'static str,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let Some(id) = self.meeting_id.clone() else {
            return;
        };
        let Some(pool) = AppServices::global(cx).pool() else {
            return;
        };
        if self.retranscribe_in_progress || self.identify_busy {
            return;
        }
        let config = match (if operation == "identify" {
            &self.identify_editor
        } else {
            &self.pipeline_editor
        })
        .read(cx)
        .snapshot()
        {
            Ok(config) => config,
            Err(error) => {
                window.push_notification(Notification::error(error), cx);
                return;
            }
        };
        let num_speakers = if operation == "identify" {
            0
        } else {
            match format::parse_speaker_count(self.retranscribe_speakers.read(cx).value().as_ref())
            {
                Ok(count) => count,
                Err(error) => {
                    window.push_notification(Notification::error(error), cx);
                    return;
                }
            }
        };
        let language = if operation == "speakers" {
            None
        } else {
            match self.retranscribe_language.trim() {
                "" | "auto" => None,
                value => Some(value.to_string()),
            }
        };
        self.processing_cancel = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let cancelled = self.processing_cancel.clone();
        self.identity_request += 1;
        self.identify_busy = operation == "identify";
        self.retranscribe_in_progress = operation != "identify";
        self.rediarize_in_progress = operation == "speakers";
        self.retranscribe_error = None;
        self.retranscribe_progress_pct = 0;
        self.retranscribe_progress_message = "Starting…".into();
        let job_progress = self.job_progress.update(cx, |progress, cx| progress.start(cx));
        let review_replacements = self.review_identity_replacements;
        let first_transcript = self.transcripts.is_empty();
        let generation = self.generation;
        let sink = AppServices::global(cx).sink.clone();
        let io = Io::global(cx);
        cx.notify();
        cx.spawn(async move |this, cx| {
            let result = io.spawn(async move {
                use parley_core::audio::processing::service::{self, ProcessingOptions, IdentificationOptions};
                let progress = |stage: &str, percent: u32, message: &str| {
                    job_progress(stage, percent, message);
                    let _ = sink.emit_value("retranscription-progress", serde_json::json!({"meeting_id":id,"stage":stage,"progress_percentage":percent,"message":message}));
                };
                if operation == "identify" {
                    let outcome=service::identify_speakers(&pool, &id, IdentificationOptions { identification: config.identification, target_id: config.target_id, replace_confirmed: review_replacements }, &cancelled, progress).await?;
                    let version=parley_core::audio::processing::runs::current_run(&pool,&id).await?;
                    Ok((None,outcome.suggestions,version))
                } else {
                    let options = ProcessingOptions { config, language, num_speakers };
                    let result = if operation == "speakers" { service::redetect_speakers(&pool, &id, options, &cancelled, progress).await } else { service::process_recording(&pool, &id, options, &cancelled, progress).await };
                    if result.is_ok() && first_transcript {
                        parley_core::summary::auto_summary::maybe_auto_summarize(sink.clone(), Some(pool.clone()), id.clone()).await;
                        parley_core::chapters::maybe_auto_chapter(sink, Some(pool), id).await;
                    }
                    result.map(|outcome| (outcome.identification_error,Vec::new(),None))
                }
            }).await;
            let _ = this.update(cx, |this, cx| {
                if this.generation != generation { return; }
                this.job_progress.update(cx, |progress, cx| progress.finish(cx));
                this.retranscribe_in_progress = false;
                this.rediarize_in_progress = false;
                this.identify_busy = false;
                this.processing_history.update(cx, |history,cx|history.refresh(cx));
                match result {
                    Ok(Ok((warning,suggestions,version))) => {
                        if let Some(warning)=warning { notify(cx,Notification::warning(format!("Transcript saved. Speaker identification needs attention: {warning}"))); }
                        this.retranscribe_open = false;
                        this.identity_suggestions=suggestions;
                        this.identity_group_version=version;
                        this.identify_open=operation=="identify";
                        this.reload_transcript(cx);
                        if operation=="transcribe" {this.load_identity_suggestions_for_review(true,cx);}
                        shell::refresh_meetings(cx);
                        notify(cx, Notification::success(if operation == "identify" {"Speaker identification finished"} else {"Processing finished"}));
                    }
                    Ok(Err(error)) => this.retranscribe_error = Some(error.to_string()),
                    Err(error) => this.retranscribe_error = Some(error.to_string()),
                }
                cx.notify();
            });
        }).detach();
    }

    fn cancel_retranscribe(&mut self, _window: &mut Window, _cx: &mut Context<Self>) {
        self.processing_cancel
            .store(true, std::sync::atomic::Ordering::SeqCst);
    }

    fn on_retranscription_progress(&mut self, event: &crate::core_events::CoreEvent, cx: &mut Context<Self>) {
        let Some(payload) = event.decode::<parley_core::audio::retranscription::RetranscriptionProgress>() else {
            return;
        };
        if self.meeting_id.as_deref() != Some(payload.meeting_id.as_str()) {
            return;
        }
        self.retranscribe_progress_pct = payload.progress_percentage;
        self.retranscribe_progress_message = payload.message;
        cx.notify();
    }

    fn on_retranscription_complete(&mut self, event: &crate::core_events::CoreEvent, cx: &mut Context<Self>) {
        let Some(payload) = event.decode::<parley_core::audio::retranscription::RetranscriptionResult>() else {
            return;
        };
        if self.meeting_id.as_deref() != Some(payload.meeting_id.as_str()) {
            return;
        }
        self.retranscribe_in_progress = false;
        self.retranscribe_open = false;
        cx.notify();
        notify(cx, Notification::success(format!("Retranscription complete: {} segments", payload.segments_count)));
        self.reload_transcript(cx);
        self.load_identity_suggestions_for_review(true, cx);
        shell::refresh_meetings(cx);
    }

    fn on_retranscription_error(&mut self, event: &crate::core_events::CoreEvent, cx: &mut Context<Self>) {
        let Some(payload) = event.decode::<parley_core::audio::retranscription::RetranscriptionError>() else {
            return;
        };
        if self.meeting_id.as_deref() != Some(payload.meeting_id.as_str()) {
            return;
        }
        self.retranscribe_in_progress = false;
        self.retranscribe_error = Some(payload.error);
        cx.notify();
    }

    /// Re-fetch just the transcript after a retranscription completes.
    fn reload_transcript(&mut self, cx: &mut Context<Self>) {
        let Some(id) = self.meeting_id.clone() else {
            return;
        };
        let Some(pool) = AppServices::global(cx).pool() else {
            return;
        };
        let io = Io::global(cx);
        let generation = self.generation;
        cx.spawn(async move |this, cx| {
            let result = io.spawn(async move { MeetingsRepository::get_meeting(&pool, &id).await }).await;
            let _ = this.update(cx, |this, cx| {
                if this.generation != generation {
                    return;
                }
                if let Ok(Ok(Some(details))) = result {
                    this.transcripts = details.transcripts;
                    this.refresh_speaker_filter();
                    this.transcript_changed(cx);
                }
                cx.notify();
            });
        })
        .detach();
    }

    // ---- Calendar event link -------------------------------------------
    // Show the linked calendar event and let the user pick or clear one,
    // via `CalendarRepository` and `calendar::service::link_meeting_with_snapshot`.

    fn load_calendar_event(&mut self, generation: u64, cx: &mut Context<Self>) {
        let Some(pool) = AppServices::global(cx).pool() else {
            self.calendar_loading = false;
            return;
        };
        let Some(id) = self.meeting_id.clone() else {
            self.calendar_loading = false;
            return;
        };
        let io = Io::global(cx);
        cx.spawn(async move |this, cx| {
            let result = io
                .spawn(async move { CalendarRepository::get_event_for_meeting(&pool, &id).await })
                .await;
            let _ = this.update(cx, |this, cx| {
                if this.generation != generation {
                    return;
                }
                this.calendar_loading = false;
                match result {
                    Ok(Ok(event)) => this.calendar_event = event,
                    Ok(Err(e)) => log::warn!("meeting: get_event_for_meeting failed: {e}"),
                    Err(e) => log::warn!("meeting: get_event_for_meeting task panicked: {e}"),
                }
                cx.notify();
            });
        })
        .detach();
    }

    /// Open the "link a calendar event" picker, fetching events within
    /// `±CALENDAR_WINDOW_DAYS` of the meeting's `created_at` (falling back
    /// to "now" if unknown) — mirrors the picker's default anchor/window.
    fn open_calendar_picker(&mut self, cx: &mut Context<Self>) {
        self.calendar_picker_open = true;
        self.calendar_picker_events = None;
        self.calendar_picker_error = None;
        cx.notify();

        let Some(pool) = AppServices::global(cx).pool() else {
            self.calendar_picker_error = Some("No database — complete onboarding first.".into());
            return;
        };
        let anchor = self.created_at.unwrap_or_else(chrono::Utc::now);
        let window = chrono::Duration::days(CALENDAR_WINDOW_DAYS);
        let from = anchor - window;
        let to = anchor + window;
        let io = Io::global(cx);
        let generation = self.generation;
        cx.spawn(async move |this, cx| {
            let result = io
                .spawn(async move { CalendarRepository::list_events_in_range(&pool, from, to).await })
                .await;
            let _ = this.update(cx, |this, cx| {
                if this.generation != generation || !this.calendar_picker_open {
                    return;
                }
                match result {
                    Ok(Ok(mut events)) => {
                        calendar_format::sort_by_proximity(&mut events, anchor, |e| e.start_at);
                        this.calendar_picker_events = Some(events);
                    }
                    Ok(Err(e)) => this.calendar_picker_error = Some(e.to_string()),
                    Err(e) => this.calendar_picker_error = Some(format!("Task panicked: {e}")),
                }
                cx.notify();
            });
        })
        .detach();
    }

    fn close_calendar_picker(&mut self, cx: &mut Context<Self>) {
        self.calendar_picker_open = false;
        cx.notify();
    }

    /// Link (or, with `event_id: None`, unlink) the meeting to a calendar
    /// event — mirrors `linkMeetingToCalendarEvent`.
    fn pick_calendar_event(&mut self, event_id: Option<String>, cx: &mut Context<Self>) {
        let Some(id) = self.meeting_id.clone() else {
            return;
        };
        let Some(pool) = AppServices::global(cx).pool() else {
            return;
        };
        self.calendar_mutating = true;
        cx.notify();

        let io = Io::global(cx);
        let generation = self.generation;
        cx.spawn(async move |this, cx| {
            let event_id_for_task = event_id.clone();
            let result = io
                .spawn(async move {
                    link_meeting_with_snapshot(&pool, &id, event_id_for_task.as_deref()).await
                })
                .await;
            let _ = this.update(cx, |this, cx| {
                if this.generation != generation {
                    return;
                }
                this.calendar_mutating = false;
                match result {
                    Ok(Ok(true)) => {
                        this.calendar_picker_open = false;
                        this.load_calendar_event(generation, cx);
                    }
                    Ok(Ok(false)) => log::warn!("meeting: calendar link no-op (meeting not found)"),
                    Ok(Err(e)) => notify(cx, Notification::error(format!("Couldn't link that calendar event: {e}"))),
                    Err(e) => notify(cx, Notification::error(format!("Calendar link task panicked: {e}"))),
                }
                cx.notify();
            });
        })
        .detach();
    }

    fn unlink_calendar_event(&mut self, cx: &mut Context<Self>) {
        self.pick_calendar_event(None, cx);
    }

    // ---- Per-segment speaker chip edit -----------------------------------
    // Mirrors `EditableSpeakerChip.tsx`: rename a named voice profile across
    // meetings, or (for an unnamed "Speaker N" cluster) either promote it to
    // a brand-new profile scoped to this meeting's rename, or merge it into
    // an existing profile.

    /// Toggle the edit panel for one transcript segment's speaker chip.
    /// Clicking the already-open chip's own label closes it.
    fn toggle_speaker_edit(
        &mut self,
        segment_id: String,
        speaker: String,
        voice_profile_id: Option<String>,
        click_position: Point<Pixels>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let key = format!("{segment_id}:{speaker}");
        if self.speaker_edit.as_ref().is_some_and(|s| s.key == key) {
            self.speaker_edit = None;
            cx.notify();
            return;
        }

        let initial_name = speaker_chip::initial_edit_name(&speaker, voice_profile_id.as_deref());
        let name_input =
            cx.new(|cx| InputState::new(window, cx).placeholder("e.g. Alice Smith").default_value(initial_name));
        let email_input = cx.new(|cx| InputState::new(window, cx).placeholder("Email (optional)"));
        window.focus(&name_input.read(cx).focus_handle(cx), cx);
        let attendees =
            self.calendar_event.as_ref().map(|e| speaker_chip::attendee_suggestions(&e.attendees)).unwrap_or_default();
        let cluster_size = self.transcripts.iter().filter(|t| t.speaker.as_deref() == Some(speaker.as_str())).count();
        self.speaker_edit = Some(SpeakerEditState {
            segment_id: segment_id.clone(),
            cluster_size,
            single_segment: cluster_size <= 1 && voice_profile_id.is_some(),
            key: key.clone(),
            speaker: speaker.clone(),
            voice_profile_id: voice_profile_id.clone(),
            name_input,
            email_input,
            profiles: Vec::new(),
            profiles_loading: true,
            picker: None,
            picker_subscription: None,
            has_ranking: false,
            merge_target: None,
            saving: false,
            error: None,
            click_position,
            attendees,
        });
        cx.notify();

        let Some(pool) = AppServices::global(cx).pool() else {
            return;
        };
        let io = Io::global(cx);
        let generation = self.generation;
        let meeting = self.meeting_id.clone().unwrap_or_default();
        cx.spawn(async move |this, cx| {
            let result = io.spawn(async move {
                let profiles = VoiceProfilesRepository::list_all(&pool).await?;
                let scores = parley_core::audio::processing::identity_candidates::for_segment(&pool, &meeting, &segment_id).await;
                Ok::<_, anyhow::Error>((profiles, scores))
            }).await;
            let _ = this.update_in(cx, |this, window, cx| {
                if this.generation != generation {
                    return;
                }
                let Some(state) = this.speaker_edit.as_mut() else {
                    return;
                };
                if state.key != key {
                    return;
                }
                state.profiles_loading = false;
                if let Ok(Ok((mut profiles, scores))) = result {
                    let scores = match scores {
                        Ok(scores) => scores,
                        Err(error) => {
                            state.error = Some(format!("Could not load saved similarities: {error}. Profiles are listed by name."));
                            Vec::new()
                        }
                    };
                    state.has_ranking = !scores.is_empty();
                    let score = |id: &str| scores.iter().find(|c| c.profile_id == id).map(|c| c.score);
                    profiles.sort_by(|a, b| speaker_chip::compare_profile_candidates(
                        &a.name, score(&a.id), &b.name, score(&b.id)).then_with(|| a.id.cmp(&b.id)));
                    let items: Vec<_> = std::iter::once(SpeakerOption {
                        id: String::new(), label: "Enter or edit a name…".into(),
                    }).chain(profiles.iter().map(|p| SpeakerOption {
                        id: p.id.clone(),
                        label: match score(&p.id) {
                            Some(score) => format!("{} — similarity {:.2}", p.name, score).into(),
                            None => p.name.clone().into(),
                        },
                    })).collect();
                    let picker = cx.new(|cx| SelectState::new(SearchableVec::new(items), Some(IndexPath::default()), window, cx).searchable(true));
                    // Named-profile edit: prefill the email field from the
                    // profile's stored value (name is already prefilled from
                    // the displayed label).
                    if let Some(vp_id) = state.voice_profile_id.clone() {
                        if let Some(me) = profiles.iter().find(|p| p.id == vp_id) {
                            let email = me.email.clone().unwrap_or_default();
                            state.email_input.update(cx, |s, cx| s.set_value(email, window, cx));
                        }
                    }
                    state.profiles = profiles;
                    state.picker = Some(picker.clone());
                    let subscription = cx.subscribe(&picker, |this, _, event: &SelectEvent<SearchableVec<SpeakerOption>>, cx| {
                        let SelectEvent::Confirm(value) = event;
                        if let Some(state) = this.speaker_edit.as_mut() {
                            state.merge_target = value.clone().filter(|id| !id.is_empty());
                            cx.notify();
                        }
                    });
                    if let Some(state) = this.speaker_edit.as_mut() {
                        state.picker_subscription = Some(std::sync::Arc::new(subscription));
                    }
                } else {
                    state.error = Some("Could not load saved speakers. Close this panel and try again.".into());
                }
                cx.notify();
            });
        })
        .detach();
    }

    fn close_speaker_edit(&mut self, cx: &mut Context<Self>) {
        self.speaker_edit = None;
        cx.notify();
    }

    /// Fill the rename form from a clicked calendar-attendee suggestion —
    /// mirrors the attendee chip's `onClick` in `EditableSpeakerChip.tsx`.
    fn apply_attendee_suggestion(&mut self, label: String, email: Option<String>, window: &mut Window, cx: &mut Context<Self>) {
        let Some(state) = self.speaker_edit.as_ref() else {
            return;
        };
        let name_input = state.name_input.clone();
        let email_input = state.email_input.clone();
        name_input.update(cx, |s, cx| s.set_value(label, window, cx));
        email_input.update(cx, |s, cx| s.set_value(email.unwrap_or_default(), window, cx));
        cx.notify();
    }

    /// Switch the panel between relabelling the whole cluster and just the
    /// clicked segment. The name field restarts for the new scope.
    fn set_speaker_edit_scope(&mut self, single_segment: bool, window: &mut Window, cx: &mut Context<Self>) {
        let Some(state) = self.speaker_edit.as_mut() else {
            return;
        };
        if state.single_segment == single_segment {
            return;
        }
        state.single_segment = single_segment;
        state.merge_target = None;
        if let Some(picker) = &state.picker {
            picker.update(cx, |picker, cx| picker.set_selected_value(&String::new(), window, cx));
        }
        state.error = None;
        let name = if single_segment {
            String::new()
        } else {
            speaker_chip::initial_edit_name(&state.speaker, state.voice_profile_id.as_deref())
        };
        let name_input = state.name_input.clone();
        name_input.update(cx, |s, cx| {
            s.set_value(name, window, cx);
            s.focus(window, cx);
        });
        cx.notify();
    }

    fn save_speaker_edit(&mut self, cx: &mut Context<Self>) {
        let Some(state) = self.speaker_edit.as_ref() else {
            return;
        };
        let Some(pool) = AppServices::global(cx).pool() else {
            return;
        };
        let Some(meeting_id) = self.meeting_id.clone() else {
            return;
        };
        let speaker = state.speaker.clone();
        let voice_profile_id = state.voice_profile_id.clone();
        let merge_target = state.merge_target.clone();
        let name = state.name_input.read(cx).value().trim().to_string();
        let email = {
            let raw = state.email_input.read(cx).value().trim().to_string();
            if raw.is_empty() { None } else { Some(raw) }
        };
        if !speaker_chip::can_save_speaker_edit(merge_target.as_deref(), &name) {
            return;
        }
        let key = state.key.clone();
        let single_segment = state.single_segment.then(|| state.segment_id.clone());

        {
            let state = self.speaker_edit.as_mut().unwrap();
            state.saving = true;
            state.error = None;
        }
        cx.notify();

        let io = Io::global(cx);
        let generation = self.generation;
        cx.spawn(async move |this, cx| {
            let outcome: Result<(), String> = if let Some(segment_id) = single_segment {
                match io
                    .spawn(async move {
                        assign_segment_speaker_core(&pool, &meeting_id, &segment_id, &name, merge_target.as_deref()).await
                    })
                    .await
                {
                    Ok(Ok(_)) => Ok(()),
                    Ok(Err(e)) => Err(e),
                    Err(e) => Err(format!("Assign task panicked: {e}")),
                }
            } else if let Some(profile_id) = merge_target {
                match io
                    .spawn(async move { merge_cluster_into_profile_core(&pool, &meeting_id, &speaker, &profile_id).await })
                    .await
                {
                    Ok(Ok(_)) => Ok(()),
                    Ok(Err(e)) => Err(e),
                    Err(e) => Err(format!("Merge task panicked: {e}")),
                }
            } else if let Some(vp_id) = voice_profile_id {
                match io
                    .spawn(async move { VoiceProfilesRepository::update_profile(&pool, &vp_id, &name, email.as_deref()).await })
                    .await
                {
                    Ok(Ok(_)) => Ok(()),
                    Ok(Err(e)) => Err(e.to_string()),
                    Err(e) => Err(format!("Update task panicked: {e}")),
                }
            } else {
                match io
                    .spawn(async move {
                        promote_speaker_to_profile_core(&pool, &speaker, &name, email.as_deref(), &meeting_id).await
                    })
                    .await
                {
                    Ok(Ok(_)) => Ok(()),
                    Ok(Err(e)) => Err(e),
                    Err(e) => Err(format!("Promote task panicked: {e}")),
                }
            };

            let _ = this.update(cx, |this, cx| {
                if this.generation != generation {
                    return;
                }
                let still_open = this.speaker_edit.as_ref().is_some_and(|s| s.key == key);
                match outcome {
                    Ok(()) => {
                        this.speaker_edit = None;
                        this.reload_transcript(cx);
                        this.load_identity_suggestions(cx);
                    }
                    Err(e) if still_open => {
                        let state = this.speaker_edit.as_mut().unwrap();
                        state.saving = false;
                        state.error = Some(e);
                    }
                    Err(_) => {}
                }
                cx.notify();
            });
        })
        .detach();
    }

    /// Undo a wrong name: the panel's lines (all of them, or just the
    /// clicked one in single-line mode) go back to a fresh "Speaker N".
    fn unassign_speaker_edit(&mut self, cx: &mut Context<Self>) {
        let Some(state) = self.speaker_edit.as_mut() else {
            return;
        };
        let Some(pool) = AppServices::global(cx).pool() else {
            return;
        };
        let Some(meeting_id) = self.meeting_id.clone() else {
            return;
        };
        let speaker = state.speaker.clone();
        let segment_id = state.single_segment.then(|| state.segment_id.clone());
        let key = state.key.clone();
        state.saving = true;
        state.error = None;
        cx.notify();

        let io = Io::global(cx);
        let generation = self.generation;
        cx.spawn(async move |this, cx| {
            let outcome = match io
                .spawn(async move { unassign_speaker_core(&pool, &meeting_id, &speaker, segment_id.as_deref()).await })
                .await
            {
                Ok(Ok((_, profile_id))) => Ok(profile_id),
                Ok(Err(e)) => Err(e),
                Err(e) => Err(format!("Unassign task panicked: {e}")),
            };
            let _ = this.update(cx, |this, cx| {
                if this.generation != generation {
                    return;
                }
                let still_open = this.speaker_edit.as_ref().is_some_and(|s| s.key == key);
                match outcome {
                    Ok(_) => {
                        this.speaker_edit = None;
                        this.reload_transcript(cx);
                        this.load_identity_suggestions(cx);
                    }
                    Err(e) if still_open => {
                        let state = this.speaker_edit.as_mut().unwrap();
                        state.saving = false;
                        state.error = Some(e);
                    }
                    Err(_) => {}
                }
                cx.notify();
            });
        })
        .detach();
    }

    // ---- Per-segment audio playback --------------------------------------
    // Only one clip plays at a time, played natively through
    // `parley_core::audio::playback`.

    fn toggle_segment_playback(
        &mut self,
        segment_id: String,
        start_secs: f64,
        end_secs: f64,
        source: Option<String>,
        cx: &mut Context<Self>,
    ) {
        if self.playing_segment.as_deref() == Some(segment_id.as_str())
            || self.loading_segment.as_deref() == Some(segment_id.as_str())
        {
            self.stop_segment_playback(cx);
            return;
        }
        self.play_segment(segment_id, start_secs, end_secs, source, cx);
    }

    fn play_segment(
        &mut self,
        segment_id: String,
        start_secs: f64,
        end_secs: f64,
        source: Option<String>,
        cx: &mut Context<Self>,
    ) {
        let Some(meeting_id) = self.meeting_id.clone() else {
            return;
        };
        let Some(pool) = AppServices::global(cx).pool() else {
            return;
        };
        self.playing_segment = None;
        self.loading_segment = Some(segment_id.clone());
        cx.notify();

        let sink = AppServices::global(cx).sink.clone();
        let io = Io::global(cx);
        let generation = self.generation;
        let segment_for_task = segment_id.clone();
        cx.spawn(async move |this, cx| {
            let result: Result<(), String> = match io
                .spawn(async move {
                    let bytes = parley_core::audio::clip::extract_clip_wav(
                        &pool,
                        &meeting_id,
                        start_secs,
                        end_secs,
                        source.as_deref(),
                    )
                    .await?;
                    let (samples, sample_rate, channels) = parley_core::audio::clip::parse_wav_pcm16(&bytes)?;
                    parley_core::audio::playback::play_pcm_i16(&sink, samples, sample_rate, channels)
                })
                .await
            {
                Ok(r) => r,
                Err(e) => Err(format!("Clip playback task panicked: {e}")),
            };

            let _ = this.update(cx, |this, cx| {
                if this.generation != generation || this.loading_segment.as_deref() != Some(segment_for_task.as_str())
                {
                    return;
                }
                this.loading_segment = None;
                match result {
                    Ok(()) => this.playing_segment = Some(segment_for_task),
                    Err(e) => notify(cx, Notification::error(e)),
                }
                cx.notify();
            });
        })
        .detach();
    }

    fn stop_segment_playback(&mut self, cx: &mut Context<Self>) {
        self.playing_segment = None;
        self.loading_segment = None;
        cx.notify();
        parley_core::audio::playback::stop();
    }

    fn on_segment_playback_ended(&mut self, cx: &mut Context<Self>) {
        self.playing_segment = None;
        self.loading_segment = None;
        cx.notify();
    }

    // ---- Confidence indicator ---------------------------------------------

    /// Read `ui_config.showConfidenceIndicator` (default `true`, matching
    /// `ConfigContext.tsx`'s `readStoredBool(.., true)`) once per `load()`.
    /// Read directly via `SettingsRepository` rather than `SettingsCache`
    /// (the Settings page's own global) — this view doesn't depend on the
    /// Settings page being visited first.
    fn load_confidence_setting(&mut self, generation: u64, cx: &mut Context<Self>) {
        let Some(pool) = AppServices::global(cx).pool() else {
            return;
        };
        let io = Io::global(cx);
        cx.spawn(async move |this, cx| {
            let result = io.spawn(async move { SettingsRepository::get_setting_json(&pool, KEY_UI_CONFIG).await }).await;
            let show = match result {
                Ok(Ok(Some(json))) => serde_json::from_str::<serde_json::Value>(&json)
                    .ok()
                    .and_then(|v| v.get("showConfidenceIndicator").and_then(|v| v.as_bool()))
                    .unwrap_or(true),
                _ => true,
            };
            let _ = this.update(cx, |this, cx| {
                if this.generation != generation {
                    return;
                }
                this.show_confidence = show;
                cx.notify();
            });
        })
        .detach();
    }

    // ---- Meeting notes ------------------------------------------------------
    // Mirrors `MeetingNotesPanel.tsx`: load silently (no loading/error UI),
    // oldest-first, append-only (delete + re-add instead of editing).

    fn load_notes(&mut self, generation: u64, cx: &mut Context<Self>) {
        let Some(pool) = AppServices::global(cx).pool() else {
            return;
        };
        let Some(id) = self.meeting_id.clone() else {
            return;
        };
        let io = Io::global(cx);
        cx.spawn(async move |this, cx| {
            let result = io.spawn(async move { MeetingNotesRepository::list_by_meeting(&pool, &id).await }).await;
            let _ = this.update(cx, |this, cx| {
                if this.generation != generation {
                    return;
                }
                if let Ok(Ok(notes)) = result {
                    this.notes = notes;
                }
                // Silent on error, matching the React panel (console.error only).
                cx.notify();
            });
        })
        .detach();
    }

    fn open_note_composer(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        self.notes_composing = true;
        self.notes_draft.update(cx, |s, cx| s.set_value("", window, cx));
        cx.notify();
    }

    fn cancel_note_composer(&mut self, cx: &mut Context<Self>) {
        self.notes_composing = false;
        cx.notify();
    }

    fn save_note(&mut self, _window: &mut Window, cx: &mut Context<Self>) {
        let Some(id) = self.meeting_id.clone() else {
            return;
        };
        let Some(pool) = AppServices::global(cx).pool() else {
            return;
        };
        let body = self.notes_draft.read(cx).value().trim().to_string();
        if body.is_empty() || self.notes_saving {
            return;
        }
        self.notes_saving = true;
        cx.notify();

        let io = Io::global(cx);
        let generation = self.generation;
        cx.spawn(async move |this, cx| {
            let result = io
                .spawn(async move { MeetingNotesRepository::create(&pool, &id, &body, NOTE_SOURCE_MANUAL).await })
                .await;
            let _ = this.update_in(cx, |this, window, cx| {
                this.notes_saving = false;
                if this.generation != generation {
                    return;
                }
                match result {
                    Ok(Ok(note)) => {
                        this.notes.push(note);
                        this.notes_composing = false;
                        this.notes_draft.update(cx, |s, cx| s.set_value("", window, cx));
                    }
                    Ok(Err(e)) => notify(cx, Notification::error(format!("Failed to add note: {e}"))),
                    Err(e) => notify(cx, Notification::error(format!("Failed to add note: {e}"))),
                }
                cx.notify();
            });
        })
        .detach();
    }

    /// Optimistic removal, mirroring `MeetingNotesPanel.tsx`'s
    /// `handleDelete`: removed from view immediately, re-inserted
    /// (sorted back into place) on failure.
    fn delete_note(&mut self, id: String, cx: &mut Context<Self>) {
        let Some(pool) = AppServices::global(cx).pool() else {
            return;
        };
        let removed = self.notes.iter().position(|n| n.id == id).map(|ix| self.notes.remove(ix));
        cx.notify();

        let io = Io::global(cx);
        let generation = self.generation;
        cx.spawn(async move |this, cx| {
            let result = io.spawn(async move { MeetingNotesRepository::delete(&pool, &id).await }).await;
            let ok = matches!(result, Ok(Ok(true)));
            let _ = this.update(cx, |this, cx| {
                if this.generation != generation {
                    return;
                }
                if !ok {
                    if let Some(note) = removed {
                        this.notes.push(note);
                        this.notes.sort_by(|a, b| a.created_at.cmp(&b.created_at));
                    }
                    notify(cx, Notification::error("Failed to delete note"));
                }
                cx.notify();
            });
        })
        .detach();
    }

    // ---- Per-meeting action items -------------------------------------------
    // Mirrors `ActionItemsPanel.tsx`: `list_by_meeting`'s order (open first,
    // oldest-first within each group) is used as-is, no client-side re-sort.

    fn load_action_items(&mut self, generation: u64, cx: &mut Context<Self>) {
        let Some(pool) = AppServices::global(cx).pool() else {
            self.action_items_loading = false;
            return;
        };
        let Some(id) = self.meeting_id.clone() else {
            self.action_items_loading = false;
            return;
        };
        let io = Io::global(cx);
        cx.spawn(async move |this, cx| {
            let result = io.spawn(async move { ActionItemsRepository::list_by_meeting(&pool, &id).await }).await;
            let _ = this.update(cx, |this, cx| {
                if this.generation != generation {
                    return;
                }
                this.action_items_loading = false;
                if let Ok(Ok(items)) = result {
                    this.action_items = items;
                }
                cx.notify();
            });
        })
        .detach();
    }

    fn on_action_items_extracted(&mut self, event: &crate::core_events::CoreEvent, cx: &mut Context<Self>) {
        #[derive(serde::Deserialize)]
        struct Payload {
            meeting_id: String,
        }
        let Some(payload) = event.decode::<Payload>() else {
            return;
        };
        if self.meeting_id.as_deref() != Some(payload.meeting_id.as_str()) {
            return;
        }
        self.ai_extracting = false;
        self.load_action_items(self.generation, cx);
    }

    fn toggle_action_item_status(&mut self, id: String, currently_done: bool, cx: &mut Context<Self>) {
        let Some(pool) = AppServices::global(cx).pool() else {
            return;
        };
        let next = if currently_done { STATUS_OPEN } else { STATUS_DONE };
        if let Some(item) = self.action_items.iter_mut().find(|i| i.id == id) {
            item.status = next.to_string();
        }
        cx.notify();

        let io = Io::global(cx);
        let generation = self.generation;
        let id_for_task = id.clone();
        cx.spawn(async move |this, cx| {
            let result = io.spawn(async move { ActionItemsRepository::set_status(&pool, &id_for_task, next).await }).await;
            if !matches!(result, Ok(Ok(Some(_)))) {
                let _ = this.update(cx, |this, cx| {
                    if this.generation == generation {
                        this.load_action_items(generation, cx);
                    }
                });
            }
        })
        .detach();
    }

    fn start_action_item_edit(&mut self, item_id: String, window: &mut Window, cx: &mut Context<Self>) {
        let Some(item) = self.action_items.iter().find(|i| i.id == item_id) else {
            return;
        };
        let text = item.text.clone();
        self.ai_editing_id = Some(item_id);
        self.ai_edit_input.update(cx, |s, cx| s.set_value(text, window, cx));
        cx.notify();
    }

    fn cancel_action_item_edit(&mut self, cx: &mut Context<Self>) {
        self.ai_editing_id = None;
        cx.notify();
    }

    fn save_action_item_edit(&mut self, cx: &mut Context<Self>) {
        let Some(id) = self.ai_editing_id.take() else {
            return;
        };
        let Some(pool) = AppServices::global(cx).pool() else {
            return;
        };
        let text = self.ai_edit_input.read(cx).value().trim().to_string();
        if text.is_empty() {
            return;
        }
        if let Some(item) = self.action_items.iter_mut().find(|i| i.id == id) {
            item.text = text.clone();
        }
        cx.notify();

        let io = Io::global(cx);
        let generation = self.generation;
        cx.spawn(async move |this, cx| {
            let result = io.spawn(async move { ActionItemsRepository::update(&pool, &id, Some(&text), None, None).await }).await;
            if !matches!(result, Ok(Ok(Some(_)))) {
                let _ = this.update(cx, |this, cx| {
                    if this.generation == generation {
                        this.load_action_items(generation, cx);
                    }
                });
            }
        })
        .detach();
    }

    fn delete_action_item(&mut self, id: String, cx: &mut Context<Self>) {
        let Some(pool) = AppServices::global(cx).pool() else {
            return;
        };
        self.action_items.retain(|i| i.id != id);
        cx.notify();

        let io = Io::global(cx);
        let generation = self.generation;
        cx.spawn(async move |this, cx| {
            let result = io.spawn(async move { ActionItemsRepository::delete(&pool, &id).await }).await;
            if !matches!(result, Ok(Ok(true))) {
                let _ = this.update(cx, |this, cx| {
                    if this.generation == generation {
                        this.load_action_items(generation, cx);
                    }
                });
            }
        })
        .detach();
    }

    fn open_action_item_composer(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        self.ai_adding = true;
        self.ai_add_input.update(cx, |s, cx| s.set_value("", window, cx));
        cx.notify();
    }

    fn cancel_action_item_composer(&mut self, cx: &mut Context<Self>) {
        self.ai_adding = false;
        cx.notify();
    }

    fn save_new_action_item(&mut self, cx: &mut Context<Self>) {
        let Some(id) = self.meeting_id.clone() else {
            return;
        };
        let Some(pool) = AppServices::global(cx).pool() else {
            return;
        };
        let text = self.ai_add_input.read(cx).value().trim().to_string();
        if text.is_empty() {
            return;
        }
        self.ai_adding = false;
        cx.notify();

        let io = Io::global(cx);
        let generation = self.generation;
        cx.spawn(async move |this, cx| {
            let item = NewActionItem { text, ..Default::default() };
            let result =
                io.spawn(async move { ActionItemsRepository::create(&pool, &id, &item, ACTION_ITEM_SOURCE_MANUAL).await }).await;
            let _ = this.update(cx, |this, cx| {
                if this.generation != generation {
                    return;
                }
                match result {
                    Ok(Ok(_)) => this.load_action_items(generation, cx),
                    Ok(Err(e)) => notify(cx, Notification::error(format!("Failed to add action item: {e}"))),
                    Err(e) => notify(cx, Notification::error(format!("Failed to add action item: {e}"))),
                }
            });
        })
        .detach();
    }

    /// "Extract from summary" / "Re-extract": try the transcript-grounded
    /// extractor first, falling back to the summary-markdown extractor if
    /// it errors (e.g. no transcript rows to window over).
    fn extract_action_items(&mut self, cx: &mut Context<Self>) {
        let Some(id) = self.meeting_id.clone() else {
            return;
        };
        let Some(pool) = AppServices::global(cx).pool() else {
            return;
        };
        if self.ai_extracting {
            return;
        }
        self.ai_extracting = true;
        cx.notify();

        let sink = AppServices::global(cx).sink.clone();
        let summary_markdown = self.summary_markdown.clone();
        let io = Io::global(cx);
        let generation = self.generation;
        cx.spawn(async move |this, cx| {
            let pool_for_config = pool.clone();
            let config = io.spawn(async move { SettingsRepository::get_model_config(&pool_for_config).await }).await;
            let Ok(Ok(Some(config))) = config else {
                let _ = this.update(cx, |this, cx| {
                    this.ai_extracting = false;
                    if this.generation == generation {
                        notify(cx, Notification::error("No model configured — set one up in Settings first."));
                    }
                    cx.notify();
                });
                return;
            };

            let provider = config.provider;
            let model = config.model;
            let pool_for_transcript = pool.clone();
            let sink_for_transcript = sink.clone();
            let id_for_transcript = id.clone();
            let provider_for_transcript = provider.clone();
            let model_for_transcript = model.clone();
            let transcript_result = io
                .spawn(async move {
                    parley_core::summary::transcript_action_items::extract_from_transcript(
                        sink_for_transcript.as_ref(),
                        &pool_for_transcript,
                        &id_for_transcript,
                        &provider_for_transcript,
                        &model_for_transcript,
                    )
                    .await
                })
                .await;

            let outcome = match transcript_result {
                Ok(Ok(count)) => Ok(count),
                _ => {
                    // Fall back to extracting from the stored summary markdown.
                    let id_for_summary = id.clone();
                    io.spawn(async move {
                        parley_core::summary::action_extraction::extract_for_meeting(
                            sink.as_ref(),
                            &pool,
                            &id_for_summary,
                            &summary_markdown,
                            &provider,
                            &model,
                        )
                        .await
                    })
                    .await
                    .unwrap_or_else(|e| Err(format!("Task panicked: {e}")))
                }
            };

            let _ = this.update(cx, |this, cx| {
                this.ai_extracting = false;
                if this.generation != generation {
                    return;
                }
                match outcome {
                    Ok(_) => this.load_action_items(generation, cx),
                    Err(e) => notify(cx, Notification::error(format!("Extraction failed: {e}"))),
                }
                cx.notify();
            });
        })
        .detach();
    }
}

/// How much of the recording a click on a chapter, bookmark or citation
/// plays.
const JUMP_PLAY_SECS: f64 = 60.0;

/// Show a notification on the main window from a context that doesn't carry
/// a `Window` (an async task after an `.await`, or a core-event
/// subscription callback) — via the `MainWindow` global stashed at startup.
fn notify(cx: &mut App, notification: Notification) {
    crate::ui::with_main_window(cx, |window, cx| window.push_notification(notification, cx));
}

/// The floating speaker-edit popover, rendered from the page root (outside
/// the virtualized transcript `list()`) as a `deferred(anchored()...)`
/// positioned at the chip's click point — mirrors
/// `EditableSpeakerChip.tsx`'s popover. gpui-kit's `Popover` doesn't anchor
/// cleanly to a trigger living inside a recycled `list()` row, so this
/// tracks the click position in
/// [`SpeakerEditState::click_position`] instead and positions the card
/// directly. Free function (not a method) because [`MeetingView::render`]
/// only has an `Entity<MeetingView>` for the closures it hands out, not a
/// second `&mut Context<Self>` borrow.
fn render_floating_speaker_edit(state: &SpeakerEditState, entity: Entity<MeetingView>, cx: &mut App) -> AnyElement {
    let single = state.single_segment;
    // In single-segment mode the panel picks a person for one line, so it
    // behaves like naming an unnamed cluster whatever the chip is now.
    let is_named_profile = state.voice_profile_id.is_some() && !single;
    let merging = state.merge_target.is_some();
    let merge_target_profile = state.merge_target.as_ref().and_then(|id| state.profiles.iter().find(|p| &p.id == id));
    let name = state.name_input.read(cx).value().trim().to_string();
    let can_save = speaker_chip::can_save_speaker_edit(state.merge_target.as_deref(), &name) && !state.saving;

    let mut panel = v_flex()
        .id("speaker-edit-popover")
        .occlude()
        .w(px(360.))
        .gap_2()
        .p_3()
        .rounded_md()
        .border_1()
        .border_color(ActiveTheme::theme(cx).border)
        .bg(ActiveTheme::theme(cx).background)
        .shadow_lg()
        .on_mouse_down_out({
            let entity = entity.clone();
            move |_, _, cx| entity.update(cx, |this, cx| this.close_speaker_edit(cx))
        })
        .on_key_down({
            let entity = entity.clone();
            move |ev, _, cx| {
                if ev.keystroke.key == "escape" {
                    entity.update(cx, |this, cx| this.close_speaker_edit(cx));
                }
            }
        })
        .child(div().text_sm().font_semibold().child(if single {
            "Who said this?"
        } else {
            speaker_chip::edit_panel_title(is_named_profile)
        }));

    if state.cluster_size > 1 {
        let entity_all = entity.clone();
        let entity_one = entity.clone();
        panel = panel.child(
            h_flex()
                .gap_1()
                .flex_wrap()
                .child(
                    Button::new("speaker-scope-all")
                        .xsmall()
                        .when(!single, |b| b.primary())
                        .when(single, |b| b.outline())
                        .label(format!("All {} \u{201c}{}\u{201d} lines", state.cluster_size, state.speaker))
                        .on_click(move |_, window, cx| {
                            entity_all.update(cx, |this, cx| this.set_speaker_edit_scope(false, window, cx));
                        }),
                )
                .child(
                    Button::new("speaker-scope-one")
                        .xsmall()
                        .when(single, |b| b.primary())
                        .when(!single, |b| b.outline())
                        .label("Only this line")
                        .tooltip("Use when this speaker is really two people")
                        .on_click(move |_, window, cx| {
                            entity_one.update(cx, |this, cx| this.set_speaker_edit_scope(true, window, cx));
                        }),
                ),
        );
    }

    if !merging && !state.attendees.is_empty() {
        let suggestions = speaker_chip::filter_attendee_suggestions(&state.attendees, &name);
        if !suggestions.is_empty() {
            let mut chips = h_flex().gap_1().flex_wrap();
            for s in suggestions {
                let label = s.label.clone();
                let email = s.email.clone();
                let entity_for_pick = entity.clone();
                let selected = name == s.label;
                chips = chips.child(
                    Button::new(SharedString::from(format!("attendee-suggestion-{}", s.label)))
                        .xsmall()
                        .when(selected, |b| b.primary())
                        .when(!selected, |b| b.outline())
                        .label(s.label.clone())
                        .on_click(move |_, window, cx| {
                            entity_for_pick.update(cx, |this, cx| {
                                this.apply_attendee_suggestion(label.clone(), email.clone(), window, cx)
                            });
                        }),
                );
            }
            panel = panel
                .child(div().text_xs().text_color(ActiveTheme::theme(cx).muted_foreground).child("From this meeting's calendar"))
                .child(chips);
        }
    } else if !merging && !is_named_profile {
        panel = panel.child(
            div()
                .text_xs()
                .text_color(ActiveTheme::theme(cx).muted_foreground)
                .child("Link this meeting to a calendar event to pick from its attendees."),
        );
    }

    panel = panel.child(div().text_xs().child("Saved speaker"));
    if state.profiles_loading {
        panel = panel.child(div().text_xs().child("Loading saved speakers…"));
    } else if let Some(picker) = &state.picker {
        panel = panel.child(Select::new(picker)
            .placeholder("Choose a speaker")
            .accessibility_label("Saved speaker")
            .search_placeholder("Search speakers…")
            .menu_max_h(px(240.))
            .disabled(state.saving));
        panel = panel.child(div().text_xs().text_color(ActiveTheme::theme(cx).muted_foreground)
            .child(if state.has_ranking {
                "Ranked by the last identification of this speaker group. Similarity is not a probability for this line."
            } else {
                "No saved ranking. Run Identify speakers to compare voices; you can still choose anyone by name."
            }));
    }

    if merging && single {
        if let Some(target) = merge_target_profile {
            panel = panel.child(
                div()
                    .text_xs()
                    .text_color(ActiveTheme::theme(cx).muted_foreground)
                    .child(format!("Only this line will be attributed to {}.", target.name)),
            );
        }
    } else if merging {
        if let Some(target) = merge_target_profile {
            panel = panel.child(
                div()
                    .text_xs()
                    .text_color(ActiveTheme::theme(cx).muted_foreground)
                    .child(format!(
                        "Eligible clean samples will be added to {}{}. All lines with this speaker label in this meeting will be relabelled.",
                        target.name,
                        target.email.as_deref().map(|e| format!(" ({e})")).unwrap_or_default(),
                    )),
            );
        }
    } else {
        panel = panel
            .when(is_named_profile, |p| p.child(div().text_xs()
                .text_color(ActiveTheme::theme(cx).muted_foreground)
                .child("Choose a saved speaker above to reassign these lines. Editing the name below renames the profile across meetings.")))
            .child(div().w_full().child(Input::new(&state.name_input)))
            .when(!single, |p| p.child(div().w_full().child(Input::new(&state.email_input))));
    }

    if let Some(err) = &state.error {
        panel = panel.child(div().text_xs().text_color(ActiveTheme::theme(cx).danger).child(err.clone()));
    }

    let entity_cancel = entity.clone();
    let entity_save = entity.clone();
    panel.child(
        h_flex()
            .gap_2()
            .child(
                Button::new("save-speaker-edit")
                    .primary()
                    .label(if state.saving { "Saving…" } else if merging && !single { "Assign all lines" } else { "Save" })
                    .loading(state.saving)
                    .disabled(!can_save)
                    .on_click(move |_, _, cx| {
                        entity_save.update(cx, |this, cx| this.save_speaker_edit(cx));
                    }),
            )
            .child(
                Button::new("cancel-speaker-edit")
                    .ghost()
                    .label("Cancel")
                    .disabled(state.saving)
                    .on_click(move |_, _, cx| {
                        entity_cancel.update(cx, |this, cx| this.close_speaker_edit(cx));
                    }),
            )
            .when(speaker_chip::can_unassign_speaker(&state.speaker), |row| {
                let entity_unassign = entity.clone();
                row.child(div().flex_1()).child(
                    Button::new("unassign-speaker-edit")
                        .ghost()
                        .danger()
                        .label("Unassign")
                        .tooltip(if single {
                            "This line goes back to an unnamed speaker"
                        } else {
                            "These lines go back to an unnamed speaker"
                        })
                        .disabled(state.saving)
                        .on_click(move |_, _, cx| {
                            entity_unassign.update(cx, |this, cx| this.unassign_speaker_edit(cx));
                        }),
                )
            }),
    )
    .into_any_element()
}

fn syntax_style(cx: &App) -> SyntaxStyle {
    let theme = ActiveTheme::theme(cx);
    SyntaxStyle {
        block_label: None,
        block_label_gen: 0,
        block_ref_count: None,
        marker: theme.muted_foreground.opacity(0.6),
        code: theme.foreground,
        code_bg: theme.muted.opacity(0.5),
        link: theme.primary,
        tag: theme.accent_foreground,
        quote: theme.muted_foreground,
        alert_note: theme.primary,
        alert_tip: theme.primary,
        alert_important: theme.primary,
        alert_warning: theme.primary,
        alert_caution: theme.danger,
        alert_icons: None,
        rule: theme.border,
        mark_bg: theme.primary.opacity(0.25),
        popover_bg: theme.popover,
        popover_border: theme.border,
        popover_fg: theme.popover_foreground,
        popover_hover: theme.accent,
        popover_divider: theme.border,
        popover_danger: theme.danger,
        mono: gpui_kit::font("monospace"),
        property_icon: None,
    }
}

impl Render for MeetingView {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        if self.search_reset_pending {
            self.search_reset_pending = false;
            self.transcript_search_input.update(cx, |input, cx| input.set_value("", window, cx));
        }
        let Some(_id) = self.meeting_id.clone() else {
            return v_flex().size_full().p_6().child("Select a meeting from the sidebar.");
        };

        v_flex()
            .size_full()
            .child(self.render_header(cx))
            .when(self.calendar_picker_open, |this| {
                this.child(self.render_calendar_picker(cx))
            })
            .when(self.history_open, |this| {
                this.child(self.processing_history.clone())
            })
            .when(self.retranscribe_open, |this| {
                this.child(self.render_retranscribe(cx))
            })
            .when(self.identify_open, |this| {
                this.child(self.render_identify(cx))
            })
            .child(
                h_flex()
                    .flex_1()
                    .min_h_0()
                    .child(
                        div()
                            .flex_1()
                            .min_w_0()
                            .h_full()
                            .border_r_1()
                            .border_color(cx.theme().border)
                            .child(self.render_transcript(cx)),
                    )
                    .child(
                        v_flex()
                            .flex_1()
                            .min_w_0()
                            .h_full()
                            .child(self.render_right_tabs(cx))
                            .child(
                                div()
                                    .flex_1()
                                    .min_h_0()
                                    .when(self.right_tab == RightTab::Summary, |this| {
                                        this.child(self.render_summary(cx))
                                    })
                                    .when(self.right_tab == RightTab::Ask, |this| {
                                        this.child(self.ask_panel.clone())
                                    })
                                    .when(self.right_tab == RightTab::Insights, |this| {
                                        this.child(self.insights_panel.clone())
                                    }),
                            ),
                    ),
            )
            .when_some(self.speaker_edit.clone(), |this, state| {
                let entity = cx.entity();
                let content = render_floating_speaker_edit(&state, entity, cx);
                this.child(
                    deferred(anchored().position(state.click_position).snap_to_window().child(content))
                        .with_priority(50),
                )
            })
    }
}

impl MeetingView {
    /// "Summary | Ask | Insights" switcher above the right-hand pane.
    fn render_right_tabs(&self, cx: &mut Context<Self>) -> impl IntoElement {
        let entity = cx.entity().downgrade();
        h_flex()
            .w_full()
            .px_2()
            .border_b_1()
            .border_color(cx.theme().border)
            .child(
                TabBar::new("meeting-right-tabs")
                    .underline()
                    .selected_index(match self.right_tab {
                        RightTab::Summary => 0,
                        RightTab::Ask => 1,
                        RightTab::Insights => 2,
                    })
                    .child(Tab::new().icon(Lucide::FileText).label("Summary"))
                    .child(Tab::new().icon(Lucide::Sparkles).label("Ask"))
                    .child(Tab::new().icon(Lucide::ChartBar).label("Insights"))
                    .on_click(move |ix, _, cx| {
                        let tab = match *ix {
                            1 => RightTab::Ask,
                            2 => RightTab::Insights,
                            _ => RightTab::Summary,
                        };
                        let _ = entity.update(cx, |this, cx| this.set_right_tab(tab, cx));
                    }),
            )
    }

    fn render_header(&self, cx: &mut Context<Self>) -> impl IntoElement {
        let date = self
            .created_at
            .map(format::format_header_date)
            .unwrap_or_default();
        let can_retranscribe = self.meeting_folder_path.is_some();

        h_flex()
            .w_full()
            .items_center()
            .justify_between()
            .gap_3()
            .px_4()
            .pt_5()
            .pb_4()
            .border_b_1()
            .border_color(cx.theme().border)
            .child(if self.editing_title {
                h_flex()
                    .flex_1()
                    .min_w_0()
                    .gap_2()
                    .items_center()
                    .child(div().flex_1().min_w_0().child(Input::new(&self.title_input)))
                    .child(
                        Button::new("save-title")
                            .primary()
                            .label("Save")
                            .on_click(cx.listener(|this, _, _, cx| this.save_title(cx))),
                    )
                    .child(
                        Button::new("cancel-title")
                            .ghost()
                            .label("Cancel")
                            .on_click(cx.listener(|this, _, _, cx| {
                                this.editing_title = false;
                                cx.notify();
                            })),
                    )
                    .into_any_element()
            } else {
                v_flex()
                    .flex_1()
                    .min_w_0()
                    .gap_2()
                    .child(
                        h_flex()
                            .gap_1()
                            .items_center()
                            .child(
                                div()
                                    .text_2xl()
                                    .font_semibold()
                                    .overflow_hidden()
                                    .text_ellipsis()
                                    .child(if self.title.trim().is_empty() {
                                        "Untitled meeting".to_string()
                                    } else {
                                        self.title.clone()
                                    }),
                            )
                            .child(
                                Button::new("edit-title")
                                    .ghost()
                                    .small()
                                    .icon(Lucide::Pencil)
                                    .tooltip("Rename")
                                    .on_click(cx.listener(|this, _, window, cx| {
                                        let title = this.title.clone();
                                        this.title_input.update(cx, |state, cx| state.set_value(title, window, cx));
                                        this.editing_title = true;
                                        cx.notify();
                                    })),
                            ),
                    )
                    .child(
                        h_flex()
                            .gap_2()
                            .items_center()
                            .flex_wrap()
                            .when(!date.is_empty(), |this| this.child(crate::ui::meta_chip(Lucide::Clock, date, cx)))
                            .children(self.render_calendar_row(cx)),
                    )
                    .into_any_element()
            })
            .child(
                h_flex()
                    .max_w(px(520.))
                    .flex_wrap()
                    .justify_end()
                    .gap_2()
                    .items_center()
                    .child(
                        Button::new("processing-history")
                            .outline()
                            .label("Processing details")
                            .disabled(self.retranscribe_in_progress || self.identify_busy)
                            .on_click(cx.listener(|this, _, _, cx| {
                                this.history_open = !this.history_open;
                                if this.history_open {
                                    this.identify_open = false;
                                    this.retranscribe_open = false;
                                    this.identity_request += 1;
                                }
                                this.processing_history
                                    .update(cx, |history, cx| history.refresh(cx));
                                cx.notify();
                            })),
                    )
                    .child(
                        Button::new("identify-speakers")
                            .outline()
                            .label({
                                let pending = self.identity_suggestions.iter().filter(|s| s.profile_id.is_some() && !s.confirmed).count();
                                if pending > 0 { format!("Confirm identities ({pending})") } else { "Identify speakers".into() }
                            })
                            .disabled(
                                self.transcripts.is_empty()
                                    || self.identify_busy
                                    || self.retranscribe_in_progress,
                            )
                            .on_click(cx.listener(|this, _, _, cx| {
                                this.identify_open = !this.identify_open;
                                if this.identify_open {
                                    this.retranscribe_open = false;
                                    this.history_open = false;
                                }
                                if this.identify_open {
                                    this.load_identity_suggestions(cx);
                                }
                                this.retranscribe_error = None;
                                this.identify_editor
                                    .update(cx, |editor, cx| editor.reload(cx));
                                cx.notify();
                            })),
                    )
                    .child(
                        Button::new("retranscribe")
                            .ghost()
                            .icon(Lucide::RefreshCw)
                            .label(if self.transcripts.is_empty() {
                                "Process recording"
                            } else {
                                "Retranscribe"
                            })
                            .tooltip("Choose language and speakers, then process saved audio")
                            .disabled(
                                !can_retranscribe
                                    || self.retranscribe_in_progress
                                    || self.identify_busy,
                            )
                            .on_click(cx.listener(|this, _, window, cx| {
                                this.open_retranscribe(window, cx)
                            })),
                    )
                    .child(
                        Button::new("rediarize")
                            .ghost()
                            .icon(Lucide::Users)
                            .tooltip("Re-detect speakers")
                            .loading(self.rediarize_in_progress)
                            .disabled(
                                !can_retranscribe
                                    || self.rediarize_in_progress
                                    || self.retranscribe_in_progress
                                    || self.identify_busy,
                            )
                            .on_click(
                                cx.listener(|this, _, window, cx| this.start_rediarize(window, cx)),
                            ),
                    )
                    .child(
                        Button::new("copy-summary-md")
                            .ghost()
                            .icon(IconName::Copy)
                            .tooltip("Copy meeting as Markdown")
                            .on_click(cx.listener(|this, _, _, cx| this.export_copy("markdown", cx))),
                    )
                    .child(
                        Button::new("copy-summary-json")
                            .ghost()
                            .icon(Lucide::Braces)
                            .tooltip("Copy meeting as JSON")
                            .on_click(cx.listener(|this, _, _, cx| this.export_copy("json", cx))),
                    )
                    .child(self.render_export_menu(cx))
                    .child(
                        Button::new("delete-meeting")
                            .danger()
                            .icon(IconName::Delete)
                            .tooltip("Delete meeting")
                            .on_click(cx.listener(|this, _, window, cx| this.confirm_delete(window, cx))),
                    ),
            )
    }

    /// The calendar-event link row shown under the header — mirrors
    /// `CalendarEventPanel.tsx`.
    /// The linked calendar event as header chips (event + time, place,
    /// attendees) with Change / Unlink, or a "Link calendar event" button.
    fn render_calendar_row(&self, cx: &mut Context<Self>) -> Vec<AnyElement> {
        if self.calendar_loading {
            return Vec::new();
        }

        let Some(event) = self.calendar_event.clone() else {
            return vec![
                    Button::new("link-calendar-event")
                        .ghost()
                        .xsmall()
                        .icon(Lucide::Link2)
                        .label("Link calendar event")
                        .disabled(self.calendar_mutating)
                        .on_click(cx.listener(|this, _, _, cx| this.open_calendar_picker(cx)))
                        .into_any_element(),
            ];
        };

        let attendee_count = event.attendees.len();
        let title = event.summary.clone().filter(|s| !s.is_empty()).unwrap_or_else(|| "(untitled event)".to_string());
        let mut items = vec![crate::ui::meta_chip(
            Lucide::Calendar,
            format!("{title} · {}", calendar_format::format_time_range(event.start_at, event.end_at)),
            cx,
        )
        .into_any_element()];
        if let Some(loc) = event.location.clone().filter(|l| !l.is_empty()) {
            items.push(crate::ui::meta_chip(Lucide::MapPin, loc, cx).into_any_element());
        }
        if attendee_count > 0 {
            items.push(
                crate::ui::meta_chip(
                    Lucide::Users,
                    format!("{attendee_count} attendee{}", if attendee_count == 1 { "" } else { "s" }),
                    cx,
                )
                .into_any_element(),
            );
        }
        items.push(
            Button::new("change-calendar-event")
                .ghost()
                .xsmall()
                .label("Change")
                .disabled(self.calendar_mutating)
                .on_click(cx.listener(|this, _, _, cx| this.open_calendar_picker(cx)))
                .into_any_element(),
        );
        items.push(
            Button::new("unlink-calendar-event")
                .ghost()
                .xsmall()
                .icon(Lucide::Link2Off)
                .tooltip("Unlink calendar event")
                .disabled(self.calendar_mutating)
                .on_click(cx.listener(|this, _, _, cx| this.unlink_calendar_event(cx)))
                .into_any_element(),
        );
        items
    }

    /// "Link calendar event" picker: events within `±CALENDAR_WINDOW_DAYS`
    /// of the meeting's `created_at`, closest-first — mirrors
    /// `CalendarEventPicker.tsx`.
    fn render_calendar_picker(&self, cx: &mut Context<Self>) -> impl IntoElement {
        let anchor = self.created_at.unwrap_or_else(chrono::Utc::now);
        let current_id = self.calendar_event.as_ref().map(|e| e.id.clone());

        let mut body = v_flex()
            .w_full()
            .gap_2()
            .p_4()
            .border_b_1()
            .border_color(cx.theme().border)
            .bg(cx.theme().muted.opacity(0.3))
            .child(
                h_flex()
                    .w_full()
                    .items_center()
                    .justify_between()
                    .child(div().text_sm().font_semibold().child("Link calendar event"))
                    .child(
                        h_flex()
                            .gap_1()
                            .items_center()
                            // Unlinking from the picker too, so a wrong
                            // auto-link can be undone wherever the user goes.
                            .when(current_id.is_some(), |el| {
                                el.child(
                                    Button::new("picker-unlink-calendar-event")
                                        .ghost()
                                        .icon(Lucide::Link2Off)
                                        .label("Unlink")
                                        .disabled(self.calendar_mutating)
                                        .on_click(cx.listener(|this, _, _, cx| this.unlink_calendar_event(cx))),
                                )
                            })
                            .child(
                                Button::new("close-calendar-picker")
                                    .ghost()
                                    .icon(Lucide::X)
                                    .on_click(cx.listener(|this, _, _, cx| this.close_calendar_picker(cx))),
                            ),
                    ),
            );

        if let Some(err) = &self.calendar_picker_error {
            body = body.child(div().text_xs().text_color(cx.theme().danger).child(err.clone()));
        }

        if self.calendar_picker_events.is_some() {
            body = body.child(div().w_full().child(Input::new(&self.calendar_picker_query)));
        }

        let query = self.calendar_picker_query.read(cx).value().trim().to_lowercase();
        let filtered: Option<Vec<CalendarEvent>> = self.calendar_picker_events.as_ref().map(|events| {
            if query.is_empty() {
                events.clone()
            } else {
                events
                    .iter()
                    .filter(|e| {
                        e.summary.as_deref().unwrap_or_default().to_lowercase().contains(&query)
                            || e.location.as_deref().unwrap_or_default().to_lowercase().contains(&query)
                    })
                    .cloned()
                    .collect()
            }
        });

        match &filtered {
            None => {
                body = body.child(div().text_xs().text_color(cx.theme().muted_foreground).child("Loading…"));
            }
            Some(events) if events.is_empty() => {
                body = body.child(
                    div()
                        .text_xs()
                        .text_color(cx.theme().muted_foreground)
                        .child(format!(
                            "No events within ±{CALENDAR_WINDOW_DAYS} days of the recording. Try refreshing your \
                             calendar in Settings."
                        )),
                );
            }
            Some(events) => {
                body = body.child(
                    v_flex()
                        .w_full()
                        .gap_1()
                        .max_h(px(280.))
                        .overflow_y_scrollbar()
                        .children(events.iter().take(30).map(|e| {
                            let is_current = current_id.as_deref() == Some(e.id.as_str());
                            let event_id = e.id.clone();
                            h_flex()
                                .id(SharedString::from(format!("calendar-event-{}", e.id)))
                                .w_full()
                                .items_center()
                                .justify_between()
                                .gap_2()
                                .px_2()
                                .py_1()
                                .rounded_md()
                                .when(is_current, |el| el.bg(cx.theme().accent.opacity(0.4)))
                                .child(
                                    v_flex()
                                        .flex_1()
                                        .min_w_0()
                                        .gap_0p5()
                                        .child(
                                            div()
                                                .text_sm()
                                                .overflow_hidden()
                                                .child(e.summary.clone().filter(|s| !s.is_empty()).unwrap_or_else(|| "(untitled event)".to_string())),
                                        )
                                        .child(
                                            div()
                                                .text_xs()
                                                .text_color(cx.theme().muted_foreground)
                                                .child(format!(
                                                    "{} · {}",
                                                    e.start_at.with_timezone(&chrono::Local).format("%a %b %-d, %-I:%M %p"),
                                                    calendar_format::format_offset(e.start_at, anchor),
                                                )),
                                        ),
                                )
                                .child(
                                    Button::new(SharedString::from(format!("pick-calendar-event-{}", e.id)))
                                        .when(is_current, |b| b.outline())
                                        .when(!is_current, |b| b.ghost())
                                        .label(if is_current { "Linked" } else { "Link" })
                                        .disabled(self.calendar_mutating || is_current)
                                        .on_click(cx.listener(move |this, _, _, cx| {
                                            this.pick_calendar_event(Some(event_id.clone()), cx)
                                        })),
                                )
                                .into_any_element()
                        })),
                );
            }
        }

        body
    }

    fn render_processing_language(&self, cx: &mut Context<Self>) -> impl IntoElement {
        use gpui_kit::component::menu::{DropdownMenu as _, PopupMenuItem};
        let view = cx.entity().downgrade();
        Button::new("processing-language")
            .outline()
            .label(crate::views::import::language_label(
                &self.retranscribe_language,
            ))
            .dropdown_menu(move |menu, _, _| {
                let mut menu = menu;
                for &(code, label) in crate::views::import::LANGUAGES {
                    let view = view.clone();
                    menu = menu.item(PopupMenuItem::new(label).on_click(move |_, _, cx| {
                        let _ = view.update(cx, |this, cx| {
                            this.retranscribe_language = code.to_string();
                            cx.notify();
                        });
                    }));
                }
                menu
            })
    }

    fn render_retranscribe(&self, cx: &mut Context<Self>) -> impl IntoElement {
        v_flex()
            .id("meeting-processing-options").max_h(px(500.)).overflow_y_scrollbar()
            .w_full()
            .gap_2()
            .p_4()
            .border_b_1()
            .border_color(cx.theme().border)
            .bg(cx.theme().muted.opacity(0.3))
            .child(div().text_sm().font_semibold().child(if self.processing_operation == "speakers" { "Re-detect speakers" } else if self.transcripts.is_empty() { "Process recording" } else { "Reprocess meeting" }))
            .child(self.pipeline_editor.clone())
            .child(div().text_xs().text_color(cx.theme().muted_foreground).child(if self.processing_operation=="speakers" {"Re-detection keeps words and timestamps. Use Identify speakers to match the resulting groups to saved voices."} else {"These choices apply only to this job. Language affects the transcript; summary output language is configured separately."}))
            .when(!self.retranscribe_in_progress && self.retranscribe_error.is_none(), |this| {
                this.child(
                    h_flex()
                        .gap_2()
                        .flex_wrap()
                        .items_center()
                        .when(self.processing_operation!="speakers",|row|row.child(div().text_xs().text_color(cx.theme().muted_foreground).child("Language")).child(self.render_processing_language(cx)))
                        .child(div().text_xs().text_color(cx.theme().muted_foreground).child("Speakers"))
                        .child(div().w_20().child(Input::new(&self.retranscribe_speakers)))
                        .child(
                            Button::new("start-retranscribe")
                                .primary()
                                .label("Start")
                                .disabled(!self.pipeline_editor.read(cx).ready())
                                .on_click(cx.listener(|this, _, window, cx| this.start_retranscribe(window, cx))),
                        )
                        .child(
                            Button::new("cancel-retranscribe-dialog")
                                .ghost()
                                .label("Cancel")
                                .on_click(cx.listener(|this, _, _, cx| this.close_retranscribe(cx))),
                        ),
                )
            })
            .when(self.retranscribe_in_progress, |this| {
                this.child(
                    h_flex()
                        .gap_2()
                        .items_center()
                        .child(
                            self.job_progress.clone(),
                        )
                        .child(
                            Button::new("cancel-retranscribe")
                                .ghost()
                                .icon(Lucide::X)
                                .label("Cancel")
                                .on_click(cx.listener(|this, _, window, cx| this.cancel_retranscribe(window, cx))),
                        ),
                )
            })
            .when_some(self.retranscribe_error.clone(), |this, err| {
                this.child(div().text_xs().text_color(cx.theme().danger).child(err)).child(
                    Button::new("close-retranscribe-error")
                        .ghost()
                        .label("Close")
                        .on_click(cx.listener(|this, _, _, cx| this.close_retranscribe(cx))),
                )
            })
    }

    fn load_identity_suggestions(&mut self, cx: &mut Context<Self>) {
        self.load_identity_suggestions_for_review(false, cx);
    }

    fn load_identity_suggestions_for_review(
        &mut self,
        show_if_available: bool,
        cx: &mut Context<Self>,
    ) {
        self.identity_request += 1;
        let request = self.identity_request;
        let Some(pool) = AppServices::global(cx).pool() else {
            return;
        };
        let Some(meeting) = self.meeting_id.clone() else {
            return;
        };
        let generation = self.generation;
        let io = Io::global(cx);
        cx.spawn(async move |this, cx| {
            let result = io
                .spawn(async move {
                    let version =
                        parley_core::audio::processing::runs::current_run(&pool, &meeting).await?;
                    let suggestions =
                        parley_core::audio::processing::service::current_identity_suggestions(
                            &pool, &meeting,
                        )
                        .await?;
                    Ok::<_, anyhow::Error>((suggestions, version))
                })
                .await;
            let _ = this.update(cx, |this, cx| {
                if this.generation != generation || this.identity_request != request {
                    return;
                }
                match result {
                    Ok(Ok((suggestions, version))) => {
                        let pending = suggestions.iter().any(|suggestion| {
                            suggestion.profile_id.is_some() && !suggestion.confirmed
                        });
                        this.identity_suggestions = suggestions;
                        this.identity_group_version = version;
                        if show_if_available
                            && pending
                            && !this.history_open
                            && !this.retranscribe_open
                        {
                            this.identify_open = true;
                            this.identify_editor
                                .update(cx, |editor, cx| editor.reload(cx));
                        }
                    }
                    Ok(Err(error)) => this.retranscribe_error = Some(error.to_string()),
                    Err(error) => this.retranscribe_error = Some(error.to_string()),
                }
                cx.notify();
            });
        })
        .detach();
    }

    fn confirm_identity_suggestion(
        &mut self,
        speaker: String,
        profile: String,
        replacement: Option<(Option<String>, Option<String>)>,
        cx: &mut Context<Self>,
    ) {
        if self.identify_busy {
            return;
        }
        self.identity_request += 1;
        let Some(version) = self.identity_group_version.clone() else {
            return;
        };
        let Some(pool) = AppServices::global(cx).pool() else {
            return;
        };
        let Some(meeting) = self.meeting_id.clone() else {
            return;
        };
        let generation = self.generation;
        let io = Io::global(cx);
        self.identify_busy = true;
        self.retranscribe_error = None;
        cx.notify();
        cx.spawn(async move |this, cx| {
            let result = io
                .spawn(async move {
                    use parley_core::audio::processing::service;
                    if let Some((profile_id, name)) = replacement {
                        service::confirm_identity_replacement(
                            &pool,
                            &meeting,
                            &version,
                            &speaker,
                            &profile,
                            service::IdentityAssignment { profile_id, name },
                        )
                        .await?;
                    } else {
                        service::confirm_identity_suggestion(
                            &pool, &meeting, &version, &speaker, &profile,
                        )
                        .await?;
                    }
                    Ok::<_, anyhow::Error>(speaker)
                })
                .await;
            let _ = this.update(cx, |this, cx| {
                if this.generation != generation {
                    return;
                }
                this.identify_busy = false;
                match result {
                    Ok(Ok(speaker)) => {
                        for suggestion in &mut this.identity_suggestions {
                            if suggestion.speaker == speaker {
                                suggestion.confirmed = true;
                                suggestion.current_profile_id = suggestion.profile_id.clone();
                                suggestion.current_name = suggestion.name.clone();
                                suggestion.replacement = false;
                            }
                        }
                        this.load_identity_suggestions(cx);
                        this.reload_transcript(cx);
                        this.processing_history
                            .update(cx, |history, cx| history.refresh(cx));
                    }
                    Ok(Err(error)) => this.retranscribe_error = Some(error.to_string()),
                    Err(error) => this.retranscribe_error = Some(error.to_string()),
                }
                cx.notify();
            });
        })
        .detach();
    }

    fn render_identity_suggestions(&self, cx: &mut Context<Self>) -> AnyElement {
        v_flex()
            .w_full()
            .gap_2()
            .children(self.identity_suggestions.iter().map(|suggestion| {
                let speaker = suggestion.speaker.clone();
                let profile = suggestion.profile_id.clone();
                let name = suggestion
                    .name
                    .clone()
                    .unwrap_or_else(|| "No confident proposal".into());
                let replacement = suggestion.replacement.then(|| {
                    (
                        suggestion.current_profile_id.clone(),
                        suggestion.current_name.clone(),
                    )
                });
                let mut description = v_flex().flex_1().min_w_0().gap_1().child(
                    div()
                        .text_sm()
                        .font_medium()
                        .child(suggestion.speaker.clone()),
                );
                if suggestion.replacement {
                    description = description
                        .child(
                            div().text_sm().child(format!(
                                "Current name: {}",
                                suggestion
                                    .current_name
                                    .as_deref()
                                    .unwrap_or("No saved identity")
                            )),
                        )
                        .child(
                            div()
                                .text_sm()
                                .child(format!("Proposed replacement: {name}")),
                        );
                } else if suggestion.confirmed {
                    description = description.child(div().text_sm().child(format!(
                        "Current name: {}",
                        suggestion.current_name.as_deref().unwrap_or(&name)
                    )));
                } else {
                    if let Some(current) = &suggestion.current_name {
                        description = description.child(
                            div()
                                .text_sm()
                                .child(format!("Current name: {current}")),
                        );
                    }
                    description = description.child(div().text_sm().child(if profile.is_some() {
                        format!("Proposed identity: {name}")
                    } else {
                        "No confident proposal".into()
                    }));
                }
                if let Some(score) = suggestion.score.filter(|_| !suggestion.confirmed) {
                    description = description.child(div().text_xs().child(format!("Voice similarity: {score:.2} (not a probability)")));
                }
                description = description.when_some(suggestion.reason.clone(), |body, reason| {
                    body.child(
                        div()
                            .text_xs()
                            .text_color(cx.theme().muted_foreground)
                            .child(reason),
                    )
                });
                h_flex()
                    .w_full()
                    .gap_2()
                    .items_center()
                    .child(description)
                    .when_some(profile.filter(|_| !suggestion.confirmed), |row, profile| {
                        row.child(
                            Button::new(format!("confirm-identity-{}", suggestion.speaker))
                                .primary()
                                .label(if suggestion.replacement {
                                    format!("Replace with {name}")
                                } else {
                                    format!("Confirm {name}")
                                })
                                .disabled(self.identify_busy)
                                .on_click(cx.listener(move |this, _, _, cx| {
                                    this.confirm_identity_suggestion(
                                        speaker.clone(),
                                        profile.clone(),
                                        replacement.clone(),
                                        cx,
                                    )
                                })),
                        )
                    })
            }))
            .into_any_element()
    }

    fn render_identify(&self, cx: &mut Context<Self>) -> AnyElement {
        let content = v_flex()
            .id("meeting-identification-options")
            // The scrollbar wrapper needs a definite viewport height here;
            // a maximum alone can collapse inside this auto-sized panel.
            .h(px(420.))
            .flex_shrink_0()
            .overflow_y_scrollbar()
            .w_full()
            .gap_3()
            .p_4()
            .child(div().text_sm().font_semibold().child("Identify speakers"))
            .child(div().text_xs().text_color(cx.theme().muted_foreground).child("Review identity proposals for existing speaker groups. Confirming a proposal names the speaker and adds eligible clean voice samples to their profile. Words and speaker boundaries stay unchanged."))
            .child(self.render_identity_suggestions(cx))
            .child(self.identify_editor.clone())
            .child(Checkbox::new("review-confirmed-identity-alternatives")
                .label("Review alternatives for existing names")
                .checked(self.review_identity_replacements)
                .disabled(self.identify_busy)
                .on_click(cx.listener(|this, checked: &bool, _, cx| {
                    this.review_identity_replacements = *checked;
                    cx.notify();
                })))
            .child(div().text_xs().text_color(cx.theme().muted_foreground).child("These choices apply to a new lookup. Saved proposals retain their original processing details."))
            .when(self.identify_busy, |body| body.child(self.job_progress.clone()))
            .when_some(self.retranscribe_error.clone(), |body, error| body.child(div().text_xs().text_color(cx.theme().danger).child(error)));
        v_flex()
            .w_full()
            .flex_shrink_0()
            .border_b_1()
            .border_color(cx.theme().border)
            .child(content)
            .child(
                h_flex()
                    .gap_2()
                    .flex_wrap()
                    .px_4()
                    .pb_4()
                    .pt_2()
                    .child(
                        Button::new("start-identify")
                            .primary()
                            .label(if self.identify_busy {
                                "Finding suggestions…"
                            } else {
                                "Find identity suggestions"
                            })
                            .disabled(self.identify_busy || !self.identify_editor.read(cx).ready())
                            .on_click(cx.listener(|this, _, window, cx| {
                                this.start_pipeline_job("identify", window, cx)
                            })),
                    )
                    .child(
                        Button::new("refresh-identity-review")
                            .outline()
                            .label("Refresh review")
                            .disabled(self.identify_busy)
                            .on_click(cx.listener(|this, _, _, cx| {
                                this.retranscribe_error = None;
                                this.load_identity_suggestions(cx);
                            })),
                    )
                    .child(
                        Button::new("close-identify")
                            .ghost()
                            .label(if self.identify_busy {
                                "Cancel"
                            } else {
                                "Close"
                            })
                            .on_click(cx.listener(|this, _, _, cx| {
                                if this.identify_busy {
                                    this.processing_cancel
                                        .store(true, std::sync::atomic::Ordering::SeqCst);
                                } else {
                                    this.identify_open = false;
                                    this.identity_request += 1;
                                }
                                cx.notify();
                            })),
                    ),
            )
            .into_any_element()
    }

    fn visible_transcript_turns(&self) -> Vec<format::TranscriptTurn> {
        transcript_search::matching_turns(
            speaker_filter::turns(&self.transcripts, &self.selected_speakers), &self.transcript_query)
    }

    fn refresh_speaker_filter(&mut self) {
        let available: std::collections::BTreeSet<_> = self.transcripts.iter().map(speaker_filter::key).collect();
        self.selected_speakers.retain(|key| available.contains(key));
        self.transcript_list_state.reset(self.visible_transcript_turns().len());
    }

    fn toggle_speaker_filter(&mut self, key: Option<speaker_filter::SpeakerKey>, cx: &mut Context<Self>) {
        if let Some(key) = key {
            if !self.selected_speakers.remove(&key) { self.selected_speakers.insert(key); }
        } else {
            self.selected_speakers.clear();
        }
        self.speaker_edit = None;
        self.refresh_speaker_filter();
        self.transcript_list_state.scroll_to(ListOffset { item_ix: 0, offset_in_item: px(0.) });
        cx.notify();
    }

    fn render_speaker_filter(&self, cx: &mut Context<Self>) -> AnyElement {
        let options = speaker_filter::options(&self.transcripts);
        let visible: usize = self.visible_transcript_turns().iter().map(|turn| turn.rows.len()).sum();
        v_flex().w_full().gap_2().px_4().py_2().border_b_1().border_color(cx.theme().border)
            .child(Input::new(&self.transcript_search_input).cleanable(true))
            .child(h_flex().gap_2().items_center()
                .child(div().text_xs().child("Filter by speaker"))
                .child(div().text_xs().text_color(cx.theme().muted_foreground)
                    .child(format!("{visible} of {} segments", self.transcripts.len()))))
            .child(h_flex().id("meeting-speaker-filters").gap_1().flex_wrap().max_h(px(112.)).overflow_y_scrollbar()
                .child(Button::new("all-transcript-speakers").xsmall()
                    .when(self.selected_speakers.is_empty(), |b| b.primary())
                    .when(!self.selected_speakers.is_empty(), |b| b.outline())
                    .label("All speakers")
                    .on_click(cx.listener(|this, _, _, cx| this.toggle_speaker_filter(None, cx))))
                .children(options.into_iter().enumerate().map(|(index, option)| {
                    let active = self.selected_speakers.contains(&option.key);
                    Button::new(SharedString::from(format!("filter-speaker-{index}"))).xsmall()
                        .when(active, |b| b.primary()).when(!active, |b| b.outline())
                        .label(format!("{} ({})", option.name, option.segments))
                        .tooltip("Select one or more speakers; click again to deselect")
                        .on_click(cx.listener(move |this, _, _, cx| this.toggle_speaker_filter(Some(option.key.clone()), cx)))
                })))
            .into_any_element()
    }

    fn render_transcript(&self, cx: &mut Context<Self>) -> AnyElement {
        let header = h_flex()
            .w_full()
            .items_center()
            .justify_between()
            .gap_2()
            .px_4()
            .py_2()
            .border_b_1()
            .border_color(cx.theme().border)
            .child(crate::ui::section_label("Transcript", cx))
            .child(
                Button::new("copy-transcript")
                    .ghost()
                    .icon(IconName::Copy)
                    .tooltip("Copy full transcript")
                    .disabled(self.transcripts.is_empty())
                    .on_click(cx.listener(|this, _, window, cx| this.copy_transcript(window, cx))),
            );

        if self.transcripts_loading {
            return v_flex()
                .size_full()
                .child(header)
                .child(
                    h_flex()
                        .gap_2()
                        .items_center()
                        .px_4()
                        .py_4()
                        .text_sm()
                        .text_color(cx.theme().muted_foreground)
                        .child(crate::ui::pulse_dot(cx.theme().primary, 6., "transcript-loading"))
                        .child("Loading transcript…"),
                )
                .into_any_element();
        }
        if let Some(err) = &self.load_error {
            return v_flex()
                .size_full()
                .child(header)
                .child(div().p_4().text_color(cx.theme().danger).child(err.clone()))
                .into_any_element();
        }
        if self.transcripts.is_empty() {
            return v_flex()
                .size_full()
                .child(header)
                .child(crate::ui::empty_state(
                    Lucide::AudioLines,
                    "No transcript yet",
                    "If audio was saved, choose Process recording above to set the language and speaker count, then create the transcript.",
                    cx,
                ))
                .into_any_element();
        }

        let transcripts = self.transcripts.clone();
        let turns = self.visible_transcript_turns();
        let search_query = self.transcript_query.clone();
        if turns.is_empty() {
            return v_flex().size_full().child(header).child(self.moments_panel.clone())
                .child(self.render_speaker_filter(cx))
                .child(div().p_4().text_sm().text_color(cx.theme().muted_foreground)
                    .child("No matching text. Clear the search or choose different speakers."))
                .into_any_element();
        }
        let list_state = self.transcript_list_state.clone();
        let entity = cx.entity();
        let meeting_id = self.meeting_id.clone();
        let speaker_edit = self.speaker_edit.clone();
        let playing_segment = self.playing_segment.clone();
        let loading_segment = self.loading_segment.clone();
        let show_confidence = self.show_confidence;
        let highlighted_segment = self.highlighted_segment.clone();
        let markers = self.moments_panel.read(cx).transcript_markers(&self.transcripts);

        v_flex()
            .size_full()
            .child(header)
            .child(self.moments_panel.clone())
            .child(self.render_speaker_filter(cx))
            .child(
                div().relative().flex_1().min_h_0().size_full()
                    .child(list(list_state.clone(), move |ix, _window, cx| {
                    let turn = &turns[ix];
                    let t = &transcripts[turn.rows.start];
                    let members = &transcripts[turn.rows.clone()];
                    let time = format::segment_timestamp(t.audio_start_time, &t.timestamp);
                    let speaker = t.speaker.clone().unwrap_or_else(|| "Speaker".to_string());
                    let editable = speaker_chip::can_edit_speaker(&speaker, t.voice_profile_id.as_deref());
                    let row_key = format!("{}:{speaker}", t.id);
                    let is_editing_this_row = speaker_edit.as_ref().is_some_and(|s| {
                        members.iter().any(|member| member.id == s.segment_id)
                    });
                    let turn_end = members.iter().filter_map(|member| member.audio_end_time)
                        .filter(|end| end.is_finite()).reduce(f64::max);
                    let can_play = meeting_id.is_some()
                        && turn_end.is_some_and(|end| end > t.audio_start_time.unwrap_or(0.0));
                    let is_playing = playing_segment.as_deref() == Some(t.id.as_str());
                    let is_loading_clip = loading_segment.as_deref() == Some(t.id.as_str());

                    let mut header_row = h_flex().gap_2().items_center().text_xs().text_color(cx.theme().muted_foreground);

                    if can_play {
                        let segment_id = t.id.clone();
                        let start = t.audio_start_time.unwrap_or(0.0);
                        let end = turn_end.unwrap_or(start);
                        let source = t.source.clone();
                        let entity = entity.clone();
                        header_row = header_row.child(
                            Button::new(SharedString::from(format!("play-segment-{}", t.id)))
                                .ghost()
                                .xsmall()
                                .icon(if is_playing { Lucide::Square } else { Lucide::Play })
                                .loading(is_loading_clip)
                                .tooltip(if is_playing { "Stop" } else { "Play this segment" })
                                .on_click(move |_, _, cx| {
                                    entity.update(cx, |this, cx| {
                                        this.toggle_segment_playback(
                                            segment_id.clone(),
                                            start,
                                            end,
                                            source.clone(),
                                            cx,
                                        )
                                    });
                                }),
                        );
                    }

                    if members.iter().any(|member| !member.overlapping_speakers.is_empty()) {
                        header_row = header_row.child(
                            div()
                                .text_xs()
                                .text_color(cx.theme().warning)
                                .child("Overlapping speech"),
                        );
                    }
                    if editable {
                        let segment_id = t.id.clone();
                        let speaker_for_click = speaker.clone();
                        let voice_profile_id = t.voice_profile_id.clone();
                        let entity = entity.clone();
                        header_row = header_row.child(
                            Button::new(SharedString::from(format!("edit-speaker-{row_key}")))
                                .ghost()
                                .xsmall()
                                .label(speaker.clone())
                                .on_click(move |ev, window, cx| {
                                    let click_position = ev.position();
                                    entity.update(cx, |this, cx| {
                                        this.toggle_speaker_edit(
                                            segment_id.clone(),
                                            speaker_for_click.clone(),
                                            voice_profile_id.clone(),
                                            click_position,
                                            window,
                                            cx,
                                        )
                                    });
                                }),
                        );
                    } else {
                        header_row = header_row.child(speaker.clone());
                    }
                    header_row = header_row.child(time);

                    if show_confidence {
                        if let Some(conf) = t.confidence {
                            let color = match format::ConfidenceLevel::for_confidence(conf) {
                                format::ConfidenceLevel::High => cx.theme().success,
                                format::ConfidenceLevel::Good | format::ConfidenceLevel::Medium => {
                                    cx.theme().warning
                                }
                                format::ConfidenceLevel::Low => cx.theme().danger,
                            };
                            header_row = header_row.child(
                                Button::new(SharedString::from(format!("confidence-{}", t.id)))
                                    .ghost()
                                    .xsmall()
                                    .tooltip(format::confidence_tooltip(conf))
                                    .child(div().size(px(8.)).rounded_full().bg(color)),
                            );
                        }
                    }

                    // Inline labels retain the original segment identity for
                    // speaker editing; grouping is a view, never a DB rewrite.
                    let links: Vec<_> = turn.speaker_links.iter().filter_map(|(range, index)| {
                        let member = &transcripts[*index];
                        let who = member.speaker.clone()?;
                        speaker_chip::can_edit_speaker(&who, member.voice_profile_id.as_deref())
                            .then(|| (range.clone(), member.id.clone(), who, member.voice_profile_id.clone()))
                    }).collect();
                    let ranges = links.iter().map(|link| link.0.clone()).collect();
                    let matches = transcript_search::highlights(&turn.body, &search_query);
                    let link_ranges: Vec<_> = links.iter().map(|link| link.0.clone()).collect();
                    let highlights = transcript_search::styled_ranges(&matches, &link_ranges).into_iter()
                        .map(|(range, matched, linked)| (range, HighlightStyle {
                            color: linked.then_some(cx.theme().primary),
                            background_color: matched.then_some(cx.theme().warning.opacity(0.3)),
                            ..Default::default()
                        }));
                    let text = StyledText::new(turn.body.clone()).with_highlights(highlights);
                    let text_entity = entity.clone();
                    let body = InteractiveText::new(
                        SharedString::from(format!("turn-body-{}", t.id)), text,
                    ).on_click(ranges, move |link_index, window, cx| {
                        let (_, id, who, profile) = &links[link_index];
                        text_entity.update(cx, |this, cx| {
                            this.toggle_speaker_edit(id.clone(), who.clone(), profile.clone(),
                                window.mouse_position(), window, cx);
                        });
                    });
                    let mut row = v_flex()
                        .w_full()
                        .gap_1()
                        .px_4()
                        .py_2()
                        .when(is_editing_this_row, |this| this.bg(cx.theme().muted.opacity(0.3)))
                        .when(members.iter().any(|member| highlighted_segment.as_deref() == Some(member.id.as_str())), |this| {
                            this.bg(cx.theme().primary.opacity(0.15))
                        });
                    for member in members {
                        if let Some(row_markers) = markers.get(&member.id) {
                            row = row.child(moments::render_row_markers(row_markers, cx));
                        }
                    }
                    let row = row.child(header_row).child(div().text_sm().child(body));

                    row.into_any_element()
                })
                .flex_1()
                .size_full())
                    .vertical_scrollbar(&list_state),
            )
            .into_any_element()
    }

    fn render_summary(&self, cx: &mut Context<Self>) -> impl IntoElement {
        let (status_label, button_label, button_disabled) = match &self.summary_phase {
            SummaryPhase::Idle => (None, "Generate summary", false),
            SummaryPhase::Loading => (Some("Loading summary…".to_string()), "Generate summary", true),
            SummaryPhase::Generating => (Some("Generating…".to_string()), "Generating…", true),
            SummaryPhase::Error(e) => (Some(e.clone()), "Regenerate summary", false),
        };
        let has_summary = self.summary_has_content;
        let template_label = self
            .templates
            .iter()
            .find(|(id, _, _)| *id == self.selected_template)
            .map(|(_, name, _)| name.clone())
            .unwrap_or_else(|| "Template".to_string());
        let template_tooltip = self
            .templates
            .iter()
            .find(|(id, _, _)| *id == self.selected_template)
            .map(|(_, _, desc)| desc.clone())
            .unwrap_or_default();

        v_flex()
            .size_full()
            .child(
                // Wraps rather than overflowing: next to the transcript, the
                // summary pane is often too narrow for the title and all
                // five actions on one line, and the primary button was the
                // one pushed out of view.
                h_flex()
                    .w_full()
                    .flex_wrap()
                    .items_center()
                    .justify_between()
                    .gap_2()
                    .px_4()
                    .py_3()
                    .border_b_1()
                    .border_color(cx.theme().border)
                    .child(
                        v_flex()
                            .gap_1()
                            .child(crate::ui::section_label("Summary", cx))
                            .when_some(status_label, |this, label| {
                                let generating = matches!(self.summary_phase, SummaryPhase::Generating);
                                this.child(
                                    h_flex()
                                        .gap_2()
                                        .items_center()
                                        .text_xs()
                                        .text_color(cx.theme().muted_foreground)
                                        .when(generating, |this| {
                                            this.child(crate::ui::pulse_dot(cx.theme().primary, 6., "summary-generating"))
                                        })
                                        .child(label),
                                )
                            }),
                    )
                    .child(
                        h_flex()
                            .flex_wrap()
                            .gap_2()
                            .items_center()
                            .when(!self.editing_summary, |this| {
                                this.child(
                                    Button::new("copy-summary")
                                        .ghost()
                                        .icon(IconName::Copy)
                                        .tooltip("Copy summary")
                                        .disabled(!has_summary)
                                        .on_click(cx.listener(|this, _, window, cx| this.copy_summary(window, cx))),
                                )
                                .child(
                                    Button::new("draft-followup-email")
                                        .ghost()
                                        .icon(Lucide::Mail)
                                        .tooltip("Draft follow-up email")
                                        .disabled(!has_summary && self.transcripts.is_empty())
                                        .on_click(cx.listener(|this, _, window, cx| {
                                            if let Some(id) = this.meeting_id.clone() {
                                                followup::open(id, window, cx);
                                            }
                                        })),
                                )
                                .child(
                                    Button::new("edit-summary")
                                        .ghost()
                                        .icon(Lucide::Pencil)
                                        .tooltip("Edit summary")
                                        .disabled(!has_summary || matches!(self.summary_phase, SummaryPhase::Generating))
                                        .on_click(cx.listener(|this, _, _, cx| this.start_edit_summary(cx))),
                                )
                                .child(
                                    Button::new("cycle-summary-template")
                                        .outline()
                                        .icon(Lucide::FileText)
                                        .label(template_label)
                                        .tooltip(if template_tooltip.is_empty() {
                                            "Summary template".to_string()
                                        } else {
                                            template_tooltip
                                        })
                                        .disabled(self.templates.is_empty() || matches!(self.summary_phase, SummaryPhase::Generating))
                                        .on_click(cx.listener(|this, _, _, cx| this.cycle_template(cx))),
                                )
                                .child(
                                    Button::new("toggle-custom-prompt")
                                        .ghost()
                                        .icon(Lucide::MessageSquare)
                                        .tooltip("Custom instructions")
                                        .disabled(matches!(self.summary_phase, SummaryPhase::Generating))
                                        .on_click(cx.listener(|this, _, _, cx| this.toggle_custom_prompt(cx))),
                                )
                                .child(
                                    Button::new("generate-summary")
                                        .primary()
                                        .label(if has_summary && button_label == "Generate summary" {
                                            "Regenerate summary"
                                        } else {
                                            button_label
                                        })
                                        .loading(matches!(self.summary_phase, SummaryPhase::Generating))
                                        .disabled(button_disabled)
                                        .on_click(cx.listener(|this, _, window, cx| this.generate_summary(window, cx))),
                                )
                            })
                            .when(self.editing_summary, |this| {
                                this.child(
                                    Button::new("cancel-edit-summary")
                                        .ghost()
                                        .label("Cancel")
                                        .disabled(self.saving_summary)
                                        .on_click(cx.listener(|this, _, _, cx| this.cancel_edit_summary(cx))),
                                )
                                .child(
                                    Button::new("save-summary")
                                        .primary()
                                        .icon(Lucide::Save)
                                        .label(if self.saving_summary { "Saving…" } else { "Save" })
                                        .loading(self.saving_summary)
                                        .disabled(self.saving_summary)
                                        .on_click(cx.listener(|this, _, window, cx| this.save_summary(window, cx))),
                                )
                            }),
                    ),
            )
            .when(self.show_custom_prompt && !self.editing_summary, |this| {
                this.child(
                    h_flex()
                        .w_full()
                        .gap_2()
                        .items_center()
                        .px_4()
                        .py_2()
                        .border_b_1()
                        .border_color(cx.theme().border)
                        .child(div().flex_1().min_w_0().child(Input::new(&self.custom_prompt)))
                        .into_any_element(),
                )
            })
            .child(
                div()
                    .id("summary-scroll")
                    .flex_1()
                    .min_h_0()
                    .overflow_y_scrollbar()
                    .p_4()
                    .child(
                        v_flex()
                            .w_full()
                            .gap_4()
                            .when(!self.editing_summary, |this| {
                                this.child(TextView::new(&self.summary_state).selectable(true))
                            })
                            .when(self.editing_summary, |this| this.child(self.summary_editor.clone()))
                            .child(self.render_action_items_panel(cx))
                            .child(self.render_notes_panel(cx)),
                    ),
            )
    }

    // ---- Per-meeting action items panel -------------------------------------
    // Mirrors `ActionItemsPanel.tsx`'s card layout: header (icon, title, open
    // count / "All done"), an "Extract from summary"/"Re-extract" button
    // (only once there's a summary), the item list, then an add row.

    fn render_action_items_panel(&self, cx: &mut Context<Self>) -> AnyElement {
        let open_count = self.action_items.iter().filter(|i| i.status != STATUS_DONE).count();
        let is_empty = self.action_items.is_empty();
        let has_summary = self.summary_has_content;

        let mut card = v_flex()
            .w_full()
            .gap_2()
            .p_3()
            .rounded_md()
            .bg(cx.theme().muted.opacity(0.3))
            .child(
                h_flex()
                    .w_full()
                    .items_center()
                    .justify_between()
                    .gap_2()
                    .child(
                        h_flex()
                            .gap_2()
                            .items_center()
                            .child(Lucide::ListChecks.view(cx))
                            .child(div().text_sm().font_semibold().child("Action Items"))
                            .when(open_count > 0, |this| {
                                this.child(
                                    div()
                                        .text_xs()
                                        .px_2()
                                        .rounded_full()
                                        .bg(cx.theme().primary.opacity(0.15))
                                        .text_color(cx.theme().primary)
                                        .child(open_count.to_string()),
                                )
                            })
                            .when(!is_empty && open_count == 0, |this| {
                                this.child(
                                    div().text_xs().text_color(cx.theme().muted_foreground).child("All done"),
                                )
                            }),
                    )
                    .when(has_summary, |this| {
                        let label = if is_empty { "Extract from summary" } else { "Re-extract" };
                        this.child(
                            Button::new("extract-action-items")
                                .ghost()
                                .xsmall()
                                .icon(Lucide::Sparkles)
                                .label(label)
                                .loading(self.ai_extracting)
                                .disabled(self.ai_extracting)
                                .on_click(cx.listener(|this, _, _, cx| this.extract_action_items(cx))),
                        )
                    }),
            );

        if is_empty {
            let empty_text = if has_summary {
                "No action items yet. Extract them from the summary, or add one below."
            } else {
                "No action items yet. Add one below."
            };
            card = card.child(div().text_xs().text_color(cx.theme().muted_foreground).child(empty_text));
        } else {
            for item in &self.action_items {
                card = card.child(self.render_action_item_row(item, cx));
            }
        }

        card = card.child(self.render_action_item_composer(cx));
        card.into_any_element()
    }

    fn render_action_item_row(&self, item: &ActionItem, cx: &mut Context<Self>) -> AnyElement {
        let is_done = item.status == STATUS_DONE;
        let id = item.id.clone();

        if self.ai_editing_id.as_deref() == Some(item.id.as_str()) {
            return h_flex()
                .w_full()
                .items_center()
                .gap_2()
                .child(div().flex_1().min_w_0().child(Input::new(&self.ai_edit_input)))
                .child(
                    Button::new(SharedString::from(format!("save-ai-{id}")))
                        .primary()
                        .xsmall()
                        .label("Save")
                        .on_click(cx.listener(|this, _, _, cx| this.save_action_item_edit(cx))),
                )
                .child(
                    Button::new(SharedString::from(format!("cancel-ai-{id}")))
                        .ghost()
                        .xsmall()
                        .label("Cancel")
                        .on_click(cx.listener(|this, _, _, cx| this.cancel_action_item_edit(cx))),
                )
                .into_any_element();
        }

        let id_for_toggle = id.clone();
        let id_for_edit = id.clone();
        let id_for_delete = id.clone();

        h_flex()
            .w_full()
            .items_start()
            .gap_2()
            .child(
                Checkbox::new(SharedString::from(format!("ai-done-{id}"))).checked(is_done).on_click(cx.listener(
                    move |this, _, _, cx| this.toggle_action_item_status(id_for_toggle.clone(), is_done, cx),
                )),
            )
            .child(
                v_flex()
                    .flex_1()
                    .min_w_0()
                    .gap_1()
                    .child(
                        div()
                            .text_sm()
                            .when(is_done, |this| this.line_through().text_color(cx.theme().muted_foreground))
                            .child(item.text.clone()),
                    )
                    .when(item.assignee.is_some() || item.due_hint.is_some(), |this| {
                        this.child(
                            h_flex()
                                .gap_2()
                                .text_xs()
                                .text_color(cx.theme().muted_foreground)
                                .when_some(item.assignee.clone(), |this, a| this.child(a))
                                .when_some(item.due_hint.clone(), |this, d| this.child(d)),
                        )
                    }),
            )
            .child(
                Button::new(SharedString::from(format!("edit-ai-{id}")))
                    .ghost()
                    .xsmall()
                    .icon(Lucide::Pencil)
                    .tooltip("Edit")
                    .on_click(cx.listener(move |this, _, window, cx| {
                        this.start_action_item_edit(id_for_edit.clone(), window, cx);
                    })),
            )
            .child(
                Button::new(SharedString::from(format!("delete-ai-{id}")))
                    .ghost()
                    .xsmall()
                    .icon(IconName::Delete)
                    .tooltip("Delete")
                    .on_click(cx.listener(move |this, _, _, cx| {
                        this.delete_action_item(id_for_delete.clone(), cx);
                    })),
            )
            .into_any_element()
    }

    fn render_action_item_composer(&self, cx: &mut Context<Self>) -> AnyElement {
        if self.ai_adding {
            return h_flex()
                .w_full()
                .items_center()
                .gap_2()
                .child(div().flex_1().min_w_0().child(Input::new(&self.ai_add_input)))
                .child(
                    Button::new("save-add-ai")
                        .primary()
                        .xsmall()
                        .label("Add")
                        .on_click(cx.listener(|this, _, _, cx| this.save_new_action_item(cx))),
                )
                .child(
                    Button::new("cancel-add-ai")
                        .ghost()
                        .xsmall()
                        .label("Cancel")
                        .on_click(cx.listener(|this, _, _, cx| this.cancel_action_item_composer(cx))),
                )
                .into_any_element();
        }

        Button::new("open-add-ai")
            .ghost()
            .xsmall()
            .icon(IconName::Plus)
            .label("Add item")
            .on_click(cx.listener(|this, _, window, cx| this.open_action_item_composer(window, cx)))
            .into_any_element()
    }

    // ---- Meeting notes panel -------------------------------------------------
    // Mirrors `MeetingNotesPanel.tsx`'s card layout: header (icon, title,
    // count badge, "Add note" button), a list of notes, no empty-state text.

    fn render_notes_panel(&self, cx: &mut Context<Self>) -> AnyElement {
        let count = self.notes.len();

        let mut card = v_flex()
            .w_full()
            .gap_2()
            .p_3()
            .rounded_md()
            .bg(cx.theme().muted.opacity(0.3))
            .child(
                h_flex()
                    .w_full()
                    .items_center()
                    .justify_between()
                    .gap_2()
                    .child(
                        h_flex()
                            .gap_2()
                            .items_center()
                            .child(Lucide::NotebookPen.view(cx))
                            .child(div().text_sm().font_semibold().child("Notes"))
                            .when(count > 0, |this| {
                                this.child(
                                    div()
                                        .text_xs()
                                        .px_2()
                                        .rounded_full()
                                        .bg(cx.theme().primary.opacity(0.15))
                                        .text_color(cx.theme().primary)
                                        .child(count.to_string()),
                                )
                            }),
                    )
                    .when(!self.notes_composing, |this| {
                        this.child(
                            Button::new("open-note-composer")
                                .ghost()
                                .xsmall()
                                .icon(IconName::Plus)
                                .label("Add note")
                                .on_click(cx.listener(|this, _, window, cx| this.open_note_composer(window, cx))),
                        )
                    }),
            );

        for note in &self.notes {
            let id = note.id.clone();
            let time = format::format_note_time(&note.created_at);
            let suffix = if note.source == "agent" { " · added by an agent" } else { "" };
            card = card.child(
                v_flex()
                    .w_full()
                    .gap_1()
                    .child(div().text_sm().child(note.body.clone()))
                    .child(
                        h_flex()
                            .items_center()
                            .justify_between()
                            .child(
                                div()
                                    .text_xs()
                                    .text_color(cx.theme().muted_foreground)
                                    .child(format!("{time}{suffix}")),
                            )
                            .child(
                                Button::new(SharedString::from(format!("delete-note-{id}")))
                                    .ghost()
                                    .xsmall()
                                    .icon(IconName::Delete)
                                    .tooltip("Delete note")
                                    .on_click(cx.listener(move |this, _, _, cx| this.delete_note(id.clone(), cx))),
                            ),
                    ),
            );
        }

        if self.notes_composing {
            card = card.child(
                v_flex()
                    .w_full()
                    .gap_2()
                    .child(div().w_full().child(Input::new(&self.notes_draft)))
                    .child(
                        h_flex()
                            .gap_2()
                            .child(
                                Button::new("save-note")
                                    .primary()
                                    .xsmall()
                                    .label(if self.notes_saving { "Saving…" } else { "Save" })
                                    .loading(self.notes_saving)
                                    .disabled(self.notes_saving)
                                    .on_click(cx.listener(|this, _, window, cx| this.save_note(window, cx))),
                            )
                            .child(
                                Button::new("cancel-note")
                                    .ghost()
                                    .xsmall()
                                    .label("Cancel")
                                    .disabled(self.notes_saving)
                                    .on_click(cx.listener(|this, _, _, cx| this.cancel_note_composer(cx))),
                            ),
                    ),
            );
        }

        card.into_any_element()
    }
}
