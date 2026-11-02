//! Immutable processing runs, with technical provenance tucked behind Details.
use gpui_kit::component::scroll::ScrollableElement as _;

use crate::{app_state::AppServices, runtime::Io};
use gpui_kit::base::StyledExt as _;
use gpui_kit::component::{
    button::{Button, ButtonVariants as _},
    h_flex,
    notification::Notification,
    v_flex, ActiveTheme, Disableable as _, WindowExt as _,
};
use gpui_kit::prelude::FluentBuilder as _;
use gpui_kit::*;
use parley_core::audio::processing::runs::{self, ProcessingRun};

fn stored_target_label(resolved: &serde_json::Value) -> &str {
    [
        "/target/name",
        "/target_snapshot/name",
        "/extraction_provenance/target/name",
        "/extraction_provenance/target_snapshot/name",
    ]
    .into_iter()
    .find_map(|path| resolved.pointer(path).and_then(|name| name.as_str()))
    .filter(|name| !name.trim().is_empty())
    .unwrap_or_else(|| {
        if resolved["target"] == "local" {
            "Local"
        } else if resolved["extracted"] == false {
            "Saved speaker evidence"
        } else {
            "Unknown target"
        }
    })
}

fn reconnectable_request(requested: &serde_json::Value) -> bool {
    requested["reconnectable"] == true
        && requested["request_id"].is_string()
        && requested["options"].is_object()
}

fn display_metadata(run: &ProcessingRun) -> serde_json::Value {
    serde_json::json!({
        "requested": runs::redact_metadata(&run.requested),
        "resolved": runs::redact_metadata(&run.resolved),
        "input": runs::redact_metadata(&run.input),
    })
}

pub struct ProcessingResultChanged;
impl EventEmitter<ProcessingResultChanged> for ProcessingHistory {}
pub struct ProcessingHistory {
    meeting: Option<String>,
    runs: Vec<ProcessingRun>,
    loading: bool,
    generation: u64,
    error: Option<String>,
    expanded: Option<String>,
    reconnecting: Option<String>,
    cancelled: std::sync::Arc<std::sync::atomic::AtomicBool>,
}
impl ProcessingHistory {
    pub fn new() -> Self {
        Self {
            meeting: None,
            runs: vec![],
            loading: false,
            generation: 0,
            error: None,
            expanded: None,
            reconnecting: None,
            cancelled: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
        }
    }
    pub fn load(&mut self, meeting: String, cx: &mut Context<Self>) {
        self.meeting = Some(meeting.clone());
        self.runs.clear();
        self.expanded = None;
        self.refresh(cx);
    }
    fn reconnect(&mut self, run: String, cx: &mut Context<Self>) {
        if self.reconnecting.is_some() {
            return;
        }
        let Some(meeting) = self.meeting.clone() else {
            return;
        };
        let Some(pool) = AppServices::global(cx).pool() else {
            return;
        };
        let io = Io::global(cx);
        let sink = AppServices::global(cx).sink.clone();
        self.reconnecting = Some(run.clone());
        self.error = None;
        self.cancelled = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let cancelled = self.cancelled.clone();
        cx.notify();
        cx.spawn(async move|this,cx|{
            let requested_meeting=meeting.clone();
            let result=io.spawn(async move {parley_core::audio::processing::service::resume_processing(&pool,&meeting,&run,&cancelled,|stage,percent,message|{let _=sink.emit_value("retranscription-progress",serde_json::json!({"meeting_id":meeting,"stage":stage,"progress_percentage":percent,"message":message}));}).await}).await;
            let _=this.update(cx,|this,cx|{this.reconnecting=None;if this.meeting.as_deref()!=Some(&requested_meeting){return;}match result {Ok(Ok(outcome))=>{this.refresh(cx);cx.emit(ProcessingResultChanged);crate::shell::refresh_meetings(cx);if let Some(warning)=outcome.identification_error{crate::ui::with_main_window(cx,|window,cx|window.push_notification(Notification::warning(format!("Transcript saved. Speaker identification needs attention: {warning}")),cx));}},Ok(Err(error))=>this.error=Some(error.to_string()),Err(error)=>this.error=Some(error.to_string())}cx.notify();});
        }).detach();
    }
    pub fn refresh(&mut self, cx: &mut Context<Self>) {
        let Some(meeting) = self.meeting.clone() else {
            return;
        };
        let Some(pool) = AppServices::global(cx).pool() else {
            return;
        };
        let io = Io::global(cx);
        self.loading = true;
        self.generation += 1;
        let generation = self.generation;
        self.error = None;
        cx.spawn(async move |this, cx| {
            let result = io
                .spawn(async move {
                    runs::list_runs(&pool, &meeting)
                        .await
                        .map(|runs| (meeting, runs))
                })
                .await;
            let _ = this.update(cx, |this, cx| {
                if this.generation != generation {
                    return;
                }
                this.loading = false;
                match result {
                    Ok(Ok((meeting, runs))) if this.meeting.as_deref() == Some(&meeting) => {
                        this.runs = runs
                    }
                    Ok(Ok(_)) => {}
                    Ok(Err(e)) => this.error = Some(e.to_string()),
                    Err(e) => this.error = Some(e.to_string()),
                }
                cx.notify();
            });
        })
        .detach();
    }
}
impl Render for ProcessingHistory {
    fn render(&mut self, _: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let mut body = v_flex()
            .id("processing-history-scroll")
            .max_h(px(360.))
            .overflow_y_scrollbar()
            .w_full()
            .gap_2()
            .p_4()
            .border_b_1()
            .border_color(cx.theme().border)
            .child(
                h_flex()
                    .w_full()
                    .justify_between()
                    .items_center()
                    .child(div().text_sm().font_semibold().child("Processing history"))
                    .child(
                        Button::new("refresh-processing-history")
                            .ghost()
                            .label("Refresh")
                            .disabled(self.loading)
                            .on_click(cx.listener(|this, _, _, cx| {
                                if !this.loading {
                                    this.refresh(cx);
                                }
                            })),
                    ),
            );
        if self.loading {
            body = body.child(div().text_xs().child("Loading processing details…"));
        }
        if self.runs.is_empty() && !self.loading {
            body = body.child(
                div()
                    .text_xs()
                    .text_color(cx.theme().muted_foreground)
                    .child(
                        "No recorded processing details. Older results have unknown provenance.",
                    ),
            );
        }
        if let Some(e) = &self.error {
            body = body.child(
                div()
                    .text_xs()
                    .text_color(cx.theme().danger)
                    .child(e.clone()),
            );
        }
        for run in &self.runs {
            let id = run.id.clone();
            let target = stored_target_label(&run.resolved);
            let operation = match run.operation.as_str() {
                "process" => "Transcription",
                "speaker_analysis" => "Speaker detection",
                "identification" => "Speaker identification",
                "recording_only" => "Recording only",
                "live_preview" => "Live preview",
                _ => &run.operation,
            };
            let title = format!(
                "{operation} · {} · {target} · {}",
                run.outcome, run.started_at
            );
            body = body.child(
                Button::new(format!("run-{id}"))
                    .ghost()
                    .label(title)
                    .on_click(cx.listener(move |this, _, _, cx| {
                        this.expanded = if this.expanded.as_deref() == Some(&id) {
                            None
                        } else {
                            Some(id.clone())
                        };
                        cx.notify();
                    })),
            );
            if run.outcome == "failed" && reconnectable_request(&run.requested) {
                let reconnect = run.id.clone();
                body =
                    body.child(
                        Button::new(format!("reconnect-run-{}", run.id))
                            .outline()
                            .label(if self.reconnecting.as_deref() == Some(&run.id) {
                                "Reconnecting…"
                            } else {
                                "Reconnect to worker job"
                            })
                            .disabled(self.reconnecting.is_some())
                            .on_click(cx.listener(move |this, _, _, cx| {
                                this.reconnect(reconnect.clone(), cx)
                            })),
                    );
            }
            if self.expanded.as_deref() == Some(&run.id) {
                let metadata = display_metadata(run);
                body = body
                    .child(
                        div()
                            .text_xs()
                            .text_color(cx.theme().muted_foreground)
                            .child(serde_json::to_string_pretty(&metadata).unwrap_or_default()),
                    )
                    .when_some(run.error.clone(), |b, error| {
                        b.child(div().text_xs().text_color(cx.theme().danger).child(error))
                    });
            }
        }
        if self.reconnecting.is_some() {
            body = body.child(
                Button::new("cancel-reconnect")
                    .ghost()
                    .label("Cancel reconnect")
                    .on_click(cx.listener(|this, _, _, cx| {
                        this.cancelled
                            .store(true, std::sync::atomic::Ordering::SeqCst);
                        cx.notify();
                    })),
            );
        }
        body
    }
}

impl Drop for ProcessingHistory {
    fn drop(&mut self) {
        self.cancelled
            .store(true, std::sync::atomic::Ordering::SeqCst);
    }
}

#[cfg(test)]
mod tests {
    use super::{display_metadata, reconnectable_request, stored_target_label};
    use parley_core::audio::processing::runs::ProcessingRun;
    use serde_json::json;

    #[test]
    fn processing_history_uses_snapshot_after_target_rename_or_deletion() {
        // This is the shape written by runs::store_tx, independent of the
        // current targets table, which may have been renamed or deleted.
        let resolved = json!({"target":{"id":"fixture-target","name":"Original target","kind":"ssh"},"runtime":{},"worker_settings":{"target_snapshot":{"id":"fixture-target","name":"Original target"}}});
        assert_eq!(stored_target_label(&resolved), "Original target");
        assert_eq!(
            stored_target_label(&json!({"target_snapshot":{"name":"Legacy snapshot"}})),
            "Legacy snapshot"
        );
    }

    #[test]
    fn identification_history_distinguishes_extraction_from_cached_evidence() {
        let extracted = json!({"extracted":true,"extraction_provenance":{"target":{"id":"fixture-target","name":"Extraction target","kind":"ssh"},"request_id":"fixture-request"}});
        assert_eq!(stored_target_label(&extracted), "Extraction target");
        assert_eq!(
            stored_target_label(&json!({"extracted":false,"extraction_provenance":null})),
            "Saved speaker evidence"
        );
        assert_eq!(stored_target_label(&json!({})), "Unknown target");
        assert_eq!(
            stored_target_label(&json!({"enabled":false,"target":"local"})),
            "Local"
        );
    }

    #[test]
    fn reconnect_requires_an_acknowledged_worker_job() {
        let mut request = json!({"request_id":"fixture-request","options":{"config":{}}});
        assert!(!reconnectable_request(&request));
        request["reconnectable"] = false.into();
        assert!(!reconnectable_request(&request));
        request["reconnectable"] = true.into();
        assert!(reconnectable_request(&request));
        request["request_id"] = serde_json::Value::Null;
        assert!(!reconnectable_request(&request));
    }

    #[test]
    fn failed_run_details_keep_requested_language_and_speaker_count() {
        let run = ProcessingRun {
            id: "fixture-failed-run".into(),
            meeting_id: Some("fixture-meeting".into()),
            operation: "process".into(),
            started_at: "2026-01-01T12:00:00Z".into(),
            finished_at: "2026-01-01T12:00:01Z".into(),
            outcome: "failed".into(),
            requested: json!({
                "request_id": "fixture-request",
                "reconnectable": false,
                "options": {
                    "config": {"target_id": "fixture-target"},
                    "language": "es",
                    "num_speakers": 2,
                },
                "target": {"worker_path": "/example/worker"},
            }),
            resolved: json!({}),
            // Preflight failures have no resolved input metadata to fall back to.
            input: json!({}),
            error: Some("Selected model is missing".into()),
        };
        let metadata = display_metadata(&run);
        assert_eq!(metadata["requested"]["options"]["language"], "es");
        assert_eq!(metadata["requested"]["options"]["num_speakers"], 2);
        assert_eq!(metadata["requested"]["request_id"], "fixture-request");
        assert_eq!(metadata["requested"]["reconnectable"], false);
        assert!(metadata["requested"]["target"].get("worker_path").is_none());
    }
}
