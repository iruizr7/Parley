//! Progress for speaker jobs, isolated by request so concurrent operations cannot
//! overwrite each other. Percentages and ETA describe the current subtask only.
use crate::{app_state::AppServices, views::import::timeline::Estimate};
use gpui_kit::{
    component::{progress::Progress, v_flex},
    *,
};
use std::{
    sync::atomic::{AtomicU64, Ordering},
    time::{Duration, Instant},
};

static NEXT_REQUEST: AtomicU64 = AtomicU64::new(1);

pub struct TaskProgress {
    request: u64,
    active: bool,
    started: Instant,
    message: String,
    percent: Option<u32>,
    estimate: Estimate,
}

impl TaskProgress {
    pub fn new(cx: &mut Context<Self>) -> Self {
        let events = AppServices::global(cx).core_events.clone();
        cx.subscribe(&events, |this, _, event, cx| {
            if !this.active
                || event.name != "speaker-task-progress"
                || event.payload["request"].as_u64() != Some(this.request)
            {
                return;
            }
            let stage = event.payload["stage"].as_str().unwrap_or_default();
            this.message = event.payload["message"].as_str().unwrap_or_default().into();
            this.percent = measured(stage)
                .then(|| event.payload["percent"].as_u64().unwrap_or(0).min(100) as u32);
            this.estimate
                .update(stage, &this.message, this.percent, this.started.elapsed());
            cx.notify();
        })
        .detach();
        Self {
            request: 0,
            active: false,
            started: Instant::now(),
            message: String::new(),
            percent: None,
            estimate: Estimate::default(),
        }
    }

    pub fn start(
        &mut self,
        cx: &mut Context<Self>,
    ) -> impl Fn(&str, u32, &str) + Send + Sync + use<> {
        self.request = NEXT_REQUEST.fetch_add(1, Ordering::Relaxed);
        self.active = true;
        self.started = Instant::now();
        self.message = "Preparing speaker processing…".into();
        self.percent = None;
        self.estimate = Estimate::default();
        let request = self.request;
        cx.spawn(async move |this, cx| {
            loop {
                cx.background_executor().timer(Duration::from_secs(1)).await;
                if !matches!(
                    this.update(cx, |this, cx| {
                        if !this.active || this.request != request {
                            return false;
                        }
                        cx.notify();
                        true
                    }),
                    Ok(true)
                ) {
                    break;
                }
            }
        })
        .detach();
        cx.notify();
        let sink = AppServices::global(cx).sink.clone();
        move |stage, percent, message| {
            let _ = sink.emit_value("speaker-task-progress", serde_json::json!({"request":request,"stage":stage,"percent":percent,"message":message}));
        }
    }

    pub fn finish(&mut self, cx: &mut Context<Self>) {
        self.active = false;
        cx.notify();
    }
}

fn measured(stage: &str) -> bool {
    matches!(
        stage,
        "audio_integrity"
            | "model_integrity"
            | "recognition"
            | "verifying"
            | "audio_conversion"
            | "audio_decoding"
            | "audio_resampling"
            | "vad_progress"
            | "worker_decoding_progress"
            | "diarizing_progress"
            | "embedding_progress"
            | "uploading"
    )
}

fn duration(value: Duration) -> String {
    format!("{}:{:02}", value.as_secs() / 60, value.as_secs() % 60)
}

impl Render for TaskProgress {
    fn render(&mut self, _window: &mut Window, _cx: &mut Context<Self>) -> impl IntoElement {
        let now = self.started.elapsed();
        let status = match self.percent {
            Some(100) => "100% of reported work · Finishing this step…".into(),
            Some(percent) => format!(
                "{percent}% of this step · {}",
                self.estimate
                    .remaining(now)
                    .map(|remaining| format!(
                        "about {} remaining in this step",
                        duration(remaining)
                    ))
                    .unwrap_or_else(|| "Estimating remaining time…".into())
            ),
            None => "Working · this step does not report a measurable total yet".into(),
        };
        let mut body = v_flex().w_full().gap_2().child(self.message.clone()).child(
            div()
                .text_xs()
                .child(format!("Elapsed {} · {status}", duration(now))),
        );
        if let Some(percent) = self.percent {
            body = body.child(Progress::new("speaker-task-subtask").value(percent as f32));
        }
        body
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn only_measured_subtasks_display_percentages() {
        for stage in [
            "uploading",
            "diarizing_progress",
            "embedding_progress",
            "audio_conversion",
        ] {
            assert!(super::measured(stage));
        }
        for stage in [
            "runtime",
            "diarizing",
            "embedding",
            "upload_verifying",
            "saving",
        ] {
            assert!(!super::measured(stage));
        }
    }
}
