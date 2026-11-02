//! Per-model enrollment evidence. Playback always reads confirmed source audio.
use crate::{app_state::AppServices, runtime::Io};
use gpui_kit::base::StyledExt as _;
use gpui_kit::component::{
    button::{Button, ButtonVariants as _},
    h_flex,
    menu::{DropdownMenu as _, PopupMenuItem},
    v_flex, ActiveTheme, Disableable as _,
};
use gpui_kit::prelude::FluentBuilder as _;
use gpui_kit::*;
use parley_core::audio::processing::{
    config::{self, Identification, ProcessingTarget},
    enrollment::{self, Enrollment, EnrollmentSample},
};

#[derive(Clone)]
struct SourceMeeting {
    id: String,
    title: String,
    speaker: String,
    group_version: Option<String>,
}
pub struct EnrollmentDetails {
    profile_id: String,
    progress: Entity<crate::views::task_progress::TaskProgress>,
    enrollments: Vec<Enrollment>,
    samples: Vec<EnrollmentSample>,
    sources: Vec<SourceMeeting>,
    targets: Vec<ProcessingTarget>,
    target_id: String,
    model: Identification,
    loading: bool,
    generation: u64,
    busy: bool,
    enrolling: bool,
    error: Option<String>,
    message: Option<String>,
    expanded_space: Option<String>,
    playing: Option<String>,
    loading_playback: bool,
    playback_generation: std::sync::Arc<std::sync::atomic::AtomicU64>,
    cancelled: std::sync::Arc<std::sync::atomic::AtomicBool>,
}
impl EnrollmentDetails {
    pub fn new(profile_id: String, cx: &mut Context<Self>) -> Self {
        let mut this = Self {
            profile_id,
            progress: cx.new(crate::views::task_progress::TaskProgress::new),
            enrollments: vec![],
            samples: vec![],
            sources: vec![],
            targets: vec![],
            target_id: String::new(),
            model: Identification::Campplus,
            loading: true,
            generation: 0,
            busy: false,
            enrolling: false,
            error: None,
            message: None,
            expanded_space: None,
            playing: None,
            loading_playback: false,
            playback_generation: std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0)),
            cancelled: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
        };
        let events = AppServices::global(cx).core_events.clone();
        cx.subscribe(&events, |this, _, event, cx| {
            if event.name == parley_core::audio::playback::PLAYBACK_ENDED_EVENT
                && !this.loading_playback
            {
                this.playing = None;
                cx.notify();
            }
        })
        .detach();
        this.refresh(cx);
        this
    }
    pub(super) fn refresh(&mut self, cx: &mut Context<Self>) {
        if self.busy { return; }
        let Some(pool) = AppServices::global(cx).pool() else {
            return;
        };
        let profile = self.profile_id.clone();
        let io = Io::global(cx);
        self.loading = true;
        self.generation += 1;
        let generation = self.generation;
        self.error = None;
        cx.spawn(async move |this, cx| {
            let result = io
                .spawn(async move {
                    enrollment::recover_confirmed_evidence(&pool, &profile).await?;
                    let enrollments = enrollment::list_enrollments(&pool, &profile).await?;
                    let mut samples = vec![];
                    for item in &enrollments {
                        samples.extend(
                            enrollment::list_samples(&pool, &profile, &item.space_id).await?,
                        );
                    }
                    let targets = config::list_targets(&pool).await?;
                    let defaults = config::load_config(&pool).await?;
                    let sources = enrollment::enrollment_sources(&pool, &profile)
                        .await?
                        .into_iter()
                        .map(|source| SourceMeeting {
                            id: source.meeting_id,
                            title: source.title,
                            speaker: source.speaker,
                            group_version: source.group_version,
                        })
                        .collect();
                    Ok::<_, anyhow::Error>((
                        enrollments,
                        samples,
                        sources,
                        targets,
                        defaults.target_id,
                        match defaults.identification {
                            Identification::Campplus => Identification::Campplus,
                            Identification::Wespeaker => Identification::Wespeaker,
                            _ => match defaults.speaker_pipeline {
                                config::SpeakerPipeline::Sherpa => Identification::Campplus,
                                config::SpeakerPipeline::Community1 => Identification::Wespeaker,
                            },
                        },
                    ))
                })
                .await;
            let _ = this.update(cx, |this, cx| {
                if this.generation != generation {
                    return;
                }
                this.loading = false;
                match result {
                    Ok(Ok((enrollments, samples, sources, targets, target, model))) => {
                        this.enrollments = enrollments;
                        this.samples = samples;
                        this.sources = sources;
                        this.targets = targets;
                        if this.target_id.is_empty() {
                            this.target_id = target;
                            this.model = model;
                        }
                    }
                    Ok(Err(e)) => this.error = Some(e.to_string()),
                    Err(e) => this.error = Some(e.to_string()),
                }
                cx.notify();
            });
        })
        .detach();
    }
    fn stop_evidence_playback(&mut self) {
        self.playback_generation
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        self.playing = None;
        self.loading_playback = false;
        parley_core::audio::playback::stop();
    }
    fn remove_sample(&mut self, id: String, cx: &mut Context<Self>) {
        if self.busy {
            return;
        }
        self.stop_evidence_playback();
        self.generation += 1;
        self.loading = false;
        let Some(pool) = AppServices::global(cx).pool() else {
            return;
        };
        let io = Io::global(cx);
        self.busy = true;
        cx.notify();
        cx.spawn(async move |this, cx| {
            let result = io
                .spawn(async move { enrollment::remove_sample(&pool, &id).await.map(|_| ()) })
                .await;
            let _ = this.update(cx, |this, cx| {
                this.finish_action(
                    result
                        .map_err(|e| e.to_string())
                        .and_then(|r| r.map_err(|e| e.to_string())),
                    "Sample removed and enrollment rebuilt",
                    cx,
                )
            });
        })
        .detach();
    }
    fn enrollment_action(&mut self, space: String, delete: bool, cx: &mut Context<Self>) {
        if self.busy {
            return;
        }
        if delete {
            self.stop_evidence_playback();
        }
        self.generation += 1;
        self.loading = false;
        let Some(pool) = AppServices::global(cx).pool() else {
            return;
        };
        let profile = self.profile_id.clone();
        let io = Io::global(cx);
        self.busy = true;
        cx.notify();
        cx.spawn(async move |this, cx| {
            let result = io
                .spawn(async move {
                    if delete {
                        enrollment::delete_enrollment(&pool, &profile, &space)
                            .await
                            .map(|_| ())
                    } else {
                        enrollment::relearn(&pool, &profile, &space)
                            .await
                            .map(|_| ())
                    }
                })
                .await;
            let _ = this.update(cx, |this, cx| {
                this.finish_action(
                    result
                        .map_err(|e| e.to_string())
                        .and_then(|r| r.map_err(|e| e.to_string())),
                    if delete {
                        "Enrollment deleted"
                    } else {
                        "Enrollment rebuilt from stored samples"
                    },
                    cx,
                )
            });
        })
        .detach();
    }
    fn finish_action(&mut self, result: Result<(), String>, success: &str, cx: &mut Context<Self>) {
        self.busy = false;
        self.enrolling = false;
        self.progress.update(cx, |progress, cx| progress.finish(cx));
        match result {
            Ok(()) => {
                self.message = Some(success.into());
                self.refresh(cx);
            }
            Err(error) => self.error = Some(error),
        }
        cx.notify();
    }
    fn enroll(&mut self, source: SourceMeeting, cx: &mut Context<Self>) {
        if self.busy {
            return;
        }
        self.generation += 1;
        self.loading = false;
        let Some(pool) = AppServices::global(cx).pool() else {
            return;
        };
        let profile = self.profile_id.clone();
        if !self
            .targets
            .iter()
            .any(|target| target.id == self.target_id)
        {
            self.error = Some("Choose an available processing target before enrollment".into());
            cx.notify();
            return;
        }
        let target = self.target_id.clone();
        let model = self.model;
        let cancelled = self.cancelled.clone();
        cancelled.store(false, std::sync::atomic::Ordering::SeqCst);
        self.busy = true;
        self.enrolling = true;
        let progress = self.progress.update(cx, |progress, cx| progress.start(cx));
        self.error = None;
        cx.notify();
        let io = Io::global(cx);
        cx.spawn(async move |this, cx| {
            let result = io
                .spawn(async move {
                    enrollment::confirm_enrollment_source(
                        &pool,
                        &profile,
                        &source.id,
                        &source.speaker,
                        source.group_version.as_deref(),
                    )
                    .await?;
                    enrollment::enroll_from_meeting(
                        &pool,
                        &profile,
                        &source.id,
                        &source.speaker,
                        model,
                        &target,
                        &cancelled,
                        progress,
                    )
                    .await
                    .map(|_| ())
                })
                .await;
            let _ = this.update(cx, |this, cx| {
                this.finish_action(
                    result
                        .map_err(|e| e.to_string())
                        .and_then(|r| r.map_err(|e| e.to_string())),
                    "Confirmed voice evidence enrolled",
                    cx,
                )
            });
        })
        .detach();
    }
    fn play(&mut self, sample_id: String, index: usize, cx: &mut Context<Self>) {
        let key = format!("{sample_id}:{index}");
        let generation = self
            .playback_generation
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst)
            + 1;
        let playback_generation = self.playback_generation.clone();
        if self.playing.as_deref() == Some(&key) {
            parley_core::audio::playback::stop();
            self.playing = None;
            self.loading_playback = false;
            cx.notify();
            return;
        }
        let Some(pool) = AppServices::global(cx).pool() else {
            return;
        };
        let sink = AppServices::global(cx).sink.clone();
        let io = Io::global(cx);
        parley_core::audio::playback::stop();
        self.loading_playback = true;
        self.playing = Some(key.clone());
        self.error = None;
        cx.spawn(async move |this, cx| {
            let result = io
                .spawn(async move {
                    let sources = enrollment::source_playback(&pool, &sample_id)
                        .await
                        .map_err(|e| e.to_string())?;
                    let source = sources
                        .get(index)
                        .ok_or_else(|| "This source excerpt is unavailable".to_string())?;
                    if !source.downmix {
                        return Err(
                            "This evidence channel needs a compatible playback extractor".into(),
                        );
                    }
                    let mut all_samples = Vec::new();
                    let mut format = None;
                    for (position, end) in playback_ranges(source.start, source.end)? {
                        if playback_generation.load(std::sync::atomic::Ordering::SeqCst)
                            != generation
                        {
                            return Ok(());
                        }
                        let bytes = parley_core::audio::clip::extract_clip_wav(
                            &pool,
                            &source.meeting_id,
                            position,
                            end,
                            None,
                        )
                        .await?;
                        let (samples, rate, channels) =
                            parley_core::audio::clip::parse_wav_pcm16(&bytes)?;
                        if format.is_some_and(|old| old != (rate, channels)) {
                            return Err(
                                "Source audio format changed while preparing playback".into()
                            );
                        }
                        format = Some((rate, channels));
                        all_samples.extend(samples);
                    }
                    if playback_generation.load(std::sync::atomic::Ordering::SeqCst) != generation {
                        return Ok(());
                    }
                    let (rate, channels) =
                        format.ok_or_else(|| "Source interval contains no audio".to_string())?;
                    let samples = all_samples;
                    parley_core::audio::playback::play_pcm_i16(&sink, samples, rate, channels)
                })
                .await;
            let _ = this.update(cx, |this, cx| {
                if this.playing.as_deref() != Some(&key) {
                    return;
                }
                this.loading_playback = false;
                if let Err(error) = result.map_err(|e| e.to_string()).and_then(|r| r) {
                    this.error = Some(error);
                    this.playing = None;
                }
                cx.notify();
            });
        })
        .detach();
        cx.notify();
    }
}
impl Drop for EnrollmentDetails {
    fn drop(&mut self) {
        self.playback_generation
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        self.cancelled
            .store(true, std::sync::atomic::Ordering::SeqCst);
        if self.playing.is_some() {
            parley_core::audio::playback::stop();
        }
    }
}
impl Render for EnrollmentDetails {
    fn render(&mut self, _: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let mut body = v_flex()
            .w_full()
            .gap_3()
            .p_4()
            .border_1()
            .border_color(cx.theme().border)
            .rounded_md()
            .child(
                h_flex()
                    .w_full()
                    .justify_between()
                    .child(
                        div()
                            .text_sm()
                            .font_semibold()
                            .child("Voice enrollment and source samples"),
                    )
                    .child(
                        Button::new("refresh-enrollment-evidence")
                            .ghost()
                            .label("Refresh")
                            .disabled(self.busy || self.loading)
                            .on_click(cx.listener(|this, _, _, cx| {
                                if !this.busy && !this.loading {
                                    this.refresh(cx);
                                }
                            })),
                    ),
            );
        if self.loading {
            body = body.child(div().text_xs().child("Loading evidence…"));
        }
        if self.enrollments.is_empty() && !self.loading {
            body=body.child(div().text_xs().text_color(cx.theme().muted_foreground).child("No enrolled voice samples for this identity. Naming a speaker can reuse clean evidence saved by its pipeline; if none is available, select a source recording below to confirm the speaker and extract it."));
        }
        for item in &self.enrollments {
            let space = item.space_id.clone();
            let relearn = space.clone();
            let delete = space.clone();
            let model = item
                .embedding_space
                .as_ref()
                .map(|space| space.model.as_str())
                .unwrap_or_default();
            let label = if model == "campplus" || space.starts_with("campplus:") {
                "CAMPPlus (Sherpa)"
            } else if model == "wespeaker"
                || space.starts_with("wespeaker:")
                || space.starts_with("community1:")
            {
                "WeSpeaker (Community-1)"
            } else {
                "Legacy / unknown model"
            };
            let details_space = space.clone();
            let mut row = v_flex()
                .w_full()
                .gap_2()
                .p_3()
                .border_1()
                .border_color(cx.theme().border)
                .rounded_md()
                .child(
                    h_flex()
                        .w_full()
                        .justify_between()
                        .items_center()
                        .gap_2()
                        .child(div().text_sm().font_medium().child(format!(
                            "{label} · {}",
                            enrollment_status_label(&item.status)
                        )))
                        .child(
                            Button::new(format!("enrollment-details-{space}"))
                                .ghost()
                                .label(if self.expanded_space.as_deref() == Some(&space) {
                                    "Hide details"
                                } else {
                                    "Details"
                                })
                                .on_click(cx.listener(move |this, _, _, cx| {
                                    this.expanded_space =
                                        if this.expanded_space.as_deref() == Some(&details_space) {
                                            None
                                        } else {
                                            Some(details_space.clone())
                                        };
                                    cx.notify();
                                })),
                        ),
                )
                .child(
                    div()
                        .text_xs()
                        .text_color(cx.theme().muted_foreground)
                        .child(format!(
                            "{} contributing sample{} · {:.1}s clean audio · Updated {}",
                            item.sample_count,
                            if item.sample_count == 1 { "" } else { "s" },
                            item.clean_seconds,
                            item.updated_at
                        )),
                );
            if self.expanded_space.as_deref() == Some(&space) {
                let metadata = serde_json::json!({"embedding_space":item.embedding_space,"space_id":item.space_id,"enrollment_revision":item.revision,"extraction_version":item.extraction_version,"status":item.status,"updated_at":item.updated_at});
                row = row.child(
                    div()
                        .text_xs()
                        .text_color(cx.theme().muted_foreground)
                        .child(serde_json::to_string_pretty(&metadata).unwrap_or_default()),
                );
            }
            let evidence: Vec<_> = self
                .samples
                .iter()
                .filter(|s| s.space_id == space)
                .collect();
            if evidence.is_empty() {
                row=row.child(div().text_xs().text_color(cx.theme().muted_foreground).child("Source samples unavailable — legacy evidence has no playable audio references."));
            }
            if evidence.iter().any(|sample| {
                evidence_is_duplicate(sample.duplicate_of.as_deref(), &sample.quality_status)
            }) {
                row = row.child(div().text_xs().text_color(cx.theme().muted_foreground).child(
                    "Removing a sample also removes duplicate references to the same audio evidence.",
                ));
            }
            for sample in evidence {
                let remove = sample.id.clone();
                let mut sample_row = v_flex()
                    .gap_1()
                    .child(
                        h_flex()
                            .gap_2()
                            .items_center()
                            .child(div().flex_1().text_xs().child(format!(
                                "{:.1}s · {}",
                                sample.clean_seconds,
                                if sample.available {
                                    "Source recording available"
                                } else {
                                    "Sample unavailable"
                                }
                            )))
                            .child(
                                Button::new(format!("remove-evidence-{}", sample.id))
                                    .ghost()
                                    .label("Remove sample")
                                    .disabled(self.busy)
                                    .on_click(cx.listener(move |this, _, _, cx| {
                                        this.remove_sample(remove.clone(), cx)
                                    })),
                            ),
                    )
                    .child(
                        div()
                            .text_xs()
                            .text_color(cx.theme().muted_foreground)
                            .child(sample_contribution_label(
                                sample.contributes,
                                sample.duplicate_of.as_deref(),
                                &sample.quality_status,
                            )),
                    );
                for (index, interval) in sample.intervals.iter().enumerate() {
                    let sample_id = sample.id.clone();
                    let key = format!("{sample_id}:{index}");
                    let meeting = sample.source_meeting_id.clone();
                    let start = interval.start;
                    sample_row = sample_row.child(
                        h_flex()
                            .gap_2()
                            .items_center()
                            .child(
                                Button::new(format!("play-evidence-{key}"))
                                    .outline()
                                    .label(if self.playing.as_deref() == Some(&key) {
                                        "Stop".into()
                                    } else {
                                        format!("Play {:.1}–{:.1}s", interval.start, interval.end)
                                    })
                                    .disabled(!sample.available)
                                    .on_click(cx.listener(move |this, _, _, cx| {
                                        this.play(sample_id.clone(), index, cx)
                                    })),
                            )
                            .when_some(meeting, |row, id| {
                                row.child(
                                    Button::new(format!("open-source-{}-{index}", sample.id))
                                        .ghost()
                                        .label("Open source meeting")
                                        .on_click(move |_, _, cx| {
                                            crate::shell::open_meeting_at(
                                                id.clone(),
                                                Some(start),
                                                cx,
                                            )
                                        }),
                                )
                            }),
                    );
                }
                row = row.child(sample_row);
            }
            body = body.child(
                row.child(
                    h_flex()
                        .gap_2()
                        .child(
                            Button::new(format!("relearn-enrollment-{space}"))
                                .outline()
                                .label("Rebuild from samples")
                                .disabled(self.busy || item.sample_count == 0)
                                .on_click(cx.listener(move |this, _, _, cx| {
                                    this.enrollment_action(relearn.clone(), false, cx)
                                })),
                        )
                        .child(
                            Button::new(format!("delete-enrollment-{space}"))
                                .ghost()
                                .label("Delete this enrollment")
                                .disabled(self.busy)
                                .on_click(cx.listener(move |this, _, _, cx| {
                                    this.enrollment_action(delete.clone(), true, cx)
                                })),
                        ),
                ),
            );
        }
        let view = cx.entity().downgrade();
        let models = [
            (Identification::Campplus, "CAMPPlus (Sherpa)"),
            (Identification::Wespeaker, "WeSpeaker (Community-1)"),
        ];
        let model_button = Button::new("enrollment-model")
            .outline()
            .label(crate::views::processing::identification_label(self.model))
            .dropdown_menu(move |menu, _, _| {
                let mut menu = menu;
                for (model, label) in models {
                    let view = view.clone();
                    menu = menu.item(PopupMenuItem::new(label).on_click(move |_, _, cx| {
                        let _ = view.update(cx, |this, cx| {
                            this.model = model;
                            cx.notify();
                        });
                    }));
                }
                menu
            });
        let target_label = self
            .targets
            .iter()
            .find(|t| t.id == self.target_id)
            .map(|t| t.name.clone())
            .unwrap_or_else(|| "Choose target".into());
        let targets = self.targets.clone();
        let view = cx.entity().downgrade();
        let target_button = Button::new("enrollment-target")
            .outline()
            .label(target_label)
            .dropdown_menu(move |menu, _, _| {
                let mut menu = menu;
                for target in &targets {
                    let id = target.id.clone();
                    let view = view.clone();
                    menu = menu.item(PopupMenuItem::new(target.name.clone()).on_click(
                        move |_, _, cx| {
                            let _ = view.update(cx, |this, cx| {
                                this.target_id = id.clone();
                                cx.notify();
                            });
                        },
                    ));
                }
                menu
            });
        let sources = self.sources.clone();
        let view = cx.entity().downgrade();
        let enroll_button = Button::new("enroll-from-confirmed-meeting")
            .outline()
            .label(if self.busy {
                "Working…"
            } else {
                "Confirm identity and enroll…"
            })
            .disabled(self.busy || sources.is_empty())
            .dropdown_menu(move |menu, _, _| {
                let mut menu = menu;
                for source in &sources {
                    let source = source.clone();
                    let view = view.clone();
                    menu = menu.item(
                        PopupMenuItem::new(format!("{} · {}", source.title, source.speaker))
                            .on_click(move |_, _, cx| {
                                let _ = view.update(cx, |this, cx| this.enroll(source.clone(), cx));
                            }),
                    );
                }
                menu
            });
        body=body.child(h_flex().gap_2().flex_wrap().child(model_button).child(target_button).child(enroll_button)).child(div().text_xs().text_color(cx.theme().muted_foreground).child("Choose a model, then confirm that the selected speaker in the saved recording is this person. New evidence is extracted from that source audio."));
        if self.enrolling {
            body = body.child(self.progress.clone()).child(
                Button::new("cancel-enrollment-work")
                    .ghost()
                    .label("Cancel")
                    .on_click(cx.listener(|this, _, _, cx| {
                        this.cancelled
                            .store(true, std::sync::atomic::Ordering::SeqCst);
                        cx.notify();
                    })),
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
        if let Some(m) = &self.message {
            body = body.child(div().text_xs().child(m.clone()));
        }
        body
    }
}

fn enrollment_status_label(status: &str) -> &str {
    match status {
        "ready" => "Ready",
        "insufficient_clean_evidence" | "insufficient_evidence" | "unusable" => {
            "Insufficient clean evidence"
        }
        "legacy_unknown" | "legacy" => "Legacy / unknown",
        "model_unavailable" | "unavailable" => "Model unavailable",
        _ => "Unknown status",
    }
}

fn evidence_is_duplicate(duplicate_of: Option<&str>, quality_status: &str) -> bool {
    duplicate_of.is_some() || quality_status == "duplicate_reference"
}

fn sample_contribution_label(
    contributes: bool,
    duplicate_of: Option<&str>,
    quality_status: &str,
) -> &'static str {
    if evidence_is_duplicate(duplicate_of, quality_status) {
        return "Duplicate audio evidence — not counted in voice profile";
    }
    if contributes {
        return "Contributes to this voice profile";
    }
    match quality_status {
        "invalid_embedding" => "Invalid voice evidence — not counted in voice profile",
        "invalid_intervals" => "Invalid source timing — not counted in voice profile",
        "insufficient_clean_evidence" => "Insufficient clean audio — not counted in voice profile",
        "usable" => "Not counted in voice profile",
        _ => "Evidence quality unknown — not counted in voice profile",
    }
}

/// The clip extractor limits each request to two minutes. Split instead of
/// silently truncating longer enrollment evidence.
fn playback_ranges(start: f64, end: f64) -> Result<Vec<(f64, f64)>, String> {
    if !start.is_finite() || !end.is_finite() || start < 0.0 || end <= start {
        return Err("Source excerpt has invalid timing".into());
    }
    let mut ranges = Vec::new();
    let mut position = start;
    while position < end {
        let next = (position + 120.0).min(end);
        ranges.push((position, next));
        position = next;
    }
    Ok(ranges)
}
#[cfg(test)]
mod tests {
    use super::playback_ranges;
    #[test]
    fn contribution_labels_distinguish_duplicates_and_unusable_evidence() {
        assert_eq!(
            super::sample_contribution_label(false, Some("original"), "usable"),
            "Duplicate audio evidence — not counted in voice profile"
        );
        assert_eq!(
            super::sample_contribution_label(false, None, "duplicate_reference"),
            "Duplicate audio evidence — not counted in voice profile"
        );
        assert_eq!(
            super::sample_contribution_label(true, None, "usable"),
            "Contributes to this voice profile"
        );
        for (status, label) in [
            ("invalid_embedding", "Invalid voice evidence"),
            ("invalid_intervals", "Invalid source timing"),
            ("insufficient_clean_evidence", "Insufficient clean audio"),
            ("unknown", "Evidence quality unknown"),
        ] {
            assert_eq!(
                super::sample_contribution_label(false, None, status),
                format!("{label} — not counted in voice profile")
            );
        }
    }
    #[test]
    fn enrollment_statuses_use_plain_labels() {
        assert_eq!(
            super::enrollment_status_label("legacy_unknown"),
            "Legacy / unknown"
        );
        assert_eq!(
            super::enrollment_status_label("insufficient_clean_evidence"),
            "Insufficient clean evidence"
        );
        assert_eq!(
            super::enrollment_status_label("model_unavailable"),
            "Model unavailable"
        );
    }
    #[test]
    fn enrollment_playback_keeps_every_source_second() {
        assert_eq!(
            playback_ranges(10.0, 311.0).unwrap(),
            vec![(10.0, 130.0), (130.0, 250.0), (250.0, 311.0)]
        );
        assert_eq!(playback_ranges(4.2, 5.8).unwrap(), vec![(4.2, 5.8)]);
    }
    #[test]
    fn invalid_evidence_ranges_are_rejected() {
        for (start, end) in [(0.0, f64::NAN), (-1.0, 2.0), (4.0, 4.0), (4.0, 2.0)] {
            assert!(playback_ranges(start, end).is_err());
        }
    }
}
