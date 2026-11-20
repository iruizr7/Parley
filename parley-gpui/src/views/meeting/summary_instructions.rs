use gpui_kit::{
    AppContext as _, Context, Entity, Window,
    component::{
        WindowExt as _,
        input::{InputEvent, InputState},
        notification::Notification,
    },
};
use parley_core::database::repositories::meeting::MeetingsRepository;
use sqlx::SqlitePool;
use tokio::sync::mpsc;

use super::MeetingView;
use crate::{app_state::AppServices, runtime::Io};

pub(super) enum State {
    Loading,
    Loaded(String),
    Ready(Entity<InputState>),
    Unavailable,
}

pub(super) struct Request {
    pool: SqlitePool,
    meeting_id: String,
    action: Action,
}

enum Action {
    Load { generation: u64 },
    Save(String),
}

pub(super) type Sender = mpsc::UnboundedSender<Request>;

// Serialize reads and writes so a quick return to a meeting reads its latest
// edit, and an older write never overwrites a newer one.
pub(super) fn start(cx: &mut Context<MeetingView>) -> Sender {
    let (tx, mut rx) = mpsc::unbounded_channel::<Request>();
    let io = Io::global(cx);
    cx.spawn(async move |view, cx| {
        while let Some(Request {
            pool,
            meeting_id,
            action,
        }) = rx.recv().await
        {
            match action {
                Action::Load { generation } => {
                    let result = io
                        .spawn(async move {
                            MeetingsRepository::get_summary_instructions(&pool, &meeting_id).await
                        })
                        .await;
                    let _ = view.update(cx, |view, cx| {
                        if view.generation != generation {
                            return;
                        }
                        view.summary_instructions = match result {
                            Ok(Ok(text)) => State::Loaded(text),
                            _ => State::Unavailable,
                        };
                        cx.notify();
                    });
                }
                Action::Save(text) => {
                    let result = io
                        .spawn(async move {
                            MeetingsRepository::set_summary_instructions(&pool, &meeting_id, &text)
                                .await
                        })
                        .await;
                    if !matches!(result, Ok(Ok(()))) {
                        let _ = view.update(cx, |_, cx| {
                            crate::ui::with_main_window(cx, |window, cx| {
                                window.push_notification(
                                    Notification::error("Could not save summary instructions."),
                                    cx,
                                );
                            });
                        });
                    }
                }
            }
        }
    })
    .detach();
    tx
}

impl MeetingView {
    pub(super) fn load_summary_instructions(&mut self, pool: SqlitePool, meeting_id: String) {
        if self
            .summary_instructions_tx
            .send(Request {
                pool,
                meeting_id,
                action: Action::Load {
                    generation: self.generation,
                },
            })
            .is_err()
        {
            self.summary_instructions = State::Unavailable;
        }
    }

    pub(super) fn prepare_summary_instructions(
        &mut self,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let State::Loaded(text) = &self.summary_instructions else {
            return;
        };
        let (Some(meeting_id), Some(pool)) = (&self.meeting_id, AppServices::global(cx).pool())
        else {
            self.summary_instructions = State::Unavailable;
            return;
        };
        let meeting_id = meeting_id.clone();
        let generation = self.generation;
        let input = cx.new(|cx| {
            InputState::new(window, cx)
                .placeholder("Custom instructions for this meeting (optional)…")
                .default_value(text.clone())
        });
        cx.subscribe(&input, move |view, input, event: &InputEvent, cx| {
            if view.generation == generation && matches!(event, InputEvent::Change) {
                let text = input.read(cx).value().to_string();
                let _ = view.summary_instructions_tx.send(Request {
                    pool: pool.clone(),
                    meeting_id: meeting_id.clone(),
                    action: Action::Save(text),
                });
            }
        })
        .detach();
        self.summary_instructions = State::Ready(input);
    }
}
