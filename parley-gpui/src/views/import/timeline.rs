//! Observed phase timings. Percentages belong to subtasks, never guessed overall time.
use std::time::Duration;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Phase { Copy, Prepare, Speech, Speakers, Verify, Save, Identify, Finish }
impl Phase {
    pub fn label(self) -> &'static str {
        match self {
            Self::Copy => "Save an audio copy of the recording",
            Self::Prepare => "Prepare and verify audio",
            Self::Speech => "Transcribe speech",
            Self::Speakers => "Detect speakers and overlapping speech",
            Self::Verify => "Verify processing results",
            Self::Save => "Save transcript and speaker groups",
            Self::Identify => "Match enrolled speakers",
            Self::Finish => "Export and finish",
        }
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum State { Pending, Active, Done, Failed, Skipped }
#[derive(Clone)]
pub struct Step {
    pub phase: Phase,
    pub state: State,
    elapsed: Duration,
}
impl Step {
    pub fn elapsed(&self, active_since: Duration, now: Duration) -> Duration {
        self.elapsed + if self.state == State::Active { now.saturating_sub(active_since) } else { Duration::ZERO }
    }
}
#[derive(Clone)]
pub struct Timeline {
    pub steps: Vec<Step>,
    pub active_since: Duration,
    active: Option<usize>,
    pub finished_at: Option<Duration>,
}
impl Timeline {
    pub fn new(speakers_first: bool, identify: bool) -> Self {
        use Phase::*;
        let mut phases = vec![Copy, Prepare];
        phases.extend(if speakers_first { [Speakers, Speech] } else { [Speech, Speakers] });
        phases.extend([Verify, Save]);
        if identify { phases.push(Identify); }
        phases.push(Finish);
        Self { steps: phases.into_iter().map(|phase| Step { phase, state: State::Pending, elapsed: Duration::ZERO }).collect(), active: None, active_since: Duration::ZERO, finished_at: None }
    }
    pub fn update(&mut self, stage: &str, now: Duration) {
        if self.finished_at.is_some() { return; }
        use Phase::*;
        if stage == "identification_failed" {
            self.close_active(now, State::Failed);
            return;
        }
        let phase = match stage {
            "validating" | "audio_inspection" | "audio_extraction" | "audio_compression" | "copying" | "flushing" | "registering" => Copy,
            "audio_integrity" | "decoding" | "vad" | "runtime" | "audio_conversion" | "audio_decoding" | "audio_resampling" | "vad_progress" | "worker_decoding" | "worker_decoding_progress" | "worker_verifying" => Prepare,
            "model_loading" | "recognition" | "transcribing" | "model_integrity" => Speech,
            "speakers" | "diarizing" | "diarizing_progress" => Speakers,
            "verifying" => Verify,
            "saving" => Save,
            "identifying" => Identify,
            "finishing" => Finish,
            "embedding" | "embedding_progress" if self.active.is_some_and(|i| self.steps[i].phase == Identify) => Identify,
            "embedding" | "embedding_progress" => Speakers,
            // Transfers and worker queueing can occur again between speech and speakers.
            _ => self.active.map(|i| self.steps[i].phase).unwrap_or(Prepare),
        };
        let Some(index) = self.steps.iter().position(|s| s.phase == phase) else { return; };
        if self.active == Some(index) { return; }
        self.close_active(now, State::Done);
        self.steps[index].state = State::Active;
        self.active = Some(index);
        self.active_since = now;
    }
    fn close_active(&mut self, now: Duration, state: State) {
        if let Some(index) = self.active.take() {
            let step = &mut self.steps[index];
            step.elapsed += now.saturating_sub(self.active_since);
            step.state = state;
        }
    }
    pub fn finish(&mut self, now: Duration, success: bool) {
        self.close_active(now, if success { State::Done } else { State::Failed });
        for step in &mut self.steps {
            if step.state == State::Pending { step.state = State::Skipped; }
        }
        self.finished_at = Some(now);
    }
}
/// Rate measured within one subtask. A percentage reset starts a new estimate.
#[derive(Clone, Default)]
pub struct Estimate {
    key: String,
    first: Option<(u32, Duration)>,
    last: Option<(u32, Duration)>,
}
impl Estimate {
    pub fn update(&mut self, stage: &str, message: &str, percent: Option<u32>, now: Duration) {
        let key = if stage == "diarizing_progress" { format!("{stage}:{message}") } else { stage.to_owned() };
        let Some(percent) = percent else { *self = Self::default(); return; };
        if self.key != key || self.last.is_some_and(|(p, _)| percent < p) {
            *self = Self { key, first: Some((percent, now)), last: Some((percent, now)) };
        } else if self.last.is_none_or(|(p, _)| percent > p) {
            self.last = Some((percent, now));
        }
    }
    pub fn remaining(&self, now: Duration) -> Option<Duration> {
        let (first, started) = self.first?;
        let (last, updated) = self.last?;
        let elapsed = updated.saturating_sub(started).as_secs_f64();
        if last >= 100 || last <= first || elapsed < 3. { return None; }
        let remaining = elapsed * (100-last) as f64 / (last-first) as f64;
        let stale = now.saturating_sub(updated).as_secs_f64();
        // Never count down to a false zero while the worker is still busy.
        let expected_update_interval = elapsed / (last-first) as f64;
        if stale >= remaining || stale > (3. * expected_update_interval).max(60.) { return None; }
        Some(Duration::from_secs_f64(remaining-stale))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn at(n: u64) -> Duration { Duration::from_secs(n) }
    #[test]
    fn extraction_and_compression_happen_before_copy_finishes() {
        let mut t = Timeline::new(true, true);
        t.update("audio_inspection", at(0));
        t.update("audio_extraction", at(1));
        t.update("audio_compression", at(3));
        t.update("copying", at(8));
        assert_eq!(t.steps[0].state, State::Active);
        assert_eq!(t.steps[1].state, State::Pending);
        t.update("audio_integrity", at(10));
        assert_eq!(t.steps[0].state, State::Done);
        assert_eq!(t.steps[0].elapsed, at(10));
    }
    #[test]
    fn upload_eta_survives_byte_counter_updates_and_stops_for_confirmation() {
        let mut e = Estimate::default();
        e.update("uploading", "Uploading recording: 0 / 100 MiB", Some(0), at(0));
        e.update("uploading", "Uploading recording: 25 / 100 MiB", Some(25), at(10));
        assert_eq!(e.remaining(at(10)), Some(at(30)));
        e.update("uploading", "Uploading recording: 50 / 100 MiB", Some(50), at(20));
        assert_eq!(e.remaining(at(20)), Some(at(20)));
        e.update("upload_verifying", "Confirming receipt and checksum", None, at(40));
        assert!(e.remaining(at(40)).is_none());
    }
    #[test]
    fn slow_batches_keep_their_estimate_between_updates() {
        let mut e = Estimate::default();
        e.update("diarizing_progress", "embeddings", Some(0), at(0));
        e.update("diarizing_progress", "embeddings", Some(1), at(120));
        assert_eq!(e.remaining(at(240)), Some(at(11760)));
        assert!(e.remaining(at(481)).is_none());
        e.update("diarizing_progress", "embeddings", Some(100), at(500));
        assert!(e.remaining(at(500)).is_none());
    }
    #[test]
    fn estimate_is_measured_resets_between_subtasks_and_expires() {
        let mut e = Estimate::default();
        e.update("diarizing_progress", "segmentation", Some(0), at(0));
        assert!(e.remaining(at(1)).is_none());
        e.update("diarizing_progress", "segmentation", Some(25), at(10));
        assert_eq!(e.remaining(at(10)), Some(at(30)));
        assert!(e.remaining(at(41)).is_none());
        e.update("diarizing_progress", "embeddings", Some(0), at(42));
        assert!(e.remaining(at(43)).is_none());
        e.update("diarizing_progress", "embeddings", Some(50), at(52));
        assert_eq!(e.remaining(at(52)), Some(at(10)));
        e.update("diarizing", "clustering", None, at(53));
        assert!(e.remaining(at(54)).is_none());
    }
    #[test]
    fn subtasks_accumulate_without_resetting_phase_time() {
        let mut t = Timeline::new(false, true);
        t.update("copying", at(0)); t.update("flushing", at(5));
        t.update("audio_integrity", at(8)); t.update("decoding", at(10));
        t.update("recognition", at(20));
        assert_eq!(t.steps[0].elapsed, at(8));
        assert_eq!(t.steps[1].elapsed, at(12));
        assert_eq!(t.steps[2].elapsed(t.active_since, at(25)), at(5));
        t.update("queued", at(26));
        assert_eq!(t.active, Some(2));
    }
    #[test]
    fn failure_does_not_mark_unrun_steps_successful() {
        let mut t = Timeline::new(true, false);
        t.update("diarizing", at(3)); t.finish(at(13), false);
        assert_eq!(t.steps[2].state, State::Failed);
        assert_eq!(t.steps[2].elapsed, at(10));
        assert_eq!(t.steps[3].state, State::Skipped);
        t.update("transcribing", at(15));
        assert_eq!(t.steps[3].state, State::Skipped);
    }
    #[test]
    fn reentered_phase_keeps_accumulated_time_and_optional_steps_are_not_invented() {
        let mut t = Timeline::new(false, false);
        t.update("recognition", at(1)); t.update("speakers", at(6));
        t.update("transcribing", at(10)); t.finish(at(15), true);
        assert_eq!(t.steps[2].elapsed, at(10));
        assert!(!t.steps.iter().any(|s| s.phase == Phase::Identify));
        assert_eq!(t.steps[0].state, State::Skipped);
    }
    #[test]
    fn optional_identity_failure_remains_visible_after_successful_import() {
        let mut t = Timeline::new(false, true);
        t.update("identifying", at(1)); t.update("identification_failed", at(5));
        t.update("finishing", at(6)); t.finish(at(7), true);
        assert_eq!(t.steps.iter().find(|s| s.phase == Phase::Identify).unwrap().state, State::Failed);
        assert_eq!(t.steps.last().unwrap().state, State::Done);
    }

}
