//! Pure formatting helpers for the meeting page: transcript timestamps and
//! the flattened transcript text sent to summary generation. The
//! transcript format matches what the retired React frontend sent, so
//! summaries of old and new meetings read the same.

use chrono::{DateTime, Utc};
use parley_core::database::models::MeetingTranscript;

/// Recording-relative `[MM:SS]` timestamp when `audio_start_time` is known,
/// otherwise the transcript's wall-clock `timestamp` string as-is (older
/// rows / imports predate audio-relative timestamps).
pub fn segment_timestamp(audio_start_time: Option<f64>, fallback_timestamp: &str) -> String {
    match audio_start_time {
        Some(seconds) if seconds.is_finite() && seconds >= 0.0 => {
            let total = seconds.floor() as u64;
            format!("[{:02}:{:02}]", total / 60, total % 60)
        }
        _ => fallback_timestamp.to_string(),
    }
}

/// A displayed turn retains its primary speaker. Other speakers' recognized
/// words appear as inline interventions; overlap metadata never creates words.
#[derive(Debug)]
pub struct TranscriptTurn {
    pub rows: std::ops::Range<usize>,
    pub body: String,
    /// Byte ranges of inline speaker labels, paired with their original row.
    pub speaker_links: Vec<(std::ops::Range<usize>, usize)>,
}

/// Preserve a new turn after a real pause, but ignore recognition chunk
/// boundaries within continuous speech. These are display groups only.
const MAX_TURN_PAUSE_SECONDS: f64 = 3.;

pub fn transcript_turns(transcripts: &[MeetingTranscript]) -> Vec<TranscriptTurn> {
    let mut turns: Vec<TranscriptTurn> = Vec::new();
    for (index, t) in transcripts.iter().enumerate() {
        let attach = turns.last().is_some_and(|turn| {
            let primary = &transcripts[turn.rows.start];
            let previous = &transcripts[index - 1];
            if primary.source != t.source || primary.speaker.is_none() || t.speaker.is_none() {
                return false;
            }
            let gap = previous.audio_end_time.zip(t.audio_start_time)
                .map(|(end, start)| start - end);
            if gap.is_some_and(|gap| !gap.is_finite() || gap > MAX_TURN_PAUSE_SECONDS) {
                return false;
            }
            if primary.speaker == t.speaker {
                // Adjacent ASR chunks are not separate speaker turns. Keep
                // explicit identity corrections and unknown timing separate.
                primary.voice_profile_id == t.voice_profile_id
                    && primary.speaker.as_ref().is_some_and(|name| !name.trim().is_empty())
                    && (gap.is_some()
                        || !t.overlapping_speakers.is_empty()
                        || !previous.overlapping_speakers.is_empty())
            } else {
                // Do not infer whose words these are from the list of active
                // voices. Only nest text already attributed to another voice
                // while the primary speaker is also known to be active.
                !t.overlapping_speakers.is_empty()
                    && (t.overlapping_speakers.iter().any(|s| Some(s) == primary.speaker.as_ref())
                        || (primary.audio_start_time.zip(primary.audio_end_time)
                            .zip(t.audio_start_time.zip(t.audio_end_time))
                            .is_some_and(|((a, b), (c, d))| a < d && c < b)))
            }
        });
        if attach {
            turns.last_mut().unwrap().rows.end = index + 1;
        } else {
            turns.push(TranscriptTurn { rows: index..index + 1, body: String::new(), speaker_links: Vec::new() });
        }
    }
    for turn in &mut turns {
        let primary = &transcripts[turn.rows.start].speaker;
        let mut interruption: Option<&str> = None;
        for index in turn.rows.clone() {
            let t = &transcripts[index];
            let inline = if t.speaker != *primary { t.speaker.as_deref() } else { None };
            if interruption != inline {
                if interruption.is_some() { turn.body.push_str("//"); }
                if !turn.body.is_empty() { turn.body.push(' '); }
                if let Some(speaker) = inline {
                    turn.body.push_str("//");
                    let start = turn.body.len();
                    turn.body.push_str(speaker);
                    turn.body.push(':');
                    turn.speaker_links.push((start..turn.body.len(), index));
                    turn.body.push(' ');
                }
            } else if !turn.body.is_empty() {
                turn.body.push(' ');
            }
            turn.body.push_str(t.text.trim());
            interruption = inline;
        }
        if interruption.is_some() { turn.body.push_str("//"); }
    }
    turns
}

/// One timestamped line per displayed turn; inline interruptions carry only
/// the interrupting speaker's label.
pub fn build_transcript_text(transcripts: &[MeetingTranscript]) -> String {
    transcript_turns(transcripts).iter().map(|turn| {
        let t = &transcripts[turn.rows.start];
        format!("{} {}", segment_timestamp(t.audio_start_time, &t.timestamp), turn.body)
    }).collect::<Vec<_>>().join("\n")
}

/// Header date, e.g. "11 Sep 2026, 14:03" in the meeting's local timezone.
pub fn format_header_date(created_at: DateTime<Utc>) -> String {
    created_at.with_timezone(&chrono::Local).format("%-d %b %Y, %H:%M").to_string()
}

/// Clipboard text for "Copy transcript": a `#`/`##` header, then one `[MM:SS] Speaker: text`
/// line per segment with a trailing two-space markdown line break.
pub fn copy_transcript_text(
    meeting_id: &str,
    title: &str,
    created_at: Option<DateTime<Utc>>,
    transcripts: &[MeetingTranscript],
) -> String {
    let header = format!("# Transcript of the Meeting: {meeting_id} - {title}\n\n");
    let date = format!(
        "## Date: {}\n\n",
        created_at.map(|d| d.with_timezone(&chrono::Local).format("%-m/%-d/%Y").to_string()).unwrap_or_default()
    );
    let lines = transcript_turns(transcripts).iter().map(|turn| {
        let t = &transcripts[turn.rows.start];
        let time = segment_timestamp(t.audio_start_time, &t.timestamp);
        let who = t.speaker.as_ref().map(|s| format!("{s}: ")).unwrap_or_default();
        format!("{time} {who}{}  ", turn.body)
    }).collect::<Vec<_>>().join("\n");
    format!("{header}{date}{lines}")
}

/// Clipboard text for "Copy summary". Mirrors `useCopyOperations.ts`'s
/// `handleCopySummary`: a title, a metadata block (meeting id, meeting date,
/// copy time), a rule, then the summary markdown as-is.
pub fn copy_summary_text(
    meeting_id: &str,
    title: &str,
    created_at: Option<DateTime<Utc>>,
    summary_markdown: &str,
) -> String {
    let header = format!("# Meeting Summary: {title}\n\n");
    let fmt = |d: DateTime<Utc>| d.with_timezone(&chrono::Local).format("%B %-d, %Y, %I:%M %p").to_string();
    let created = created_at.map(fmt).unwrap_or_default();
    let copied = fmt(Utc::now());
    let metadata =
        format!("**Meeting ID:** {meeting_id}\n**Date:** {created}\n**Copied on:** {copied}\n\n---\n\n");
    format!("{header}{metadata}{summary_markdown}")
}

/// Picks the default summary template for the generate/regenerate picker:
/// the stored default setting if it names a template that actually exists,
/// else `"standard_meeting"` if that exists, else the first available
/// template, else `"standard_meeting"` regardless (so a caller always gets
/// *some* id to pass to summary generation even if the template list
/// couldn't be loaded yet).
pub fn resolve_default_template(configured: Option<&str>, available: &[String]) -> String {
    if let Some(configured) = configured {
        if available.iter().any(|id| id == configured) {
            return configured.to_string();
        }
    }
    if available.iter().any(|id| id == "standard_meeting") {
        return "standard_meeting".to_string();
    }
    available.first().cloned().unwrap_or_else(|| "standard_meeting".to_string())
}

/// Whether the summary editor is open (`editing`) with text that differs
/// from the last-saved markdown — the condition the navigation-away guard
/// checks before letting the user leave the meeting page or switch to a
/// different meeting.
pub fn has_unsaved_summary_edits(editing: bool, editor_text: &str, saved_markdown: &str) -> bool {
    editing && editor_text != saved_markdown
}

// ---- Confidence indicator -------------------------------------------------
// Thresholds and labels carried over unchanged from the retired React
// frontend (including its `>= 0.4` "Medium" cutoff), so confidence reads
// the same as it always has.

/// One of the four confidence bands shown next to a transcript segment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConfidenceLevel {
    High,
    Good,
    Medium,
    Low,
}

impl ConfidenceLevel {
    /// `conf` is the raw 0.0-1.0 Whisper confidence score.
    pub fn for_confidence(conf: f32) -> Self {
        if conf >= 0.8 {
            ConfidenceLevel::High
        } else if conf >= 0.7 {
            ConfidenceLevel::Good
        } else if conf >= 0.4 {
            ConfidenceLevel::Medium
        } else {
            ConfidenceLevel::Low
        }
    }

    /// Mirrors `getConfidenceLabel`.
    pub fn label(self) -> &'static str {
        match self {
            ConfidenceLevel::High => "High confidence",
            ConfidenceLevel::Good => "Good confidence",
            ConfidenceLevel::Medium => "Medium confidence",
            ConfidenceLevel::Low => "Low confidence",
        }
    }
}

/// Tooltip text, e.g. "87% confidence - High confidence" — mirrors the
/// `title` attribute `ConfidenceIndicator.tsx` sets on its dot.
pub fn confidence_tooltip(conf: f32) -> String {
    let percent = (conf * 100.0).round() as i64;
    format!("{percent}% confidence - {}", ConfidenceLevel::for_confidence(conf).label())
}

// ---- Meeting notes ---------------------------------------------------------
// Mirrors `MeetingNotesPanel.tsx`'s `formatNoteTime`.

/// "Sep 11, 2:30 PM"-style note timestamp, or `""` for an unparseable
/// timestamp (matches the React component's `Number.isNaN` guard).
pub fn format_note_time(created_at: &str) -> String {
    match DateTime::parse_from_rfc3339(created_at) {
        Ok(dt) => dt.with_timezone(&chrono::Local).format("%b %-d, %-I:%M %p").to_string(),
        Err(_) => String::new(),
    }
}

// ---- Summary typewriter reveal ---------------------------------------------
// There's no existing React "typewriter" for `summary-stream` specifically
// (see `useSummaryStream.ts`, which appends deltas as-is) — this adapts the
// pacing algorithm `useTranscriptStreaming.ts` uses for live transcript
// segments (50ms tick, `max(2, ceil(remaining / ticks))` chars per tick)
// into a delta-buffering model, since the summary case streams over an
// unbounded duration rather than one fixed-length segment.

/// Reveal tick interval, matching `useTranscriptStreaming.ts`'s `INTERVAL_MS`.
pub const SUMMARY_REVEAL_INTERVAL_MS: u64 = 50;

/// How many characters to reveal per tick for a buffer of `pending_len`
/// characters not yet shown. Always reveals at least 1 char/tick so a small
/// trailing buffer still drains instead of stalling.
pub fn summary_reveal_chars_per_tick(pending_len: usize) -> usize {
    if pending_len == 0 {
        return 0;
    }
    // Same shape as useTranscriptStreaming's `Math.max(2, Math.ceil(remaining / totalTicks))`,
    // sized against one tick's worth of the *current* pending buffer so a
    // burst of deltas (e.g. after a network stall) catches up within a
    // handful of ticks instead of trickling out at a fixed rate forever.
    const TARGET_TICKS: usize = 8;
    (pending_len.div_ceil(TARGET_TICKS)).max(2).min(pending_len)
}

/// Index of the transcript row an "Ask Parley" citation at `seconds` points
/// at: the latest segment starting at or before that second (citations are
/// floored to whole seconds), else the earliest timed segment. `None` when
/// no segment has a recording-relative time. Doesn't assume `transcripts`
/// is sorted.
pub fn transcript_index_for_time(transcripts: &[MeetingTranscript], seconds: f64) -> Option<usize> {
    let timed = transcripts
        .iter()
        .enumerate()
        .filter_map(|(ix, t)| t.audio_start_time.filter(|s| s.is_finite()).map(|s| (ix, s)));
    let mut at_or_before: Option<(usize, f64)> = None;
    let mut earliest: Option<(usize, f64)> = None;
    for (ix, start) in timed {
        if start.floor() <= seconds && at_or_before.is_none_or(|(_, best)| start > best) {
            at_or_before = Some((ix, start));
        }
        if earliest.is_none_or(|(_, best)| start < best) {
            earliest = Some((ix, start));
        }
    }
    at_or_before.or(earliest).map(|(ix, _)| ix)
}

/// Zero means automatic estimation; explicit counts are bounded by the worker.
pub fn parse_speaker_count(value: &str) -> Result<i32, &'static str> {
    let value = value.trim();
    if value.is_empty() || value.eq_ignore_ascii_case("auto") {
        return Ok(0);
    }
    match value.parse::<i32>() {
        Ok(count @ 1..=64) => Ok(count),
        _ => Err("Enter 1–64 speakers, or Auto."),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn overlap(text: &str, speaker: &str) -> MeetingTranscript {
        let mut t = seg(text, Some(1.));
        t.speaker = Some(speaker.into());
        t.overlapping_speakers = vec!["S00".into(), speaker.into()];
        t
    }

    fn primary(text: &str, start: f64) -> MeetingTranscript {
        let mut t = seg(text, Some(start));
        t.speaker = Some("S00".into());
        t
    }

    fn timed_primary(text: &str, start: f64, end: f64) -> MeetingTranscript {
        let mut t = primary(text, start);
        t.audio_end_time = Some(end);
        t
    }

    #[test]
    fn continuous_same_speaker_chunks_form_one_display_and_clipboard_turn() {
        let rows = vec![timed_primary("We can discuss", 0., 2.),
            timed_primary("the details", 2., 4.), timed_primary("after a short pause.", 5.9, 8.)];
        let turns = transcript_turns(&rows);
        assert_eq!(turns.len(), 1);
        assert_eq!(turns[0].rows, 0..3);
        assert_eq!(turns[0].body, "We can discuss the details after a short pause.");
        assert!(turns[0].speaker_links.is_empty());
        let copy = copy_transcript_text("test", "Example", None, &rows);
        assert_eq!(copy.matches("S00:").count(), 1);
        assert!(copy.contains("[00:00] S00: We can discuss the details after a short pause."));
        // Fine-grained source timing is still available for citations/playback.
        assert_eq!(transcript_index_for_time(&rows, 6.), Some(2));
        assert!(turns[0].rows.contains(&2));
        assert_eq!(rows[2].audio_end_time, Some(8.));
    }

    #[test]
    fn long_pauses_identity_changes_and_unknown_timing_keep_turns_separate() {
        let rows = vec![timed_primary("First.", 0., 1.), timed_primary("After a pause.", 4.1, 5.)];
        assert_eq!(transcript_turns(&rows).len(), 2);
        let mut rows = vec![timed_primary("First.", 0., 1.), timed_primary("Next.", 1., 2.)];
        rows[1].voice_profile_id = Some("fictional-other-profile".into());
        assert_eq!(transcript_turns(&rows).len(), 2);
        rows[1].voice_profile_id = None;
        rows[1].speaker = None;
        assert_eq!(transcript_turns(&rows).len(), 2);
        rows[1].speaker = rows[0].speaker.clone();
        rows[0].audio_end_time = None;
        assert_eq!(transcript_turns(&rows).len(), 2);
        rows[0].audio_end_time = Some(f64::NAN);
        assert_eq!(transcript_turns(&rows).len(), 2);
    }

    #[test]
    fn merged_turn_keeps_inline_interruption_and_normal_speaker_change() {
        let mut interruption = overlap("One moment.", "S01");
        interruption.audio_start_time = Some(2.);
        interruption.audio_end_time = Some(3.);
        let mut next = timed_primary("My turn.", 6., 8.);
        next.speaker = Some("S02".into());
        let rows = vec![timed_primary("We can", 0., 1.), timed_primary("start.", 1., 2.),
            interruption, timed_primary("Continuing", 3., 4.), timed_primary("now.", 4., 5.), next];
        let turns = transcript_turns(&rows);
        assert_eq!(turns.len(), 2);
        assert_eq!(turns[0].rows, 0..5);
        assert_eq!(turns[0].body, "We can start. //S01: One moment.// Continuing now.");
        assert_eq!(turns[0].speaker_links[0].1, 2);
        assert_eq!(turns[1].body, "My turn.");
    }

    #[test]
    fn an_overlap_flag_alone_does_not_repeat_the_primary_label_or_invent_an_interruption() {
        let mut t = primary(" Keep going. ", 0.);
        t.overlapping_speakers = vec!["S00".into(), "S01".into()];
        let turns = transcript_turns(&[t.clone()]);
        assert_eq!(turns[0].body, "Keep going.");
        assert!(turns[0].speaker_links.is_empty());
        assert_eq!(t.text, " Keep going. ");
    }

    #[test]
    fn interruption_is_inside_the_primary_turn_and_resumption_has_no_label() {
        let rows = vec![primary("We can start.", 0.), overlap(" One moment,", "S01"),
            overlap(" please.", "S01"), primary("Let us continue.", 3.)];
        let turns = transcript_turns(&rows);
        assert_eq!(turns.len(), 1);
        assert_eq!(turns[0].rows, 0..4);
        assert_eq!(turns[0].body, "We can start. //S01: One moment, please.// Let us continue.");
        let (range, source) = &turns[0].speaker_links[0];
        assert_eq!(&turns[0].body[range.clone()], "S01:");
        assert_eq!(*source, 1);
        let copied = copy_transcript_text("test", "Example", None, &rows);
        assert!(copied.contains("[00:00] S00: We can start. //S01: One moment, please.// Let us continue."));
        assert_eq!(copied.matches("S00:").count(), 1);
        assert_eq!(copied.matches("S01:").count(), 1);
        assert_eq!(build_transcript_text(&rows).lines().count(), 1);
    }

    #[test]
    fn multiple_interruptions_and_primary_overlap_words_stay_in_the_same_turn() {
        let rows = vec![primary("Start.", 0.), overlap("Still speaking.", "S00"),
            overlap("Yes.", "S01"), overlap("Agreed.", "S02"),
            primary("Continue.", 2.), overlap("Okay.", "S01")];
        let turns = transcript_turns(&rows);
        assert_eq!(turns.len(), 1);
        assert_eq!(turns[0].body,
            "Start. Still speaking. //S01: Yes.// //S02: Agreed.// Continue. //S01: Okay.//");
    }

    #[test]
    fn ordinary_speaker_changes_and_unrelated_overlaps_remain_separate() {
        let mut other = primary("New turn.", 2.);
        other.speaker = Some("S03".into());
        let rows = vec![primary("Start.", 0.), overlap("Yes.", "S01"), other,
            overlap("Elsewhere.", "S02"), primary("Later.", 4.)];
        let turns = transcript_turns(&rows);
        assert_eq!(turns.iter().map(|t| t.rows.clone()).collect::<Vec<_>>(), [0..2, 2..3, 3..4, 4..5]);
        assert_eq!(turns[2].body, "Elsewhere.");
    }

    #[test]
    fn named_speakers_have_utf8_safe_inline_links_and_streams_stay_separate() {
        let mut base = primary("Start.", 0.);
        base.speaker = Some("Example host".into());
        let mut other = overlap("Yes.", "Fictional José");
        other.overlapping_speakers = vec!["Example host".into(), "Fictional José".into()];
        let turns = transcript_turns(&[base.clone(), other.clone()]);
        assert_eq!(turns.len(), 1);
        assert_eq!(turns[0].body, "Start. //Fictional José: Yes.//");
        let range = turns[0].speaker_links[0].0.clone();
        assert_eq!(&turns[0].body[range], "Fictional José:");
        other.source = Some("mic".into());
        assert_eq!(transcript_turns(&[base, other]).len(), 2);
    }

    #[test]
    fn post_meeting_speaker_input_accepts_auto_or_a_bounded_count() {
        for value in ["", " Auto ", "AUTO"] {
            assert_eq!(parse_speaker_count(value), Ok(0));
        }
        for value in ["1", "6", " 64 "] {
            assert_eq!(
                parse_speaker_count(value),
                Ok(value.trim().parse().unwrap())
            );
        }
        for value in ["0", "-1", "65", "six", "2.5"] {
            assert!(parse_speaker_count(value).is_err());
        }
    }

    #[test]
    fn timestamp_from_audio_start_time() {
        assert_eq!(segment_timestamp(Some(65.0), "2026-09-11T00:00:00Z"), "[01:05]");
        assert_eq!(segment_timestamp(Some(0.0), "x"), "[00:00]");
        assert_eq!(segment_timestamp(Some(3599.9), "x"), "[59:59]");
    }

    #[test]
    fn falls_back_to_wall_clock_timestamp() {
        assert_eq!(segment_timestamp(None, "2026-09-11T00:00:00Z"), "2026-09-11T00:00:00Z");
        assert_eq!(segment_timestamp(Some(-1.0), "fallback"), "fallback");
    }

    fn seg(text: &str, audio_start_time: Option<f64>) -> MeetingTranscript {
        MeetingTranscript {
            overlapping_speakers: Vec::new(),
            id: "1".into(),
            text: text.into(),
            timestamp: "2026-09-11T00:00:00Z".into(),
            audio_start_time,
            audio_end_time: None,
            duration: None,
            speaker: None,
            voice_profile_id: None,
            source: None,
            confidence: None,
        }
    }

    #[test]
    fn finds_transcript_row_for_citation_time() {
        let transcripts = vec![
            seg("a", Some(0.0)),
            seg("b", Some(10.4)),
            seg("untimed", None),
            seg("c", Some(754.9)),
        ];
        assert_eq!(transcript_index_for_time(&transcripts, 754.0), Some(3));
        assert_eq!(transcript_index_for_time(&transcripts, 12.0), Some(1));
        assert_eq!(transcript_index_for_time(&transcripts, 10.0), Some(1));
        assert_eq!(transcript_index_for_time(&transcripts, 9.0), Some(0));
        let late_start = vec![seg("x", Some(30.0)), seg("y", Some(20.0))];
        assert_eq!(transcript_index_for_time(&late_start, 5.0), Some(1));
        assert_eq!(transcript_index_for_time(&[seg("u", None)], 5.0), None);
    }

    #[test]
    fn builds_one_line_per_segment() {
        let transcripts = vec![seg("Hello", Some(0.0)), seg("World", Some(65.0))];
        assert_eq!(build_transcript_text(&transcripts), "[00:00] Hello\n[01:05] World");
    }

    #[test]
    fn empty_transcript_list_is_empty_string() {
        assert_eq!(build_transcript_text(&[]), "");
    }

    #[test]
    fn copy_transcript_includes_header_date_and_speaker_lines() {
        let mut a = seg("Hello", Some(0.0));
        a.speaker = Some("Alice".to_string());
        let transcripts = vec![a, seg("World", Some(65.0))];
        let created = DateTime::parse_from_rfc3339("2026-07-15T10:00:00Z")
            .unwrap()
            .with_timezone(&Utc);
        let text = copy_transcript_text("meeting-1", "Weekly Sync", Some(created), &transcripts);
        assert!(text.starts_with("# Transcript of the Meeting: meeting-1 - Weekly Sync\n\n"));
        assert!(text.contains("## Date:"));
        assert!(text.contains("[00:00] Alice: Hello  "));
        assert!(text.contains("[01:05] World  "));
    }

    #[test]
    fn copy_summary_includes_metadata_and_body_verbatim() {
        let created = DateTime::parse_from_rfc3339("2026-07-15T10:00:00Z")
            .unwrap()
            .with_timezone(&Utc);
        let text = copy_summary_text("meeting-1", "Weekly Sync", Some(created), "## Key Points\n\n- Shipped");
        assert!(text.starts_with("# Meeting Summary: Weekly Sync\n\n"));
        assert!(text.contains("**Meeting ID:** meeting-1\n"));
        assert!(text.contains("**Date:**"));
        assert!(text.contains("**Copied on:**"));
        assert!(text.ends_with("---\n\n## Key Points\n\n- Shipped"));
    }

    #[test]
    fn resolve_default_template_uses_configured_when_available() {
        let available = vec!["standard_meeting".to_string(), "one_on_one".to_string()];
        assert_eq!(resolve_default_template(Some("one_on_one"), &available), "one_on_one");
    }

    #[test]
    fn resolve_default_template_falls_back_to_standard_meeting() {
        let available = vec!["standard_meeting".to_string(), "one_on_one".to_string()];
        assert_eq!(resolve_default_template(Some("missing"), &available), "standard_meeting");
        assert_eq!(resolve_default_template(None, &available), "standard_meeting");
    }

    #[test]
    fn resolve_default_template_falls_back_to_first_available() {
        let available = vec!["one_on_one".to_string(), "retro".to_string()];
        assert_eq!(resolve_default_template(None, &available), "one_on_one");
    }

    #[test]
    fn resolve_default_template_falls_back_to_standard_meeting_id_when_nothing_loaded() {
        assert_eq!(resolve_default_template(None, &[]), "standard_meeting");
    }

    #[test]
    fn no_unsaved_edits_when_not_editing() {
        assert!(!has_unsaved_summary_edits(false, "changed", "original"));
    }

    #[test]
    fn no_unsaved_edits_when_text_matches_saved() {
        assert!(!has_unsaved_summary_edits(true, "same", "same"));
    }

    #[test]
    fn unsaved_edits_when_editing_and_text_differs() {
        assert!(has_unsaved_summary_edits(true, "changed", "original"));
    }

    #[test]
    fn confidence_level_thresholds_match_react() {
        assert_eq!(ConfidenceLevel::for_confidence(1.0), ConfidenceLevel::High);
        assert_eq!(ConfidenceLevel::for_confidence(0.8), ConfidenceLevel::High);
        assert_eq!(ConfidenceLevel::for_confidence(0.79), ConfidenceLevel::Good);
        assert_eq!(ConfidenceLevel::for_confidence(0.7), ConfidenceLevel::Good);
        assert_eq!(ConfidenceLevel::for_confidence(0.69), ConfidenceLevel::Medium);
        assert_eq!(ConfidenceLevel::for_confidence(0.4), ConfidenceLevel::Medium);
        assert_eq!(ConfidenceLevel::for_confidence(0.39), ConfidenceLevel::Low);
        assert_eq!(ConfidenceLevel::for_confidence(0.0), ConfidenceLevel::Low);
    }

    #[test]
    fn confidence_tooltip_matches_react_format() {
        assert_eq!(confidence_tooltip(0.873), "87% confidence - High confidence");
        assert_eq!(confidence_tooltip(0.35), "35% confidence - Low confidence");
    }

    #[test]
    fn format_note_time_renders_valid_rfc3339() {
        let formatted = format_note_time("2026-09-11T14:30:00Z");
        // Exact wall-clock text depends on the local timezone offset, but it
        // must parse to something non-empty in the expected shape.
        assert!(!formatted.is_empty());
    }

    #[test]
    fn format_note_time_is_empty_for_unparseable_input() {
        assert_eq!(format_note_time("not-a-date"), "");
    }

    #[test]
    fn summary_reveal_chars_per_tick_drains_small_buffers_fully() {
        assert_eq!(summary_reveal_chars_per_tick(0), 0);
        assert_eq!(summary_reveal_chars_per_tick(1), 1);
        assert_eq!(summary_reveal_chars_per_tick(2), 2);
    }

    #[test]
    fn summary_reveal_chars_per_tick_scales_with_backlog() {
        // 80 pending chars / 8 target ticks = 10 chars/tick.
        assert_eq!(summary_reveal_chars_per_tick(80), 10);
        // Below the floor of 2 chars/tick, still never exceeds what's pending.
        assert_eq!(summary_reveal_chars_per_tick(3), 2);
        assert_eq!(summary_reveal_chars_per_tick(1), 1);
    }
}
