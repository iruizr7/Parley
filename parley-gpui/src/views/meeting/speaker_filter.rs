//! View-only speaker filtering, preserving original transcript indices.
use super::format::{transcript_turns, TranscriptTurn};
use parley_core::database::models::MeetingTranscript;
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum SpeakerKey {
    Profile(String),
    Label(String),
    Unknown,
}
pub fn key(row: &MeetingTranscript) -> SpeakerKey {
    if let Some(id) = &row.voice_profile_id {
        SpeakerKey::Profile(id.clone())
    } else if let Some(label) = row.speaker.as_deref().filter(|s| !s.trim().is_empty()) {
        SpeakerKey::Label(label.to_owned())
    } else {
        SpeakerKey::Unknown
    }
}

pub struct SpeakerOption {
    pub key: SpeakerKey,
    pub name: String,
    pub segments: usize,
}
pub fn options(rows: &[MeetingTranscript]) -> Vec<SpeakerOption> {
    let mut options = BTreeMap::new();
    for row in rows {
        let key = key(row);
        let option = options.entry(key.clone()).or_insert_with(|| SpeakerOption {
            key,
            name: row.speaker.as_deref().filter(|s| !s.trim().is_empty())
                .unwrap_or("Unknown speaker").to_owned(),
            segments: 0,
        });
        option.segments += 1;
    }
    let mut options: Vec<_> = options.into_values().collect();
    options.sort_by(|a, b| a.name.to_lowercase().cmp(&b.name.to_lowercase()).then(a.key.cmp(&b.key)));
    options
}

/// Empty selection means all speakers. Filtering occurs before display grouping
/// so an interrupting speaker's words remain independently selectable. Never
/// join across hidden rows: a playback span must not bridge hidden speakers.
pub fn turns(rows: &[MeetingTranscript], selected: &BTreeSet<SpeakerKey>) -> Vec<TranscriptTurn> {
    let mut turns = Vec::new();
    let matches = |row: &MeetingTranscript| selected.is_empty() || selected.contains(&key(row));
    let mut index = 0;
    while index < rows.len() {
        if !matches(&rows[index]) {
            index += 1;
            continue;
        }
        let start = index;
        while index < rows.len() && matches(&rows[index]) { index += 1; }
        for mut turn in transcript_turns(&rows[start..index]) {
            turn.rows = (turn.rows.start + start)..(turn.rows.end + start);
            for (_, row) in &mut turn.speaker_links { *row += start; }
            turns.push(turn);
        }
    }
    turns
}

#[cfg(test)]
mod tests {
    use super::*;
    fn row(id: &str, speaker: Option<&str>, profile: Option<&str>, start: f64) -> MeetingTranscript {
        MeetingTranscript {
            id: id.into(), text: format!("Words from {id}"), timestamp: String::new(),
            speaker: speaker.map(str::to_owned), voice_profile_id: profile.map(str::to_owned),
            audio_start_time: Some(start), audio_end_time: Some(start + 1.), duration: Some(1.),
            overlapping_speakers: vec![], source: None, confidence: None,
        }
    }
    #[test]
    fn filter_keeps_original_rows_and_never_joins_across_hidden_speech() {
        let rows = vec![row("a", Some("Alice"), Some("alice"), 0.), row("b", Some("Bob"), None, 1.), row("c", Some("Alice"), Some("alice"), 2.)];
        let visible = turns(&rows, &BTreeSet::from([key(&rows[0])]));
        assert_eq!(visible.iter().map(|t| t.rows.clone()).collect::<Vec<_>>(), vec![0..1, 2..3]);
        assert!(visible.iter().all(|t| !t.body.contains("Words from b")));
        assert_eq!(turns(&rows, &BTreeSet::new()).len(), transcript_turns(&rows).len());
        assert_eq!(options(&rows)[0].segments, 2);
    }
    #[test]
    fn interruptions_are_selectable_and_inline_links_keep_original_indices() {
        let mut rows = vec![row("hidden", None, None, 0.), row("a", Some("Alice"), None, 1.), row("b", Some("Bob"), None, 1.5)];
        rows[2].overlapping_speakers = vec!["Alice".into(), "Bob".into()];
        let visible = turns(&rows, &BTreeSet::from([key(&rows[2])]));
        assert_eq!(visible[0].rows, 2..3);
        assert_eq!(visible[0].body, "Words from b");
        let both = turns(&rows, &BTreeSet::from([key(&rows[1]), key(&rows[2])]));
        assert_eq!(both[0].rows, 1..3);
        assert_eq!(both[0].speaker_links[0].1, 2);
        // Overlap metadata alone must never attribute Alice's words to Bob.
        rows[1].overlapping_speakers = vec!["Bob".into()];
        assert_eq!(turns(&rows, &BTreeSet::from([key(&rows[2])])).len(), 1);
    }
    #[test]
    fn keys_survive_profile_renames_and_distinguish_same_names_and_unknowns() {
        let mut rows = vec![row("a", Some("Alex"), Some("one"), 0.), row("b", Some("Alex"), Some("two"), 1.), row("c", None, None, 2.)];
        assert_eq!(options(&rows).len(), 3);
        let selected = BTreeSet::from([key(&rows[0]), SpeakerKey::Unknown]);
        rows[0].speaker = Some("Renamed".into());
        assert_eq!(turns(&rows, &selected).iter().map(|t| t.rows.clone()).collect::<Vec<_>>(), vec![0..1, 2..3]);
        assert!(turns(&[], &selected).is_empty());
    }
}
