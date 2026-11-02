//! Literal, case-insensitive search with highlights in original UTF-8 offsets.
use super::format::TranscriptTurn;
use std::ops::Range;

pub fn matching_turns(turns: Vec<TranscriptTurn>, query: &str) -> Vec<TranscriptTurn> {
    let needle = query.trim().to_lowercase();
    if needle.is_empty() { return turns; }
    turns.into_iter().filter(|turn| turn.body.to_lowercase().contains(&needle)).collect()
}

pub fn highlights(text: &str, query: &str) -> Vec<Range<usize>> {
    let needle = query.trim().to_lowercase();
    if needle.is_empty() { return vec![]; }
    // A Unicode character may expand during lowercasing. Keep each folded byte
    // mapped to its complete original character so highlights never split UTF-8.
    let mut folded = String::new();
    let mut offsets = Vec::new();
    for (start, ch) in text.char_indices() {
        let lower = ch.to_lowercase().to_string();
        offsets.extend(std::iter::repeat_n(start..start + ch.len_utf8(), lower.len()));
        folded.push_str(&lower);
    }
    let mut ranges: Vec<Range<usize>> = Vec::new();
    for (start, _) in folded.match_indices(&needle) {
        let range = offsets[start].start..offsets[start + needle.len() - 1].end;
        if let Some(previous) = ranges.last_mut().filter(|p| p.end > range.start) {
            previous.end = previous.end.max(range.end);
        } else {
            ranges.push(range);
        }
    }
    ranges
}

/// Split overlapping search and speaker-link ranges into non-overlapping runs,
/// preserving both styles where a search match crosses a clickable label.
pub fn styled_ranges(matches: &[Range<usize>], links: &[Range<usize>]) -> Vec<(Range<usize>, bool, bool)> {
    let mut boundaries: Vec<_> = matches.iter().chain(links).flat_map(|r| [r.start, r.end]).collect();
    boundaries.sort_unstable();
    boundaries.dedup();
    boundaries.windows(2).filter_map(|pair| {
        let range = pair[0]..pair[1];
        let matched = matches.iter().any(|r| r.start <= range.start && r.end >= range.end);
        let linked = links.iter().any(|r| r.start <= range.start && r.end >= range.end);
        (matched || linked).then_some((range, matched, linked))
    }).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn highlights_unicode_and_repeated_literal_matches_without_splitting_characters() {
        let text = "ÁRBOL, árbol; İ [x].";
        let found: Vec<_> = highlights(text, " árbol ").into_iter().map(|r| &text[r]).collect();
        assert_eq!(found, vec!["ÁRBOL", "árbol"]);
        let range = highlights(text, "i").pop().unwrap();
        assert_eq!(&text[range], "İ");
        assert_eq!(highlights(text, "[x]").len(), 1);
        assert!(highlights(text, "  ").is_empty());
        assert!(highlights(text, "absent").is_empty());
    }
    #[test]
    fn overlapping_search_and_link_styles_form_non_overlapping_runs() {
        assert_eq!(styled_ranges(&[2..8], &[5..10]), vec![
            (2..5, true, false), (5..8, true, true), (8..10, false, true),
        ]);
        assert!(styled_ranges(&[], &[]).is_empty());
    }
    #[test]
    fn filters_whole_display_turns_preserving_original_indices_and_links() {
        let turn = |body: &str, start| TranscriptTurn { rows: start..start + 2, body: body.into(), speaker_links: vec![(0..3, start + 1)] };
        let turns = vec![turn("Hello world", 0), turn("Another topic", 4)];
        let visible = matching_turns(turns, "HELLO WORLD");
        assert_eq!(visible.len(), 1);
        assert_eq!(visible[0].rows, 0..2);
        assert_eq!(visible[0].speaker_links[0].1, 1);
        assert!(matching_turns(vec![turn("Hello", 0)], "missing").is_empty());
        assert_eq!(matching_turns(vec![turn("Hello", 0)], " ").len(), 1);
    }
}
