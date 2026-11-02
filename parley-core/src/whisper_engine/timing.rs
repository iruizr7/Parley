//! Decoder-timed text units, normally one word. Zero-duration words sharing a
//! neighbour's boundary stay in that small group; no individual times are invented.
#[derive(Debug, Clone)]
pub struct TimedWord {
    pub start: f64,
    pub end: f64,
    pub text: String,
    pub probability: f32,
}

pub(crate) fn words_from_tokens(
    tokens: Vec<(Vec<u8>, i64, i64, f32)>,
    duration: f64,
    text: &str,
) -> Vec<TimedWord> {
    let mut groups: Vec<(Vec<u8>, i64, i64, f32, usize)> = vec![];
    for (bytes, start, end, probability) in tokens {
        if bytes.is_empty() {
            continue;
        }
        if bytes[0].is_ascii_whitespace() || groups.is_empty() {
            groups.push((bytes, start, end, probability, 1));
        } else {
            let group = groups.last_mut().unwrap();
            group.0.extend(bytes);
            group.2 = group.2.max(end);
            group.3 += probability;
            group.4 += 1;
        }
    }
    let mut words = vec![];
    for (bytes, start, end, probability, count) in groups {
        let Ok(text) = String::from_utf8(bytes) else {
            return vec![];
        };
        if text.trim().is_empty() {
            continue;
        }
        let end = (end as f64 / 100.).min(duration);
        let start = start as f64 / 100.;
        if start < 0. || end < start || words.last().is_some_and(|w: &TimedWord| w.start > start) {
            return vec![];
        }
        words.push(TimedWord {
            start,
            end,
            text,
            probability: probability / count as f32,
        });
    }
    // Missing tokens, incomplete UTF-8 or cleanup must not silently remove words.
    let recognized = words.iter().map(|w| w.text.as_str()).collect::<String>();
    if recognized.split_whitespace().ne(text.split_whitespace()) {
        return vec![];
    }
    // Some decoder words consist entirely of tokens at the same instant.
    // Preserve their text with an adjacent, measured interval containing that
    // instant. This is explicitly a multiword unit, not a fabricated word time.
    let mut i = 0;
    while i < words.len() {
        if words[i].end > words[i].start {
            i += 1;
            continue;
        }
        let at = words[i].start;
        if i + 1 < words.len() && words[i + 1].start <= at && words[i + 1].end > at {
            let word = words.remove(i);
            words[i].text.insert_str(0, &word.text);
            words[i].probability = words[i].probability.min(word.probability);
        } else if i > 0 && words[i - 1].start < at && words[i - 1].end >= at {
            let word = words.remove(i);
            words[i - 1].text.push_str(&word.text);
            words[i - 1].probability = words[i - 1].probability.min(word.probability);
        } else if i > 0
            && i + 1 < words.len()
            && words[i - 1].end <= at
            && at <= words[i + 1].start
            && words[i + 1].end > words[i + 1].start
        {
            // The point falls between two measured words. Keep the three
            // words as one bounded phrase, preserving both outer boundaries.
            let missing = words.remove(i);
            let next = words.remove(i);
            let previous = &mut words[i - 1];
            previous.text.push_str(&missing.text);
            previous.text.push_str(&next.text);
            previous.end = next.end;
            previous.probability = previous
                .probability
                .min(missing.probability)
                .min(next.probability);
        } else {
            return vec![];
        }
    }
    words
}

/// Follow the existing text cleanup without throwing away unaffected timing.
/// Cleanup only deletes text/continuation markers; a non-subsequence fails closed.
pub(crate) fn reconcile_cleaned_words(
    words: Vec<TimedWord>,
    cleaned: &str,
    normalize: impl Fn(&str) -> String,
) -> Vec<TimedWord> {
    let source: Vec<_> = words
        .iter()
        .enumerate()
        .flat_map(|(index, word)| {
            normalize(&word.text)
                .split_whitespace()
                .map(|part| (index, part.to_owned()))
                .collect::<Vec<_>>()
        })
        .collect();
    let mut cursor = 0;
    let mut selected: Vec<(usize, TimedWord)> = vec![];
    for part in cleaned.split_whitespace() {
        let Some(offset) = source[cursor..].iter().position(|(_, text)| text == part) else {
            return vec![];
        };
        cursor += offset;
        let index = source[cursor].0;
        if let Some((_, word)) = selected.last_mut().filter(|(i, _)| *i == index) {
            word.text.push(' ');
            word.text.push_str(part);
        } else {
            let mut word = words[index].clone();
            word.text = format!(" {part}");
            selected.push((index, word));
        }
        cursor += 1;
    }
    selected.into_iter().map(|(_, word)| word).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn a_point_between_words_uses_their_measured_phrase_bounds() {
        let words = words_from_tokens(
            vec![
                (b" We".to_vec(), 0, 40, 0.9),
                (b" can".to_vec(), 50, 50, 0.8),
                (b" continue".to_vec(), 60, 90, 0.9),
                (b" now".to_vec(), 90, 120, 0.9),
            ],
            2.,
            "We can continue now",
        );
        assert_eq!(words.len(), 2);
        assert_eq!((words[0].start, words[0].end), (0., 0.9));
        assert_eq!(words[0].text, " We can continue");
        assert_eq!(words[1].text, " now");
    }
    #[test]
    fn zero_duration_word_keeps_the_rest_of_the_chunk_timed() {
        let words = words_from_tokens(
            vec![
                (b" We".to_vec(), 0, 50, 0.9),
                (b" can".to_vec(), 50, 50, 0.8),
                (b" continue".to_vec(), 50, 90, 0.9),
                (b" now".to_vec(), 90, 120, 0.9),
            ],
            2.,
            "We can continue now",
        );
        assert_eq!(words.len(), 3);
        assert_eq!(words[1].text, " can continue");
        assert_eq!((words[1].start, words[1].end), (0.5, 0.9));
        assert_eq!(words[2].start, 0.9);
    }
    #[test]
    fn cleanup_keeps_real_times_through_whitespace_and_deletions() {
        let words = words_from_tokens(
            vec![
                (b" Yes".to_vec(), 0, 20, 0.9),
                (b" Yes".to_vec(), 20, 40, 0.9),
                (b"  continue".to_vec(), 50, 90, 0.9),
            ],
            1.,
            "Yes Yes continue",
        );
        let words = reconcile_cleaned_words(words, "Yes continue", str::to_owned);
        assert_eq!(words.len(), 2);
        assert_eq!((words[1].start, words[1].end), (0.5, 0.9));
        assert_eq!(
            words.iter().map(|w| w.text.as_str()).collect::<String>(),
            " Yes continue"
        );
        assert!(reconcile_cleaned_words(words, "Different words", str::to_owned).is_empty());
    }
    #[test]
    fn backwards_word_times_fall_back_to_the_source_segment() {
        assert!(words_from_tokens(
            vec![
                (b" first".to_vec(), 50, 70, 0.9),
                (b" second".to_vec(), 20, 30, 0.9),
            ],
            1.,
            "first second"
        )
        .is_empty());
    }
    #[test]
    fn joins_utf8_tokens_and_punctuation_without_inventing_times() {
        let w = words_from_tokens(
            vec![
                (b" caf\xc3".to_vec(), 10, 30, 0.9),
                (b"\xa9".to_vec(), 30, 40, 0.8),
                (b",".to_vec(), 40, 40, 0.9),
                (b" please".to_vec(), 50, 90, 0.9),
            ],
            1.,
            "café, please",
        );
        assert_eq!(w.len(), 2);
        assert_eq!(w[0].text, " café,");
        assert_eq!(w[0].end, 0.4);
        assert_eq!(w[1].start, 0.5);
    }
    #[test]
    fn incomplete_alignment_preserves_parent_text_via_fallback() {
        assert!(words_from_tokens(vec![(b" hello".to_vec(), 0, 0, 0.9)], 1., "hello").is_empty());
        assert!(
            words_from_tokens(vec![(b" hello".to_vec(), 0, 50, 0.9)], 1., "hello world").is_empty()
        );
    }
}
