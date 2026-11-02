//! Pure identity context shared by legacy live and versioned saved matching.
//! Legacy live baselines, ranking and source-aware refinement remain explicit
//! caller policies; sharing context does not make their embedding spaces equal.
pub(crate) const IDENTIFICATION_POLICY_VERSION: &str = "conservative-review-required-v4";
/// An uninvited non-self identity needs this additional similarity. Legacy
/// live uses its existing baseline; qualified saved spaces use theirs.
pub(crate) const NON_ATTENDEE_PENALTY: f32 = 0.15;
pub(crate) const AMBIGUITY_MARGIN: f32 = 0.10;

/// Preserve the legacy convention: only a loaded self profile establishes a
/// case-insensitive, trimmed same-name alias. Names never merge identities or
/// authorize comparisons between embedding spaces.
pub(crate) fn is_self_alias(
    profile_id: &str,
    profile_name: Option<&str>,
    self_id: Option<&str>,
    self_name: Option<&str>,
) -> bool {
    let Some(self_id) = self_id else { return false };
    profile_id == self_id
        || profile_name.is_some_and(|name| {
            self_name.is_some_and(|own_name| {
                name.trim().to_lowercase() == own_name.trim().to_lowercase()
            })
        })
}
pub(crate) fn is_unexpected(is_self: bool, expected: Option<bool>) -> bool {
    !is_self && expected == Some(false)
}
pub(crate) fn effective_threshold(baseline: f32, is_self: bool, expected: Option<bool>) -> f32 {
    baseline
        + if is_unexpected(is_self, expected) {
            NON_ATTENDEE_PENALTY
        } else {
            0.
        }
}
pub(crate) fn saved_baseline(model: &str) -> f32 {
    if model.eq_ignore_ascii_case("campplus") {
        0.75
    } else if model.eq_ignore_ascii_case("wespeaker") {
        0.60
    } else {
        0.80
    }
}
pub(crate) fn saved_threshold(model: &str, is_self: bool, expected: Option<bool>) -> f32 {
    effective_threshold(saved_baseline(model), is_self, expected)
}

/// Select by attendee-adjusted margin, then require raw separation from every
/// finite comparable competitor, even one below its own eligibility threshold.
pub(crate) fn saved_choice(scores: &[(f32, f32)]) -> Option<usize> {
    let (index, (score, threshold)) = scores
        .iter()
        .enumerate()
        .filter(|(_, (score, threshold))| score.is_finite() && threshold.is_finite())
        .max_by(|(ai, a), (bi, b)| (a.0 - a.1).total_cmp(&(b.0 - b.1)).then(bi.cmp(ai)))?;
    (*score >= *threshold
        && scores.iter().enumerate().all(|(other, (value, _))| {
            other == index || !value.is_finite() || score - value >= AMBIGUITY_MARGIN
        }))
    .then_some(index)
}
