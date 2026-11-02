//! Authority of a displayed transcript attribution, independent of its name.
pub(crate) const MACHINE: &str = "machine";
pub(crate) const HUMAN: &str = "human";
pub(crate) const LEGACY_UNKNOWN: &str = "legacy_unknown";
pub(crate) fn is_named(label: Option<&str>) -> bool {
    label.is_some_and(|s| {
        !s.trim().is_empty()
            && s != "Unknown"
            && s != "Speaker"
            && !s
                .strip_prefix("Speaker ")
                .is_some_and(|n| !n.is_empty() && n.chars().all(|c| c.is_ascii_digit()))
    })
}
pub(crate) fn protects(origin: &str, label: Option<&str>, profile: Option<&str>) -> bool {
    origin != MACHINE && (origin == HUMAN || profile.is_some() || is_named(label))
}
