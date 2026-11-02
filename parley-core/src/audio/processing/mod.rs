//! Configurable saved-recording processing and immutable provenance.
pub(crate) mod attribution;
pub mod benchmark;
pub mod config;
pub mod enrollment;
mod evidence;
pub mod executor;
pub mod identification;
pub mod native;
pub mod preparation;
mod recovery;
pub mod runs;
pub mod service;
#[cfg(test)]
mod tests;
pub mod types;

#[cfg(test)]
mod legacy_analysis_tests;

pub mod identity_candidates;
