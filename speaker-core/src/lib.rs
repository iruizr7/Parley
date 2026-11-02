//! Portable speaker algorithms shared by Parley and its headless worker.
use anyhow::{anyhow, bail, Result};
use serde::{Deserialize, Serialize};
use std::path::Path;
pub mod embedder;
pub mod embedding_math;
pub mod offline;
pub use embedder::SpeakerEmbedder;
pub const LEGACY_NATIVE_EMBEDDING: &str =
    "campplus:3dspeaker_speech_campplus_sv_en_voxceleb_16k.onnx";
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeakerEmbedding {
    pub speaker: String,
    pub model_id: String,
    pub vector: Vec<f32>,
    pub clean_seconds: f64,
    /// Number of windows represented by this centroid (not audio seconds).
    #[serde(default = "one_sample")]
    pub sample_count: usize,
    #[serde(default)]
    pub embedding_space: Option<EmbeddingSpace>,
    #[serde(default)]
    pub source_intervals: Vec<SourceInterval>,
}
fn one_sample() -> usize {
    1
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceInterval {
    pub speaker: String,
    pub start: f64,
    pub end: f64,
    #[serde(default)]
    pub channel: u32,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingSpace {
    pub id: String,
    pub model: String,
    pub revision: String,
    pub preprocessing: String,
    pub normalization: String,
    pub aggregation: String,
    pub extraction_version: String,
    pub dimension: usize,
}

pub fn extract_clean_embeddings(
    samples: &[f32],
    model: &Path,
    intervals: &[SourceInterval],
) -> Result<Vec<SpeakerEmbedding>> {
    let embedder = SpeakerEmbedder::from_path(&model, 1)?;
    use sha2::Digest;
    let model_hash = format!("{:x}", sha2::Sha256::digest(std::fs::read(&model)?));
    let mut result = vec![];
    let speakers: std::collections::BTreeSet<_> =
        intervals.iter().map(|i| i.speaker.clone()).collect();
    for speaker in speakers {
        let mut windows: Vec<_> = intervals
            .iter()
            .filter(|i| {
                i.speaker == speaker
                    && i.channel == 0
                    && i.start.is_finite()
                    && i.end.is_finite()
                    && i.start >= 0.
                    && i.end - i.start >= 3.
            })
            .cloned()
            .collect();
        windows.sort_by(|a, b| (b.end - b.start).total_cmp(&(a.end - a.start)));
        let mut vectors = vec![];
        let mut sources = vec![];
        let mut clean = 0.;
        for w in windows.into_iter().take(3) {
            let start = ((w.start + 0.1) * 16000.) as usize;
            let end = ((w.end.min(w.start + 10.) - 0.1) * 16000.) as usize;
            if end <= start || end - start < 48000 {
                continue;
            }
            if let Some(slice) = samples.get(start..end) {
                let Ok(mut v) = embedder.embed(slice) else {
                    continue;
                };
                if v.is_empty()
                    || v.iter().any(|x| !x.is_finite())
                    || v.iter().map(|x| x * x).sum::<f32>() < 0.001
                {
                    continue;
                }
                embedding_math::l2_normalize(&mut v);
                vectors.push(v);
                clean += (end - start) as f64 / 16000.;
                sources.push(SourceInterval {
                    speaker: speaker.clone(),
                    start: start as f64 / 16000.,
                    end: end as f64 / 16000.,
                    channel: 0,
                });
            }
        }
        if !vectors.is_empty() {
            let vector = embedding_math::average_and_normalize(&vectors);
            let dim = vector.len();
            result.push(SpeakerEmbedding {
                speaker,
                model_id: LEGACY_NATIVE_EMBEDDING.into(),
                vector,
                clean_seconds: clean,
                sample_count: vectors.len(),
                embedding_space: Some(EmbeddingSpace {
                    id: campplus_space_id(&model_hash),
                    model: "campplus".into(),
                    revision: format!("sha256:{model_hash}"),
                    preprocessing: "pcm16k-mono-ffmpeg-v1".into(),
                    normalization: "l2".into(),
                    aggregation: "longest-three-trimmed-10s-v1".into(),
                    extraction_version: "parley-native-v2".into(),
                    dimension: dim,
                }),
                source_intervals: sources,
            });
        }
    }
    Ok(result)
}
pub fn campplus_space_id(model_hash: &str) -> String {
    format!("campplus:sha256:{model_hash}:pcm16k-mono-ffmpeg-v1:l2:longest-three-trimmed-10s-v1:parley-native-v2")
}

/// Strict normalized worker input: little-endian PCM16, 16 kHz, one channel.
pub fn read_pcm16_wav(path: &Path) -> Result<Vec<f32>> {
    use std::io::{Read, Seek, SeekFrom};
    let mut file = std::fs::File::open(path)?;
    let mut header = [0u8; 12];
    file.read_exact(&mut header)?;
    if &header[..4] != b"RIFF" || &header[8..] != b"WAVE" {
        bail!("Native speaker input must be PCM16 WAV");
    }
    let mut valid_format = false;
    let mut result = None;
    loop {
        let mut chunk = [0u8; 8];
        if file.read_exact(&mut chunk).is_err() {
            break;
        }
        let size = u32::from_le_bytes(chunk[4..8].try_into().unwrap()) as usize;
        match &chunk[..4] {
            b"fmt " => {
                if size < 16 || size > 4096 {
                    bail!("Invalid WAV format");
                }
                let mut fmt = vec![0u8; size];
                file.read_exact(&mut fmt)?;
                valid_format = u16::from_le_bytes(fmt[..2].try_into().unwrap()) == 1
                    && u16::from_le_bytes(fmt[2..4].try_into().unwrap()) == 1
                    && u32::from_le_bytes(fmt[4..8].try_into().unwrap()) == 16000
                    && u16::from_le_bytes(fmt[14..16].try_into().unwrap()) == 16;
            }
            b"data" => {
                if !valid_format || size % 2 != 0 || size > 16000 * 2 * 7200 {
                    bail!("Native input requires mono16k PCM16, at most two hours");
                }
                let mut pcm = vec![0u8; size];
                file.read_exact(&mut pcm)?;
                result = Some(
                    pcm.chunks_exact(2)
                        .map(|c| i16::from_le_bytes([c[0], c[1]]) as f32 / 32768.)
                        .collect(),
                );
                break;
            }
            _ => {
                file.seek(SeekFrom::Current(size as i64))?;
            }
        }
        if size % 2 != 0 {
            file.seek(SeekFrom::Current(1))?;
        }
    }
    result.ok_or_else(|| anyhow!("Normalized WAV contains no audio samples"))
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OverlapInterval {
    pub start: f64,
    pub end: f64,
    pub speakers: Vec<String>,
}
pub fn overlap_intervals(turns: &[SourceInterval]) -> Vec<OverlapInterval> {
    let mut events: Vec<_> = turns
        .iter()
        .flat_map(|t| {
            [
                (t.start, 1, t.speaker.clone()),
                (t.end, -1, t.speaker.clone()),
            ]
        })
        .collect();
    events.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
    let mut active = std::collections::BTreeMap::<String, i32>::new();
    let mut previous = 0.;
    let mut overlaps = vec![];
    for (at, delta, speaker) in events {
        let speakers: Vec<_> = active
            .iter()
            .filter(|(_, n)| **n > 0)
            .map(|(s, _)| s.clone())
            .collect();
        if at > previous && speakers.len() > 1 {
            overlaps.push(OverlapInterval {
                start: previous,
                end: at,
                speakers,
            });
        }
        *active.entry(speaker).or_default() += delta;
        previous = at;
    }
    overlaps
}
pub fn clean_intervals(
    turns: &[SourceInterval],
    overlaps: &[OverlapInterval],
) -> Vec<SourceInterval> {
    let mut grouped: std::collections::BTreeMap<String, Vec<(f64, f64)>> =
        std::collections::BTreeMap::new();
    for t in turns {
        grouped
            .entry(t.speaker.clone())
            .or_default()
            .push((t.start, t.end));
    }
    let mut result = vec![];
    for (speaker, mut turns) in grouped {
        turns.sort_by(|a, b| a.0.total_cmp(&b.0));
        let mut merged: Vec<(f64, f64)> = vec![];
        for (s, e) in turns {
            if let Some(last) = merged.last_mut().filter(|last| s <= last.1) {
                last.1 = last.1.max(e);
            } else {
                merged.push((s, e));
            }
        }
        for (start, end) in merged {
            let mut spans = vec![(start, end)];
            for o in overlaps {
                spans = spans
                    .into_iter()
                    .flat_map(|(s, e)| {
                        if o.end <= s || o.start >= e {
                            vec![(s, e)]
                        } else {
                            vec![(s, o.start.max(s)), (o.end.min(e), e)]
                        }
                    })
                    .collect();
            }
            result.extend(
                spans
                    .into_iter()
                    .filter(|(s, e)| e - s >= 3.)
                    .map(|(start, end)| SourceInterval {
                        speaker: speaker.clone(),
                        start,
                        end,
                        channel: 0,
                    }),
            );
        }
    }
    result
}

pub fn runtime_info() -> std::collections::BTreeMap<&'static str, String> {
    std::collections::BTreeMap::from([
        (
            "adapter_package_version",
            env!("CARGO_PKG_VERSION").to_owned(),
        ),
        ("sherpa_version", sherpa_onnx::version().to_owned()),
        ("sherpa_git_sha1", sherpa_onnx::git_sha1().to_owned()),
        ("sherpa_git_date", sherpa_onnx::git_date().to_owned()),
    ])
}
