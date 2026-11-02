//! Portable native speaker adapter: no capture, GUI, app database or ASR.
use anyhow::{anyhow, bail, Result};
use serde_json::{json, Value};
use speaker_core::{offline::diarize_offline, SourceInterval};
use std::path::{Path, PathBuf};
fn main() -> Result<()> {
    let arguments: Vec<_> = std::env::args().collect();
    if arguments.iter().any(|a| a == "--probe") {
        println!(
            "{}",
            json!({"version":1,"protocol_version":1,"kind":"native_speaker","device":"cpu","runtime_ready":true,"runtime":speaker_core::runtime_info(),"platform":{"os":std::env::consts::OS,"architecture":std::env::consts::ARCH}})
        );
        return Ok(());
    }

    let argument = |name: &str| -> Result<PathBuf> {
        arguments
            .iter()
            .position(|s| s == name)
            .and_then(|i| arguments.get(i + 1))
            .map(PathBuf::from)
            .ok_or_else(|| anyhow!("Missing {name}"))
    };
    let request: Value = serde_json::from_slice(&std::fs::read(argument("--request")?)?)?;
    if request["version"] != 1 {
        bail!("Unsupported native speaker protocol");
    }
    if request["speaker_device"]
        .as_str()
        .is_some_and(|d| d != "cpu")
    {
        bail!("The native speaker package requires CPU execution");
    }
    let audio = Path::new(
        request["audio_path"]
            .as_str()
            .ok_or_else(|| anyhow!("Missing normalized audio"))?,
    );
    let samples = speaker_core::read_pcm16_wav(audio)?;
    if samples.is_empty() {
        bail!("Empty audio");
    }
    let embedding = Path::new(
        request["native_models"]["embedding"]
            .as_str()
            .ok_or_else(|| anyhow!("Missing embedding artifact"))?,
    );
    let count = request["num_speakers"].as_i64().unwrap_or(0);
    if !(0..=64).contains(&count) {
        bail!("Speaker count must be Auto or 1–64");
    }
    let (turns, overlaps, intervals) = match request["operation"].as_str().unwrap_or("") {
        "speaker_analysis" | "diarize" => {
            let segmentation = Path::new(
                request["native_models"]["segmentation"]
                    .as_str()
                    .ok_or_else(|| anyhow!("Missing segmentation artifact"))?,
            );
            let turns: Vec<_> =
                diarize_offline(&samples, segmentation, embedding, count as i32, 4)?
                    .into_iter()
                    .map(|t| SourceInterval {
                        speaker: format!("Speaker {}", t.speaker + 1),
                        start: t.start as f64,
                        end: t.end as f64,
                        channel: 0,
                    })
                    .collect();
            let overlaps = speaker_core::overlap_intervals(&turns);
            let intervals = speaker_core::clean_intervals(&turns, &overlaps);
            (turns, overlaps, intervals)
        }
        "extract_embeddings" | "embed" => {
            let spans: Vec<SourceInterval> =
                serde_json::from_value(request["source_intervals"].clone())?;
            let duration = samples.len() as f64 / 16000.;
            if spans.iter().any(|s| {
                !s.start.is_finite()
                    || !s.end.is_finite()
                    || s.start < 0.
                    || s.end <= s.start
                    || s.end > duration + 0.05
                    || s.channel != 0
            }) {
                bail!("Invalid normalized evidence intervals");
            }
            (vec![], vec![], spans)
        }
        _ => bail!("Unsupported native speaker operation"),
    };
    let embeddings = speaker_core::extract_clean_embeddings(&samples, embedding, &intervals)?;
    let output = json!({"version":1,"duration":samples.len()as f64/16000.,"language":request["language"],"request_id":request["request_id"],"audio_sha256":request["audio_sha256"],"segments":request.get("segments").cloned().unwrap_or(json!([])),"words":[],"turns":turns,"overlaps":overlaps,"embeddings":embeddings,"diarization_device":"cpu","settings":{"runtime":"parley-native-v2","native_runtime":speaker_core::runtime_info(),"preprocessing":"pcm16k-mono-ffmpeg-v1","package_version":env!("CARGO_PKG_VERSION")}});
    std::fs::write(argument("--response")?, serde_json::to_vec(&output)?)?;
    Ok(())
}
