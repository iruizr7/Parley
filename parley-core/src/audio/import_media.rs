//! Prepare a small audio-only import before copying it into the meeting folder.
//! The saved file is also the upload/recovery source: no second transport hash
//! or disposable remote payload is needed. Never modify the user's original.
use anyhow::{bail, Context, Result};
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::atomic::{AtomicBool, Ordering};
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::Command;

pub(super) struct PreparedAudio {
    pub path: PathBuf,
    _temporary: Option<tempfile::TempDir>,
}

fn lossless_pcm(codec: &str) -> bool {
    // FLAC preserves these integer samples exactly, without resampling or
    // channel mixing. Float and 32-bit PCM stay unchanged: reducing them to
    // the 24-bit input supported by older FLAC encoders would not be lossless.
    matches!(
        codec,
        "pcm_s16le" | "pcm_s16be" | "pcm_s24le" | "pcm_s24be" | "pcm_u8" | "pcm_s8"
    )
}

fn audio_codec(diagnostics: &str) -> Option<&str> {
    diagnostics
        .lines()
        .filter(|line| line.trim_start().starts_with("Stream #0:"))
        .find_map(|line| {
            line.split_once("Audio: ")
                .and_then(|(_, rest)| rest.split([',', ' ']).next())
        })
}

pub(super) async fn prepare(
    source: &Path,
    duration: f64,
    cancelled: &AtomicBool,
    progress: impl Fn(&str, u32, &str),
) -> Result<PreparedAudio> {
    let original = || PreparedAudio {
        path: source.to_owned(),
        _temporary: None,
    };
    if cancelled.load(Ordering::Relaxed) {
        bail!("Import cancelled");
    }
    let extension = source
        .extension()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    let container = matches!(extension.as_str(), "mp4" | "m4a" | "mkv" | "webm" | "wma");
    if !container && extension != "wav" {
        return Ok(original());
    }
    let ffmpeg = match super::ffmpeg::find_ffmpeg_path() {
        Some(path) => path,
        // Audio-only WAV can still be imported without optional compression.
        None if !container => return Ok(original()),
        None => bail!("FFmpeg is required to extract audio before importing this media. Install FFmpeg and retry."),
    };
    progress(
        "audio_inspection",
        0,
        "Inspecting the audio stream before saving the recording",
    );
    // Use the same available FFmpeg executable, without adding an ffprobe
    // dependency. A zero-duration output reads stream headers, not the meeting.
    let mut probe = Command::new(&ffmpeg);
    probe
        .args(["-hide_banner", "-nostdin", "-i"])
        .arg(source)
        .args([
            "-map", "0:a:0", "-t", "0", "-vn", "-sn", "-dn", "-f", "null", "-",
        ])
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .kill_on_drop(true);
    let output = probe.output();
    tokio::pin!(output);
    let output = loop {
        tokio::select! {
            output = &mut output => break output.context("Cannot inspect the audio stream")?,
            _ = tokio::time::sleep(std::time::Duration::from_millis(100)) => {
                if cancelled.load(Ordering::Relaxed) { bail!("Import cancelled"); }
            }
        }
    };
    let diagnostics = String::from_utf8_lossy(&output.stderr);
    let codec =
        audio_codec(&diagnostics).context("The selected media has no readable audio stream")?;
    if !output.status.success() {
        bail!("Cannot inspect the selected audio stream; the original recording was kept");
    }
    let compress = lossless_pcm(codec);
    if !container && !compress {
        return Ok(original());
    }
    let temporary = tempfile::tempdir()?;
    let path = temporary
        .path()
        .join(if compress { "audio.flac" } else { "audio.mkv" });
    let (stage, message) = if compress {
        (
            "audio_compression",
            "Compressing audio losslessly with FLAC before saving",
        )
    } else {
        (
            "audio_extraction",
            "Extracting audio without re-encoding; leaving video out of the meeting copy",
        )
    };
    progress(stage, 0, message);
    let mut cmd = Command::new(&ffmpeg);
    cmd.args([
        "-hide_banner",
        "-nostdin",
        "-nostats",
        "-loglevel",
        "error",
        "-progress",
        "pipe:2",
        "-n",
        "-i",
    ])
    .arg(source)
    .args([
        "-map",
        "0:a:0",
        "-vn",
        "-sn",
        "-dn",
        "-map_metadata",
        "-1",
        "-map_chapters",
        "-1",
        "-c:a",
    ])
    .arg(if compress { "flac" } else { "copy" });
    if compress {
        cmd.args(["-compression_level", "5"]);
    }
    cmd.arg(&path)
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .kill_on_drop(true);
    let mut child = cmd.spawn().context("Cannot start audio extraction")?;
    let mut lines = BufReader::new(
        child
            .stderr
            .take()
            .context("Missing extraction progress stream")?,
    )
    .lines();
    let mut last = 0;
    loop {
        tokio::select! {
            line = lines.next_line() => {
                let Some(line) = line? else { break; };
                if let Some(micros) = line.strip_prefix("out_time_us=").and_then(|v| v.parse::<u64>().ok()) {
                    if duration.is_finite() && duration > 0. {
                        let percent = ((micros as f64 / 1_000_000. / duration * 100.) as u32).min(99);
                        if percent > last { progress(stage, percent, message); last = percent; }
                    }
                }
            }
            _ = tokio::time::sleep(std::time::Duration::from_millis(100)) => {
                if cancelled.load(Ordering::Relaxed) {
                    child.kill().await?;
                    child.wait().await?;
                    bail!("Import cancelled during audio extraction");
                }
            }
        }
    }
    if !child.wait().await?.success() || !path.is_file() || path.metadata()?.len() == 0 {
        bail!("Audio extraction failed; the original recording was kept");
    }
    if cancelled.load(Ordering::Relaxed) {
        bail!("Import cancelled");
    }
    progress(stage, 100, message);
    // Noise/tiny clips may not benefit from FLAC. Keep the original audio-only
    // file in that case; video containers must still lose their video stream.
    if !container && path.metadata()?.len() >= source.metadata()?.len() {
        return Ok(original());
    }
    Ok(PreparedAudio {
        path,
        _temporary: Some(temporary),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ffmpeg(args: &[&str], output: &Path) {
        let result = std::process::Command::new(super::super::ffmpeg::find_ffmpeg_path().unwrap())
            .args(["-hide_banner", "-nostdin", "-loglevel", "error"])
            .args(args)
            .arg(output)
            .output()
            .unwrap();
        assert!(
            result.status.success(),
            "Synthetic media generation failed: {}",
            String::from_utf8_lossy(&result.stderr)
        );
    }

    fn audio_bytes(path: &Path, compressed: bool) -> Vec<u8> {
        let mut cmd = std::process::Command::new(super::super::ffmpeg::find_ffmpeg_path().unwrap());
        cmd.args(["-v", "error", "-i"])
            .arg(path)
            .args(["-map", "0:a:0"]);
        if compressed {
            cmd.args(["-c:a", "copy", "-f", "hash", "-hash", "sha256"]);
        } else {
            cmd.args(["-c:a", "pcm_s32le", "-f", "s32le"]);
        }
        let output = cmd.arg("-").output().unwrap();
        assert!(output.status.success());
        output.stdout
    }

    #[tokio::test]
    async fn video_extraction_preserves_compressed_audio_packets_and_removes_video() {
        if super::super::ffmpeg::find_ffmpeg_path().is_none() {
            return;
        }
        let dir = tempfile::tempdir().unwrap();
        let source = dir.path().join("fictional-video.mkv");
        ffmpeg(
            &[
                "-f",
                "lavfi",
                "-i",
                "testsrc2=size=128x128:rate=10:duration=2",
                "-f",
                "lavfi",
                "-i",
                "sine=frequency=440:duration=2",
                "-c:v",
                "ffv1",
                "-c:a",
                "libopus",
            ],
            &source,
        );
        let original = std::fs::read(&source).unwrap();
        let mut events = Vec::new();
        // Callback is shared so the same function is usable by the importer.
        let observed = std::sync::Mutex::new(&mut events);
        let prepared = prepare(&source, 2., &AtomicBool::new(false), |stage, percent, _| {
            observed.lock().unwrap().push((stage.to_owned(), percent));
        })
        .await
        .unwrap();
        assert_ne!(prepared.path, source);
        assert!(prepared.path.metadata().unwrap().len() < source.metadata().unwrap().len());
        assert_eq!(
            audio_bytes(&source, true),
            audio_bytes(&prepared.path, true)
        );
        let probe = std::process::Command::new(super::super::ffmpeg::find_ffmpeg_path().unwrap())
            .args(["-hide_banner", "-i"])
            .arg(&prepared.path)
            .output()
            .unwrap();
        assert!(!String::from_utf8_lossy(&probe.stderr).contains("Video:"));
        assert!(events.contains(&("audio_extraction".into(), 100)));
        let temporary = prepared.path.clone();
        drop(prepared);
        assert!(!temporary.exists());
        assert_eq!(std::fs::read(&source).unwrap(), original);
    }

    #[tokio::test]
    async fn flac_keeps_pcm_samples_sample_rate_and_channels_exactly() {
        if super::super::ffmpeg::find_ffmpeg_path().is_none() {
            return;
        }
        let dir = tempfile::tempdir().unwrap();
        for codec in ["pcm_s16le", "pcm_s24le"] {
            let source = dir.path().join(format!("{codec}.wav"));
            ffmpeg(
                &[
                    "-f",
                    "lavfi",
                    "-i",
                    "sine=frequency=440:sample_rate=48000:duration=2",
                    "-ac",
                    "2",
                    "-c:a",
                    codec,
                ],
                &source,
            );
            let prepared = prepare(&source, 2., &AtomicBool::new(false), |_, _, _| {})
                .await
                .unwrap();
            assert_eq!(prepared.path.extension().unwrap(), "flac");
            assert!(prepared.path.metadata().unwrap().len() < source.metadata().unwrap().len());
            assert_eq!(
                audio_bytes(&source, false),
                audio_bytes(&prepared.path, false)
            );
        }
    }

    #[tokio::test]
    async fn float_audio_is_not_quantized_and_cancellation_keeps_original() {
        if super::super::ffmpeg::find_ffmpeg_path().is_none() {
            return;
        }
        let dir = tempfile::tempdir().unwrap();
        let source = dir.path().join("float.wav");
        ffmpeg(
            &["-f", "lavfi", "-i", "sine=duration=1", "-c:a", "pcm_f32le"],
            &source,
        );
        let prepared = prepare(&source, 1., &AtomicBool::new(false), |_, _, _| {})
            .await
            .unwrap();
        assert_eq!(prepared.path, source);
        let original = std::fs::read(&source).unwrap();
        assert!(prepare(&source, 1., &AtomicBool::new(true), |_, _, _| {})
            .await
            .is_err());
        assert_eq!(std::fs::read(&source).unwrap(), original);
    }

    #[tokio::test]
    async fn video_without_audio_fails_without_modifying_the_original() {
        if super::super::ffmpeg::find_ffmpeg_path().is_none() {
            return;
        }
        let dir = tempfile::tempdir().unwrap();
        let source = dir.path().join("silent-video.mkv");
        ffmpeg(
            &[
                "-f",
                "lavfi",
                "-i",
                "color=duration=1:size=16x16",
                "-c:v",
                "ffv1",
            ],
            &source,
        );
        let original = std::fs::read(&source).unwrap();
        assert!(prepare(&source, 1., &AtomicBool::new(false), |_, _, _| {})
            .await
            .is_err());
        assert_eq!(std::fs::read(&source).unwrap(), original);
    }

    #[test]
    fn only_exactly_representable_pcm_is_compressed() {
        for codec in ["pcm_s16le", "pcm_s24le", "pcm_s24be", "pcm_u8"] {
            assert!(lossless_pcm(codec));
        }
        for codec in [
            "opus",
            "aac",
            "mp3",
            "flac",
            "pcm_f32le",
            "pcm_f64le",
            "pcm_s32le",
        ] {
            assert!(!lossless_pcm(codec));
        }
        assert_eq!(audio_codec("  Stream #0:0: Video: h264\n  Stream #0:1: Audio: opus, 48000 Hz, mono\n  Stream #1:0: Audio: pcm_s16le, 48000 Hz"), Some("opus"));
    }
}
