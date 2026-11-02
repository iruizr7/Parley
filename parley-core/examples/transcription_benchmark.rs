//! Run against an already-extracted private sample; does not open the app DB.
//! cargo run --release -p parley-core --features vulkan --example transcription_benchmark -- sample.wav result.json [model] [language]
#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args: Vec<_> = std::env::args().collect();
    anyhow::ensure!(
        args.len() >= 3,
        "usage: transcription_benchmark sample.wav result.json [model] [language] | --speech-only [model] [language] [device]"
    );
    let result = if args.get(3).map(String::as_str) == Some("--speech-only") {
        parley_core::whisper_engine::models::set_models_directory();
        parley_core::speaker_diarization::model::set_models_dir();
        parley_core::audio::processing::native::recognize_local(
            std::path::Path::new(&args[1]), args.get(4).map(String::as_str).unwrap_or("large-v3-turbo"),
            args.get(5).cloned(), args.get(6).map(String::as_str).unwrap_or("auto"), &std::sync::atomic::AtomicBool::new(false),
            |stage, percent, _| eprintln!("{stage}: {percent}%"),
        ).await?
    } else if args.get(3).map(String::as_str) == Some("--request") {
        let request: serde_json::Value = serde_json::from_slice(&std::fs::read(
            args.get(4)
                .ok_or_else(|| anyhow::anyhow!("--request needs a JSON file"))?,
        )?)?;
        let target: parley_core::audio::processing::config::ProcessingTarget =
            serde_json::from_value(request["target"].clone())?;
        let options: parley_core::audio::processing::service::ProcessingOptions =
            serde_json::from_value(request["options"].clone())?;
        let cancel = std::sync::atomic::AtomicBool::new(false);
        parley_core::audio::processing::service::analyze_audio(
            &target,
            &options,
            std::path::Path::new(&args[1]),
            &cancel,
            |stage, percent, _| eprintln!("{stage}: {percent}%"),
        )
        .await?
    } else {
        parley_core::audio::processing::benchmark::benchmark_native(
            std::path::Path::new(&args[1]),
            args.get(3)
                .map(String::as_str)
                .unwrap_or("large-v3-turbo-q5_0"),
            args.get(4).cloned().unwrap_or_else(|| "es".into()),
        )
        .await?
    };
    std::fs::write(&args[2], serde_json::to_vec_pretty(&result)?)?;
    println!(
        "Transcription benchmark: {:.1}s audio, {:.1}s processing, {} segments, {} speaker turns",
        result.duration,
        result.elapsed_seconds,
        result.segments.len(),
        result.turns.len()
    );
    Ok(())
}
