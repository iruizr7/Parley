//! Exercise the same pipeline APIs as the desktop against private fixtures.
//! Usage: pipeline_validation request.json response.json
//! Requests/results may contain private connection details, text and embeddings;
//! keep them outside the checkout. This tool never opens the default app database.
use anyhow::{anyhow, bail, Context, Result};
use parley_core::audio::processing::{config, enrollment, executor, preparation, runs, service};
use serde_json::{json, Value};
use std::{path::PathBuf, sync::atomic::AtomicBool};

fn required<'a>(value: &'a Value, key: &str) -> Result<&'a str> {
    value[key].as_str().ok_or_else(|| anyhow!("Missing {key}"))
}

fn progress(stage: &str, percent: u32, _: &str) {
    eprintln!("{stage}: {percent}%");
}

#[tokio::main]
async fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    anyhow::ensure!(
        args.len() == 3,
        "Usage: pipeline_validation request.json response.json"
    );
    let request: Value = serde_json::from_slice(&std::fs::read(&args[1])?)?;
    let target: config::ProcessingTarget = serde_json::from_value(request["target"].clone())?;
    if target.kind == config::TargetKind::Local {
        // Resolve existing model assets without starting the application or
        // opening its database. Offline VAD shares the speaker model directory.
        parley_core::whisper_engine::models::set_models_directory();
        parley_core::speaker_diarization::model::set_models_dir();
    }
    let cancelled = AtomicBool::new(false);
    let started = std::time::Instant::now();
    let action = required(&request, "action")?;
    let result: Result<Value> = async {
        match action {
            "probe" => executor::probe(&target).await,
            "install" => executor::install(&target, progress).await,
            "rollback" => executor::rollback(&target, progress).await,
            "install_bundle" => executor::install_bundle(&target, &PathBuf::from(required(&request,"bundle_path")?), progress).await,
            "prepare" => executor::prepare(&target, &serde_json::from_value(request["config"].clone())?, progress).await,
            "execute" => executor::execute(&target, request["request"].clone(), &cancelled, progress).await,
            "resume" => executor::resume(&target, required(&request,"request_id")?, &cancelled, progress).await,
            "resume_with_audio" => executor::resume_with_audio(&target, required(&request,"request_id")?, &PathBuf::from(required(&request,"audio_path")?), &cancelled, progress).await,
            "status" => executor::status(&target, required(&request, "request_id")?).await,
            "cancel" => executor::cancel(&target, required(&request, "request_id")?).await,
            "cleanup" => executor::cleanup(&target, required(&request, "request_id")?).await,
            "process" | "identify" | "redetect" | "confirm_identity" | "confirm_profile" | "confirm_and_enroll" | "profile_details" | "remove_sample" | "delete_enrollment" | "prepare_pipeline" => {
                let database = PathBuf::from(required(&request, "database")?);
                anyhow::ensure!(database.is_absolute(), "Use an absolute, isolated validation database");
                let app_data = parley_core::paths::app_data_dir().map_err(|e| anyhow!(e))?;
                let app_data = app_data.canonicalize().unwrap_or(app_data);
                let resolved = if database.exists() { database.canonicalize()? } else {
                    database.parent().context("Missing database parent")?.canonicalize()?.join(database.file_name().context("Missing database name")?)
                };
                anyhow::ensure!(!resolved.starts_with(&app_data), "Validation must not open an application database");
                let db = parley_core::database::manager::DatabaseManager::new(database.to_str().context("Non-UTF8 database path")?).await?;
                let pool = db.pool();
                config::save_target(pool, &target).await?;
                let pipeline: config::PipelineConfig = serde_json::from_value(request["config"].clone())?;
                let meeting = request["meeting_id"].as_str().map(str::to_owned).unwrap_or_else(|| format!("validation-{}", uuid::Uuid::new_v4()));
                if action == "process" {
                    let source = PathBuf::from(required(&request, "audio_path")?).canonicalize()?;
                    let fixture = database.parent().context("Missing fixture parent")?.join(&meeting);
                    std::fs::create_dir_all(&fixture)?;
                    let audio = fixture.join(format!("audio.{}", source.extension().and_then(|s| s.to_str()).unwrap_or("wav")));
                    if !audio.exists() {
                        // A link avoids copying potentially large private recordings.
                        #[cfg(unix)]
                        std::os::unix::fs::symlink(&source, &audio)?;
                        #[cfg(not(unix))]
                        std::fs::copy(&source, &audio)?;
                    }
                    sqlx::query("INSERT INTO meetings(id,title,created_at,updated_at,folder_path) VALUES(?,'Pipeline validation',datetime('now'),datetime('now'),?) ON CONFLICT(id) DO NOTHING")
                        .bind(&meeting).bind(fixture.to_string_lossy().as_ref()).execute(pool).await?;
                }
                let options = service::ProcessingOptions { config: pipeline.clone(), language: request["language"].as_str().map(str::to_owned), num_speakers: request["num_speakers"].as_i64().unwrap_or(0) as i32 };
                let outcome = match action {
                    "process" => serde_json::to_value(service::process_recording(pool, &meeting, options, &cancelled, progress).await?)?,
                    "redetect" => serde_json::to_value(service::redetect_speakers(pool, &meeting, options, &cancelled, progress).await?)?,
                    "prepare_pipeline" => {
                        preparation::prepare_pipeline(&target, &pipeline, progress).await?;
                        json!({"prepared":true})
                    }
                    "confirm_identity" => {
                        let group=required(&request,"group_version")?;
                        let speaker=required(&request,"speaker")?;
                        let profile=required(&request,"profile_id")?;
                        if let Some(current)=request.get("expected_current") {
                            service::confirm_identity_replacement(pool,&meeting,group,speaker,profile,serde_json::from_value(current.clone())?).await?;
                        } else {
                            service::confirm_identity_suggestion(pool,&meeting,group,speaker,profile).await?;
                        }
                        json!({"confirmed":true})
                    }
                    "confirm_profile" => serde_json::to_value(enrollment::confirm_profile(pool, &meeting, required(&request,"speaker")?, required(&request,"name")?, None, request["profile_id"].as_str()).await?)?,
                    "confirm_and_enroll" => {
                        let profile=required(&request,"profile_id")?;
                        let speaker=required(&request,"speaker")?;
                        let group=runs::current_run(pool,&meeting).await?;
                        enrollment::confirm_enrollment_source(pool,profile,&meeting,speaker,group.as_deref()).await?;
                        enrollment::enroll_from_meeting(pool,profile,&meeting,speaker,pipeline.identification,&target.id,&cancelled,progress).await?;
                        json!({"enrolled":true})
                    }
                    "profile_details" => {
                        let profile=required(&request,"profile_id")?;
                        let enrollments=enrollment::list_enrollments(pool,profile).await?;
                        let mut details=vec![];
                        for item in enrollments {
                            let samples=enrollment::list_samples(pool,profile,&item.space_id).await?;
                            let mut playback=vec![];
                            for sample in &samples {
                                if sample.available {playback.push(enrollment::source_playback(pool,&sample.id).await?);}
                            }
                            details.push(json!({"enrollment":item,"samples":samples,"playback":playback}));
                        }
                        json!(details)
                    }
                    "remove_sample" => {enrollment::remove_sample(pool,required(&request,"sample_id")?).await?;json!({"removed":true})}
                    "delete_enrollment" => {enrollment::delete_enrollment(pool,required(&request,"profile_id")?,required(&request,"space_id")?).await?;json!({"removed":true})}
                    _ => serde_json::to_value(service::identify_speakers(pool, &meeting, service::IdentificationOptions { identification: pipeline.identification, target_id: target.id.clone(), replace_confirmed: request["replace_confirmed"].as_bool().unwrap_or(false) }, &cancelled, progress).await?)?,
                };
                Ok(json!({"meeting_id":meeting,"outcome":outcome,"runs":runs::list_runs(pool,&meeting).await?,"analysis":runs::load(pool,&meeting).await?}))
            }
            _ => bail!("Unknown validation action"),
        }
    }.await;
    let elapsed = started.elapsed().as_secs_f64();
    let response = match &result {
        Ok(value) => json!({"ok":true,"elapsed_seconds":elapsed,"result":value}),
        Err(error) => json!({"ok":false,"elapsed_seconds":elapsed,"error":format!("{error:#}")}),
    };
    std::fs::write(&args[2], serde_json::to_vec_pretty(&response)?)?;
    println!(
        "{action}: {} ({elapsed:.2}s)",
        if result.is_ok() {
            "succeeded"
        } else {
            "failed"
        }
    );
    result.map(|_| ())
}
