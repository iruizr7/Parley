use super::types::*;
use anyhow::Result;
use sqlx::{SqliteConnection, SqlitePool};

pub async fn load(pool: &SqlitePool, meeting: &str) -> Result<Option<Analysis>> {
    let row: Option<String> = sqlx::query_scalar(
        "SELECT analysis_json FROM meeting_transcription_analysis WHERE meeting_id = ?",
    )
    .bind(meeting)
    .fetch_optional(pool)
    .await?;
    row.map(|r| serde_json::from_str(&r).map_err(Into::into))
        .transpose()
}

pub async fn store_tx(
    conn: &mut SqliteConnection,
    meeting: &str,
    analysis: Option<&Analysis>,
) -> Result<(), sqlx::Error> {
    if let Some(analysis) = analysis {
        analysis
            .validate()
            .map_err(|e| sqlx::Error::Protocol(e.to_string()))?;
        let json =
            serde_json::to_string(analysis).map_err(|e| sqlx::Error::Protocol(e.to_string()))?;
        sqlx::query("INSERT INTO meeting_transcription_analysis VALUES (?, ?) ON CONFLICT(meeting_id) DO UPDATE SET analysis_json=excluded.analysis_json")
            .bind(meeting).bind(&json).execute(&mut *conn).await?;
        sqlx::query("UPDATE meetings SET duration_seconds=?,updated_at=datetime('now'),status='completed',completed_at=COALESCE(completed_at,datetime('now')) WHERE id=?").bind(analysis.duration).bind(meeting).execute(&mut *conn).await?;
        let run_id = analysis.settings["request_id"]
            .as_str()
            .map(str::to_owned)
            .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());
        let now = chrono::Utc::now().to_rfc3339();
        let requested = analysis
            .settings
            .get("config")
            .unwrap_or(&analysis.settings);
        let resolved = serde_json::json!({"target":analysis.settings.get("target_snapshot"),"runtime":analysis.provenance,"worker_settings":redact_metadata(&analysis.settings),"recognition_seconds":analysis.recognition_seconds,"diarization_seconds":analysis.diarization_seconds,"recognition_device":analysis.recognition_device,"diarization_device":analysis.diarization_device,"compute_type":analysis.compute_type,"elapsed_seconds":analysis.elapsed_seconds});
        let input = serde_json::json!({"audio_sha256":analysis.settings.get("audio_sha256"),"duration":analysis.duration,"language":analysis.language,"detected_language":analysis.language,"requested_language":analysis.settings.get("requested_language"),"request_snapshot_available":analysis.settings.get("requested_language").is_some(),"num_speakers":analysis.settings.get("num_speakers"),"expected_speaker_count":analysis.settings.get("num_speakers")});
        sqlx::query("INSERT INTO processing_runs(id,meeting_id,operation,started_at,finished_at,outcome,requested_json,resolved_json,input_json,result_json,error) VALUES(?,?,?,?,?,?,?,?,?,?,?)")
            .bind(&run_id)
            .bind(meeting)
            .bind(analysis.settings["operation"].as_str().unwrap_or("process"))
            .bind(analysis.settings["started_at"].as_str().unwrap_or(&now))
            .bind(&now)
            .bind("succeeded")
            .bind(requested.to_string())
            .bind(resolved.to_string())
            .bind(input.to_string())
            .bind(
                redact_metadata(
                    &serde_json::to_value(analysis)
                        .map_err(|e| sqlx::Error::Protocol(e.to_string()))?,
                )
                .to_string(),
            )
            .bind(Option::<String>::None)
            .execute(&mut *conn)
            .await?;
        sqlx::query("INSERT INTO meeting_processing_state VALUES(?,?,?) ON CONFLICT(meeting_id) DO UPDATE SET current_run_id=excluded.current_run_id,group_version=excluded.group_version").bind(meeting).bind(&run_id).bind(&run_id).execute(&mut *conn).await?;
        sqlx::query("DELETE FROM speaker_group_embeddings WHERE meeting_id=?")
            .bind(meeting)
            .execute(&mut *conn)
            .await?;
        for e in &analysis.embeddings {
            if let Some(space) = &e.embedding_space {
                sqlx::query("INSERT INTO speaker_group_embeddings VALUES(?,?,?,?,?,?)")
                    .bind(meeting)
                    .bind(&run_id)
                    .bind(&e.speaker)
                    .bind(&space.id)
                    .bind(
                        serde_json::to_string(e)
                            .map_err(|e| sqlx::Error::Protocol(e.to_string()))?,
                    )
                    .bind(analysis.provenance.to_string())
                    .execute(&mut *conn)
                    .await?;
            }
        }
    } else {
        sqlx::query("DELETE FROM meeting_transcription_analysis WHERE meeting_id = ?")
            .bind(meeting)
            .execute(&mut *conn)
            .await?;
        let id = uuid::Uuid::new_v4().to_string();
        let now = chrono::Utc::now().to_rfc3339();
        let rows: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM transcripts WHERE meeting_id=?")
            .bind(meeting)
            .fetch_one(&mut *conn)
            .await?;
        if rows > 0 {
            sqlx::query("INSERT INTO processing_runs VALUES(?,?,?,?,?,'succeeded','{}',?,'{}',NULL,NULL)").bind(&id).bind(meeting).bind("local_preview_or_legacy_native").bind(&now).bind(&now).bind(serde_json::json!({"provenance":"unknown","source":"legacy native path; no inferred artifact precision"}).to_string()).execute(&mut *conn).await?;
            sqlx::query("INSERT INTO meeting_processing_state VALUES(?,?,?) ON CONFLICT(meeting_id) DO UPDATE SET current_run_id=excluded.current_run_id,group_version=excluded.group_version").bind(meeting).bind(&id).bind(&id).execute(&mut *conn).await?;
        }
    }
    Ok(())
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ProcessingRun {
    pub id: String,
    pub meeting_id: Option<String>,
    pub operation: String,
    pub started_at: String,
    pub finished_at: String,
    pub outcome: String,
    pub requested: serde_json::Value,
    pub resolved: serde_json::Value,
    pub input: serde_json::Value,
    pub error: Option<String>,
}
pub async fn list_runs(pool: &SqlitePool, meeting: &str) -> Result<Vec<ProcessingRun>> {
    use sqlx::Row;
    let rows = sqlx::query(
        "SELECT * FROM processing_runs WHERE meeting_id=? ORDER BY started_at DESC,id DESC",
    )
    .bind(meeting)
    .fetch_all(pool)
    .await?;
    let mut result: Vec<ProcessingRun> = rows
        .into_iter()
        .map(|r| {
            Ok(ProcessingRun {
                id: r.get("id"),
                meeting_id: r.get("meeting_id"),
                operation: r.get("operation"),
                started_at: r.get("started_at"),
                finished_at: r.get("finished_at"),
                outcome: r.get("outcome"),
                requested: serde_json::from_str(r.get("requested_json"))?,
                resolved: serde_json::from_str(r.get("resolved_json"))?,
                input: serde_json::from_str(r.get("input_json"))?,
                error: r.get("error"),
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let identities = sqlx::query("SELECT * FROM identification_runs WHERE meeting_id=?")
        .bind(meeting)
        .fetch_all(pool)
        .await?;
    for r in identities {
        let value: serde_json::Value = serde_json::from_str(r.get("result_json"))?;
        let created: String = r.get("created_at");
        result.push(ProcessingRun{id:r.get("id"),meeting_id:r.get("meeting_id"),operation:"identification".into(),started_at:created.clone(),finished_at:created,outcome:value["outcome"].as_str().unwrap_or("succeeded").into(),requested:serde_json::from_str(r.get("request_json"))?,input:serde_json::json!({"processing_run_id":r.get::<Option<String>,_>("processing_run_id"),"group_version":r.get::<Option<String>,_>("group_version"),"matching_policy":r.get::<String,_>("policy_version")}),error:value["error"].as_str().map(str::to_owned),resolved:value});
    }
    result.sort_by(|a, b| {
        b.started_at
            .cmp(&a.started_at)
            .then_with(|| b.id.cmp(&a.id))
    });
    Ok(result)
}

pub async fn current_run(pool: &SqlitePool, meeting: &str) -> Result<Option<String>> {
    Ok(
        sqlx::query_scalar(
            "SELECT current_run_id FROM meeting_processing_state WHERE meeting_id=?",
        )
        .bind(meeting)
        .fetch_optional(pool)
        .await?
        .flatten(),
    )
}
pub async fn record_failure(
    pool: &SqlitePool,
    meeting: &str,
    operation: &str,
    requested: &serde_json::Value,
    error: &str,
    cancelled: bool,
) -> Result<String> {
    let id = uuid::Uuid::new_v4().to_string();
    let now = chrono::Utc::now().to_rfc3339();
    sqlx::query("INSERT INTO processing_runs(id,meeting_id,operation,started_at,finished_at,outcome,requested_json,resolved_json,input_json,result_json,error) VALUES(?,?,?,?,?,?,?,?,?,?,?)")
        .bind(&id)
        .bind(meeting)
        .bind(operation)
        .bind(requested["started_at"].as_str().unwrap_or(&now))
        .bind(&now)
        .bind(if cancelled { "cancelled" } else { "failed" })
        .bind(serde_json::to_string(requested)?)
        .bind("{}")
        .bind("{}")
        .bind(Option::<String>::None)
        .bind(error)
        .execute(pool)
        .await?;
    Ok(id)
}
/// Public export metadata excludes target connection paths and biometric data.
pub fn redact_metadata(value: &serde_json::Value) -> serde_json::Value {
    match value {
        serde_json::Value::Object(o) => serde_json::Value::Object(
            o.iter()
                .filter(|(k, _)| {
                    ![
                        "ssh_host",
                        "worker_path",
                        "python",
                        "assets_dir",
                        "audio_path",
                        "vector",
                        "embedding_json",
                        "resolved_policy",
                        "candidate_contexts",
                        "expected_profile_ids",
                    ]
                    .contains(&k.as_str())
                })
                .map(|(k, v)| (k.clone(), redact_metadata(v)))
                .collect(),
        ),
        serde_json::Value::Array(a) => a.iter().map(redact_metadata).collect(),
        v => v.clone(),
    }
}

pub async fn audio_hash(path: std::path::PathBuf) -> Result<String> {
    audio_hash_with_progress(path, |_| {}).await
}

/// Hash every byte as usual, emitting at most one update per integer percentage.
/// Completion is reported only after the complete read succeeds.
pub async fn audio_hash_with_progress(
    path: std::path::PathBuf,
    progress: impl Fn(u8) + Send + Sync,
) -> Result<String> {
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
    let task = tokio::task::spawn_blocking(move || -> Result<String> {
        let progress = |percent| { let _ = tx.send(percent); };
        use sha2::Digest;
        use std::io::Read;
        let mut input = std::fs::File::open(path)?;
        let total = input.metadata()?.len();
        let mut read = 0u128;
        let mut last_percent = 0u8;
        progress(0);
        let mut hash = sha2::Sha256::new();
        let mut bytes = [0u8; 64 * 1024];
        loop {
            let n = input.read(&mut bytes)?;
            if n == 0 {
                break;
            }
            hash.update(&bytes[..n]);
            read += n as u128;
            // Reserve 100 for EOF, including when a file grows during the read.
            let percent = if total > 0 {
                (read * 100 / u128::from(total)).min(99) as u8
            } else {
                0
            };
            if percent > last_percent {
                progress(percent);
                last_percent = percent;
            }
        }
        let digest = format!("{:x}", hash.finalize());
        progress(100);
        Ok(digest)
    });
    while let Some(percent) = rx.recv().await { progress(percent); }
    task.await?
}

/// Refresh the ordinary transcript/meeting files from the committed database.
/// Runtime and private target details are deliberately absent from exports.
pub(crate) async fn refresh_exports(pool: &SqlitePool, meeting: &str) -> Result<()> {
    use sqlx::Row;
    let folder: Option<String> = sqlx::query_scalar("SELECT folder_path FROM meetings WHERE id=?")
        .bind(meeting)
        .fetch_optional(pool)
        .await?
        .flatten();
    let Some(folder) = folder.filter(|f| !f.is_empty()) else {
        return Ok(());
    };
    let rows=sqlx::query("SELECT id,transcript,timestamp,audio_start_time,audio_end_time,duration,speaker,voice_profile_id,sequence_id,source,confidence FROM transcripts WHERE meeting_id=? ORDER BY COALESCE(sequence_id,9223372036854775807),audio_start_time,id").bind(meeting).fetch_all(pool).await?;
    let segments: Vec<_> = rows
        .into_iter()
        .map(|r| crate::audio::common::TranscriptSegment {
            id: r.get("id"),
            text: r.get("transcript"),
            timestamp: r.get("timestamp"),
            audio_start_time: r.get("audio_start_time"),
            audio_end_time: r.get("audio_end_time"),
            duration: r.get("duration"),
            display_time: None,
            confidence: r.get("confidence"),
            sequence_id: r.get::<Option<i64>, _>("sequence_id").map(|n| n as u64),
            speaker: r.get("speaker"),
            voice_profile_id: r.get("voice_profile_id"),
            source: r.get("source"),
        })
        .collect();
    let path = std::path::Path::new(&folder);
    crate::audio::common::write_transcripts_json(path, &segments)?;
    let metadata = crate::audio::common::MeetingMetadata {
        meeting_id: Some(meeting.into()),
        status: Some("completed".into()),
        transcript_file: Some("transcripts.json".into()),
        ..Default::default()
    };
    crate::audio::common::write_metadata(path, &metadata, || metadata.clone())?;
    Ok(())
}
