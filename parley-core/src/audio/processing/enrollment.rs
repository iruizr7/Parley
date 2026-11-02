use super::config::NATIVE_EMBEDDING;
use super::runs::load;
use anyhow::{bail, Result};
use sqlx::SqlitePool;

/// Called only after a human names/links a saved speaker group. The embedding
/// persists with the meeting, so enrollment still works after an app restart.
pub async fn confirm_profile(
    pool: &SqlitePool,
    meeting: &str,
    speaker: &str,
    name: &str,
    email: Option<&str>,
    target: Option<&str>,
) -> Result<Option<(String, u64)>> {
    let Some(analysis) = load(pool, meeting).await? else {
        return Ok(None);
    };
    let attributed:Vec<(f64,f64)>=sqlx::query_as("SELECT audio_start_time,audio_end_time FROM transcripts WHERE meeting_id=? AND speaker=? AND audio_end_time>audio_start_time").bind(meeting).bind(speaker).fetch_all(pool).await?;
    let groups = source_groups(&analysis, &attributed, speaker);
    let other_spans:Vec<(f64,f64)>=sqlx::query_as("SELECT audio_start_time,audio_end_time FROM transcripts WHERE meeting_id=? AND COALESCE(speaker,'')!=? AND audio_end_time>audio_start_time").bind(meeting).bind(speaker).fetch_all(pool).await?;
    let allowed_clean = confirmed_cached_intervals(&analysis, &attributed, &other_spans, &groups);
    let embeddings: Vec<_> = analysis
        .embeddings
        .iter()
        .filter(|e| e.speaker == speaker || groups.contains(e.speaker.as_str()))
        .filter(|e| {
            e.embedding_space.is_none()
                || e.source_intervals.iter().all(|i| {
                    allowed_clean.iter().any(|allowed| {
                        allowed.speaker == i.speaker
                            && i.start >= allowed.start
                            && i.end <= allowed.end
                    })
                })
        })
        .collect();
    let group_version = super::runs::current_run(pool, meeting)
        .await?
        .unwrap_or_else(|| "legacy_unknown".into());
    let mut tx = pool.begin().await?;
    let present: bool = sqlx::query_scalar(
        "SELECT EXISTS(SELECT 1 FROM transcripts WHERE meeting_id=? AND speaker=?)",
    )
    .bind(meeting)
    .bind(speaker)
    .fetch_one(&mut *tx)
    .await?;
    if !present {
        bail!("Speaker label has changed; reload the meeting before naming it");
    }
    let id = if let Some(id) = target {
        let exists: bool =
            sqlx::query_scalar("SELECT EXISTS(SELECT 1 FROM voice_profiles WHERE id=?)")
                .bind(id)
                .fetch_one(&mut *tx)
                .await?;
        if !exists {
            bail!("Voice profile no longer exists");
        }
        id.to_owned()
    } else {
        let existing: Option<String> = sqlx::query_scalar(
            "SELECT id FROM voice_profiles WHERE name = ? COLLATE NOCASE LIMIT 1",
        )
        .bind(name)
        .fetch_optional(&mut *tx)
        .await?;
        if let Some(id) = existing {
            id
        } else {
            let id = format!("voice-{}", uuid::Uuid::new_v4());
            // An identity without CAM++ evidence must not pretend to contain
            // a CAM++ vector. Community evidence lives in the typed table.
            sqlx::query("INSERT INTO voice_profiles (id,name,email,embedding,embedding_dim,sample_count,is_self,created_at,updated_at) VALUES (?,?,?,X'',0,0,0,datetime('now'),datetime('now'))")
                .bind(&id).bind(name).bind(email).execute(&mut *tx).await?;
            id
        }
    };
    // Reassigning a named group corrects this meeting only. Revoke pooled
    // evidence that included the corrected lines before enrolling the new person.
    let previous: Vec<(String, f64, f64)> = sqlx::query_as(
        "SELECT voice_profile_id,audio_start_time,audio_end_time FROM transcripts WHERE meeting_id=? AND speaker=? AND voice_profile_id IS NOT NULL AND voice_profile_id!=? AND audio_end_time>audio_start_time",
    ).bind(meeting).bind(speaker).bind(&id).fetch_all(&mut *tx).await?;
    let mut revoked = std::collections::BTreeMap::<String, Vec<(f64, f64)>>::new();
    for (profile, start, end) in previous {
        revoked.entry(profile).or_default().push((start, end));
    }
    for (profile, spans) in revoked {
        revoke_evidence_tx(&mut tx, &profile, meeting, &spans).await?;
    }
    for e in embeddings {
        store_evidence_tx(
            &mut tx,
            &id,
            meeting,
            Some(&group_version),
            analysis
                .audio_sha256
                .as_deref()
                .or_else(|| analysis.settings["audio_sha256"].as_str()),
            e,
        )
        .await?;
        if e.embedding_space.is_some() {
            continue;
        }
        sqlx::query("INSERT INTO profile_enrollments(profile_id,space_id,revision,status,embedding_json,clean_seconds,updated_at,extraction_version) VALUES(?,?,?,'legacy_unknown',?,?,datetime('now'),'unknown') ON CONFLICT(profile_id,space_id) DO UPDATE SET embedding_json=excluded.embedding_json,clean_seconds=excluded.clean_seconds,revision=excluded.revision,updated_at=excluded.updated_at").bind(&id).bind(format!("legacy:{}",e.model_id)).bind(uuid::Uuid::new_v4().to_string()).bind(serde_json::to_string(&e.vector)?).bind(e.clean_seconds).execute(&mut *tx).await?;
        // Keep the longest confirmed clean sample rather than adapting from
        // short/noisy snippets and gradually drifting toward another voice.
        let learned = sqlx::query("INSERT INTO voice_profile_embeddings VALUES (?,?,?,?) ON CONFLICT(profile_id,model_id) DO UPDATE SET embedding_json=excluded.embedding_json,clean_seconds=excluded.clean_seconds WHERE excluded.clean_seconds > voice_profile_embeddings.clean_seconds")
            .bind(&id).bind(&e.model_id).bind(serde_json::to_string(&e.vector)?).bind(e.clean_seconds).execute(&mut *tx).await?.rows_affected() > 0;
        if e.model_id == NATIVE_EMBEDDING && learned {
            let bytes: Vec<u8> = e.vector.iter().flat_map(|v| v.to_le_bytes()).collect();
            sqlx::query("UPDATE voice_profiles SET embedding=?,embedding_dim=?,sample_count=?,updated_at=datetime('now') WHERE id=?")
                .bind(bytes).bind(e.vector.len() as i64).bind(e.sample_count.max(1) as i64).bind(&id)
                .execute(&mut *tx).await?;
        }
    }
    let label: String = sqlx::query_scalar(
        "SELECT CASE WHEN is_self=1 THEN 'Me' ELSE name END FROM voice_profiles WHERE id=?",
    )
    .bind(&id)
    .fetch_one(&mut *tx)
    .await?;
    let changed = sqlx::query(
        "UPDATE transcripts SET speaker=?,voice_profile_id=?,identity_origin='human' WHERE meeting_id=? AND speaker=?",
    )
    .bind(label)
    .bind(&id)
    .bind(meeting)
    .bind(speaker)
    .execute(&mut *tx)
    .await?
    .rows_affected();
    let confirmed_groups: Vec<&str> = if groups.is_empty() {
        vec![speaker]
    } else {
        groups.iter().map(String::as_str).collect()
    };
    for group in confirmed_groups {
        sqlx::query("INSERT INTO speaker_identity_decisions VALUES(?,?,?,?,1,NULL,datetime('now')) ON CONFLICT(meeting_id,group_version,speaker) DO UPDATE SET profile_id=excluded.profile_id,confirmed=1,updated_at=excluded.updated_at").bind(meeting).bind(&group_version).bind(group).bind(&id).execute(&mut *tx).await?;
    }
    sqlx::query(
        "INSERT INTO identity_decision_events VALUES(?,?,?,?,?,'confirmed',datetime('now'),?)",
    )
    .bind(uuid::Uuid::new_v4().to_string())
    .bind(meeting)
    .bind(&group_version)
    .bind(speaker)
    .bind(&id)
    .bind(
        serde_json::json!({"source":"explicit_speaker_naming","profile_name_snapshot":name})
            .to_string(),
    )
    .execute(&mut *tx)
    .await?;
    recover_confirmed_evidence_tx(&mut tx, &id).await?;
    tx.commit().await?;
    if let Err(error) = super::runs::refresh_exports(pool, meeting).await {
        log::warn!("Committed speaker export could not be refreshed: {error}");
    }
    Ok(Some((id, changed)))
}

use super::{config::Identification, types::*};
use serde::{Deserialize, Serialize};
use sqlx::{Row, SqliteConnection};
use std::sync::atomic::AtomicBool;
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Enrollment {
    pub profile_id: String,
    pub space_id: String,
    pub revision: String,
    pub status: String,
    pub clean_seconds: f64,
    pub updated_at: String,
    pub extraction_version: String,
    pub sample_count: usize,
    pub embedding_space: Option<EmbeddingSpace>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnrollmentSample {
    pub id: String,
    pub profile_id: String,
    pub space_id: String,
    pub source_meeting_id: Option<String>,
    pub audio_sha256: Option<String>,
    pub channel: u32,
    pub intervals: Vec<SourceInterval>,
    pub source_run_id: Option<String>,
    pub confirmation: serde_json::Value,
    pub clean_seconds: f64,
    pub available: bool,
    /// Only independent, numerically usable source evidence contributes.
    #[serde(default)]
    pub contributes: bool,
    #[serde(default)]
    pub duplicate_of: Option<String>,
    /// usable, duplicate_reference, invalid_embedding, invalid_intervals,
    /// insufficient_clean_evidence; unknown for older serialized DTOs.
    #[serde(default = "unknown_quality_status")]
    pub quality_status: String,
}
fn unknown_quality_status() -> String {
    "unknown".into()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlaybackSource {
    pub meeting_id: String,
    pub start: f64,
    pub end: f64,
    pub channel: u32,
    /// Extractors currently use the normalized mono mix, not an arbitrary original channel.
    pub downmix: bool,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnrollmentSource {
    pub meeting_id: String,
    pub title: String,
    pub speaker: String,
    pub group_version: Option<String>,
}
pub async fn list_enrollments(pool: &SqlitePool, profile: &str) -> Result<Vec<Enrollment>> {
    let rows =
        sqlx::query("SELECT * FROM profile_enrollments WHERE profile_id=? ORDER BY space_id")
            .bind(profile)
            .fetch_all(pool)
            .await?;
    let mut result = vec![];
    for r in rows {
        let space_id: String = r.get("space_id");
        let embedding_space: Option<EmbeddingSpace> =
            serde_json::from_str(r.get("space_json")).ok();
        let samples=sqlx::query_as::<_,super::evidence::EvidenceRow>(&format!("SELECT {} FROM enrollment_samples WHERE profile_id=? AND space_id=? ORDER BY created_at,id",super::evidence::SAMPLE_COLUMNS))
            .bind(profile).bind(&space_id).fetch_all(pool).await?;
        let quality = super::evidence::assess(
            &samples,
            embedding_space.as_ref().map(|space| space.dimension),
        );
        result.push(Enrollment {
            profile_id: r.get("profile_id"),
            space_id,
            revision: r.get("revision"),
            status: r.get("status"),
            clean_seconds: r.get("clean_seconds"),
            updated_at: r.get("updated_at"),
            extraction_version: r.get("extraction_version"),
            sample_count: quality.contribution_count,
            embedding_space,
        });
    }
    Ok(result)
}
pub async fn list_samples(
    pool: &SqlitePool,
    profile: &str,
    space: &str,
) -> Result<Vec<EnrollmentSample>> {
    let samples=sqlx::query_as::<_,super::evidence::EvidenceRow>(&format!("SELECT {} FROM enrollment_samples WHERE profile_id=? AND space_id=? ORDER BY created_at,id",super::evidence::SAMPLE_COLUMNS))
        .bind(profile).bind(space).fetch_all(pool).await?;
    let metadata: Option<String> = sqlx::query_scalar(
        "SELECT space_json FROM profile_enrollments WHERE profile_id=? AND space_id=?",
    )
    .bind(profile)
    .bind(space)
    .fetch_optional(pool)
    .await?;
    let dimension = metadata
        .and_then(|json| serde_json::from_str::<EmbeddingSpace>(&json).ok())
        .map(|space| space.dimension);
    let quality = super::evidence::assess(&samples, dimension);
    let rows=sqlx::query("SELECT s.*,m.folder_path FROM enrollment_samples s LEFT JOIN meetings m ON s.source_meeting_id=m.id WHERE s.profile_id=? AND s.space_id=? ORDER BY s.created_at,s.id").bind(profile).bind(space).fetch_all(pool).await?;
    rows.into_iter()
        .map(|r| {
            let folder: Option<String> = r.get("folder_path");
            let available = folder
                .and_then(|f| {
                    crate::audio::retranscription::find_audio_file(std::path::Path::new(&f)).ok()
                })
                .is_some();
            let id: String = r.get("id");
            let assessment = quality.quality.get(&id);
            Ok(EnrollmentSample {
                contributes: assessment.is_some_and(|q| q.contributes),
                duplicate_of: assessment.and_then(|q| q.duplicate_of.clone()),
                quality_status: assessment.map(|q| q.status).unwrap_or("unknown").into(),
                id,
                profile_id: r.get("profile_id"),
                space_id: r.get("space_id"),
                source_meeting_id: r.get("source_meeting_id"),
                audio_sha256: r.get("audio_sha256"),
                channel: r.get::<i64, _>("channel") as u32,
                intervals: serde_json::from_str(r.get("intervals_json"))?,
                source_run_id: r.get("source_run_id"),
                confirmation: serde_json::from_str(r.get("confirmation_json"))?,
                clean_seconds: r.get("clean_seconds"),
                available,
            })
        })
        .collect()
}
pub async fn source_playback(pool: &SqlitePool, sample_id: &str) -> Result<Vec<PlaybackSource>> {
    let r=sqlx::query("SELECT s.*,m.folder_path FROM enrollment_samples s LEFT JOIN meetings m ON m.id=s.source_meeting_id WHERE s.id=?").bind(sample_id).fetch_optional(pool).await?.ok_or_else(||anyhow::anyhow!("Sample no longer exists"))?;
    let meeting: Option<String> = r.get("source_meeting_id");
    let meeting = meeting
        .ok_or_else(|| anyhow::anyhow!("Sample unavailable: source recording was removed"))?;
    let folder: Option<String> = r.get("folder_path");
    let path = folder
        .and_then(|f| crate::audio::retranscription::find_audio_file(std::path::Path::new(&f)).ok())
        .ok_or_else(|| anyhow::anyhow!("Sample unavailable: source recording was removed"))?;
    if let Some(hash) = r.get::<Option<String>, _>("audio_sha256") {
        let current = super::runs::audio_hash(path.clone()).await?;
        if current != hash {
            bail!("Sample unavailable: source recording has changed");
        }
    }
    let intervals: Vec<SourceInterval> = serde_json::from_str(r.get("intervals_json"))?;
    Ok(intervals
        .into_iter()
        .map(|i| PlaybackSource {
            meeting_id: meeting.clone(),
            start: i.start,
            end: i.end,
            channel: i.channel,
            downmix: true,
        })
        .collect())
}
pub async fn enrollment_sources(pool: &SqlitePool, profile: &str) -> Result<Vec<EnrollmentSource>> {
    let rows=sqlx::query("SELECT DISTINCT m.id,m.title,t.speaker,m.folder_path,s.current_run_id FROM transcripts t JOIN meetings m ON t.meeting_id=m.id LEFT JOIN meeting_processing_state s ON s.meeting_id=m.id WHERE t.voice_profile_id=? AND m.folder_path IS NOT NULL ORDER BY m.created_at DESC").bind(profile).fetch_all(pool).await?;
    Ok(rows
        .into_iter()
        .filter(|r| {
            r.get::<Option<String>, _>("folder_path").is_some_and(|f| {
                crate::audio::retranscription::find_audio_file(std::path::Path::new(&f)).is_ok()
            })
        })
        .map(|r| EnrollmentSource {
            meeting_id: r.get("id"),
            title: r.get("title"),
            speaker: r.get::<Option<String>, _>("speaker").unwrap_or_default(),
            group_version: r.get("current_run_id"),
        })
        .collect())
}
pub async fn remove_sample(pool: &SqlitePool, id: &str) -> Result<()> {
    let mut tx = pool.begin().await?;
    let found: Option<(String, String)> =
        sqlx::query_as("SELECT profile_id,space_id FROM enrollment_samples WHERE id=?")
            .bind(id)
            .fetch_optional(&mut *tx)
            .await?;
    if let Some((profile, space)) = found {
        let rows=sqlx::query_as::<_,super::evidence::EvidenceRow>(&format!("SELECT {} FROM enrollment_samples WHERE profile_id=? AND space_id=? ORDER BY created_at,id",super::evidence::SAMPLE_COLUMNS))
            .bind(&profile).bind(&space).fetch_all(&mut *tx).await?;
        let source_key = |row: &super::evidence::EvidenceRow| {
            let intervals: Vec<SourceInterval> = serde_json::from_str(&row.intervals_json).ok()?;
            super::evidence::contribution_key(
                &row.space_id,
                row.source_meeting_id.as_deref(),
                row.audio_sha256.as_deref(),
                &intervals,
                &row.id,
            )
            .ok()
        };
        let selected = rows.iter().find(|row| row.id == id).and_then(source_key);
        for row in &rows {
            if row.id == id
                || selected
                    .as_ref()
                    .is_some_and(|key| source_key(row).as_ref() == Some(key))
            {
                sqlx::query("DELETE FROM enrollment_samples WHERE id=?")
                    .bind(&row.id)
                    .execute(&mut *tx)
                    .await?;
            }
        }
        rebuild_tx(&mut tx, &profile, &space).await?;
    }
    tx.commit().await?;
    Ok(())
}
pub async fn delete_enrollment(pool: &SqlitePool, profile: &str, space: &str) -> Result<()> {
    let mut tx = pool.begin().await?;
    sqlx::query("DELETE FROM enrollment_samples WHERE profile_id=? AND space_id=?")
        .bind(profile)
        .bind(space)
        .execute(&mut *tx)
        .await?;
    sqlx::query("DELETE FROM profile_enrollments WHERE profile_id=? AND space_id=?")
        .bind(profile)
        .bind(space)
        .execute(&mut *tx)
        .await?;
    // Remove compatible legacy mirrors too, so deletion actually stops recognition.
    if space.starts_with("legacy:") && space.contains("campplus") {
        sqlx::query("DELETE FROM voice_profile_embeddings WHERE profile_id=? AND model_id=?")
            .bind(profile)
            .bind(NATIVE_EMBEDDING)
            .execute(&mut *tx)
            .await?;
        sqlx::query(
            "UPDATE voice_profiles SET embedding=X'',embedding_dim=0,sample_count=0 WHERE id=?",
        )
        .bind(profile)
        .execute(&mut *tx)
        .await?;
    }
    sqlx::query("DELETE FROM voice_profile_embeddings WHERE profile_id=? AND (model_id=? OR ?='legacy:'||model_id)").bind(profile).bind(space).bind(space).execute(&mut *tx).await?;
    tx.commit().await?;
    Ok(())
}
pub async fn relearn(pool: &SqlitePool, profile: &str, space: &str) -> Result<()> {
    let mut tx = pool.begin().await?;
    rebuild_tx(&mut tx, profile, space).await?;
    tx.commit().await?;
    Ok(())
}
pub(crate) async fn rebuild_tx(
    tx: &mut SqliteConnection,
    profile: &str,
    space: &str,
) -> Result<()> {
    let rows=sqlx::query_as::<_,super::evidence::EvidenceRow>(&format!("SELECT {} FROM enrollment_samples WHERE profile_id=? AND space_id=? ORDER BY created_at,id",super::evidence::SAMPLE_COLUMNS))
        .bind(profile).bind(space).fetch_all(&mut *tx).await?;
    let metadata: Option<String> = sqlx::query_scalar(
        "SELECT space_json FROM profile_enrollments WHERE profile_id=? AND space_id=?",
    )
    .bind(profile)
    .bind(space)
    .fetch_optional(&mut *tx)
    .await?;
    let dimension = metadata
        .and_then(|json| serde_json::from_str::<EmbeddingSpace>(&json).ok())
        .map(|space| space.dimension);
    let assessment = super::evidence::assess(&rows, dimension);
    let seconds = assessment.clean_seconds;
    let centroid = assessment
        .centroid
        .as_ref()
        .map(serde_json::to_string)
        .transpose()?;
    let usable = centroid.is_some();
    sqlx::query("UPDATE profile_enrollments SET embedding_json=?,clean_seconds=?,revision=?,status=?,updated_at=datetime('now') WHERE profile_id=? AND space_id=?").bind(&centroid).bind(seconds).bind(uuid::Uuid::new_v4().to_string()).bind(if usable{"ready"}else{"insufficient_clean_evidence"}).bind(profile).bind(space).execute(&mut *tx).await?;
    let revision:Option<(String,String,String)>=sqlx::query_as("SELECT revision,status,extraction_version FROM profile_enrollments WHERE profile_id=? AND space_id=?").bind(profile).bind(space).fetch_optional(&mut *tx).await?;
    if let Some((revision, status, extraction_version)) = revision {
        let sources:Vec<(String,Option<String>,Option<String>,String,Option<String>)>=sqlx::query_as("SELECT id,source_meeting_id,audio_sha256,intervals_json,source_run_id FROM enrollment_samples WHERE profile_id=? AND space_id=?").bind(profile).bind(space).fetch_all(&mut *tx).await?;
        let metadata:Vec<_>=sources.into_iter().map(|(id,meeting,hash,intervals,run)|{let quality=assessment.quality.get(&id);serde_json::json!({"sample_id":id,"meeting_id":meeting,"audio_sha256":hash,"intervals":serde_json::from_str::<serde_json::Value>(&intervals).unwrap_or_default(),"processing_run_id":run,"contributes":quality.is_some_and(|q|q.contributes),"duplicate_of":quality.and_then(|q|q.duplicate_of.as_ref()),"quality_status":quality.map(|q|q.status).unwrap_or("unknown")})}).collect();
        sqlx::query("INSERT INTO enrollment_versions VALUES(?,?,?,datetime('now'),?)").bind(profile).bind(space).bind(revision).bind(serde_json::json!({"status":status,"clean_seconds":seconds,"contribution_count":assessment.contribution_count,"quality_policy":"semantic-audio-identity-v1:numerical-minimum-v1","extraction_version":extraction_version,"samples":metadata}).to_string()).execute(&mut *tx).await?;
    }
    Ok(())
}
pub(crate) async fn store_evidence_tx(
    tx: &mut SqliteConnection,
    profile: &str,
    meeting: &str,
    run: Option<&str>,
    audio_hash: Option<&str>,
    e: &SpeakerEmbedding,
) -> Result<()> {
    super::identification::validate_embedding(e)?;
    let Some(space) = &e.embedding_space else {
        return Ok(());
    };
    if e.source_intervals.is_empty() {
        return Ok(());
    };
    let rows=sqlx::query_as::<_,super::evidence::EvidenceRow>(&format!("SELECT {} FROM enrollment_samples WHERE profile_id=? AND space_id=? ORDER BY created_at,id",super::evidence::SAMPLE_COLUMNS))
        .bind(profile).bind(&space.id).fetch_all(&mut *tx).await?;
    let key = super::evidence::contribution_key(
        &space.id,
        Some(meeting),
        audio_hash,
        &e.source_intervals,
        "new-source",
    )?;
    if super::evidence::unique_keys(&rows, Some(space.dimension)).contains(&key) {
        return Ok(());
    }
    let now = chrono::Utc::now().to_rfc3339();
    sqlx::query("INSERT INTO profile_enrollments(profile_id,space_id,revision,status,embedding_json,clean_seconds,updated_at,extraction_version,space_json) VALUES(?,?,?,'ready',?,?,?,?,?) ON CONFLICT(profile_id,space_id) DO NOTHING").bind(profile).bind(&space.id).bind(uuid::Uuid::new_v4().to_string()).bind(serde_json::to_string(&e.vector)?).bind(e.clean_seconds).bind(&now).bind(&space.extraction_version).bind(serde_json::to_string(space)?).execute(&mut *tx).await?;
    sqlx::query("INSERT INTO enrollment_samples VALUES(?,?,?,?,?,?,?,?,?,?,?,?)")
        .bind(uuid::Uuid::new_v4().to_string())
        .bind(profile)
        .bind(&space.id)
        .bind(meeting)
        .bind(audio_hash)
        .bind(0)
        .bind(serde_json::to_string(&e.source_intervals)?)
        .bind(run)
        .bind(
            serde_json::json!({"kind":"human_confirmed","speaker":e.speaker,"confirmed_at":now})
                .to_string(),
        )
        .bind(serde_json::to_string(&e.vector)?)
        .bind(e.clean_seconds)
        .bind(&now)
        .execute(&mut *tx)
        .await?;
    rebuild_tx(tx, profile, &space.id).await
}
pub async fn enroll_from_meeting(
    pool: &SqlitePool,
    profile: &str,
    meeting: &str,
    speaker: &str,
    identification: Identification,
    target_id: &str,
    cancelled: &AtomicBool,
    progress: impl Fn(&str, u32, &str) + Send + Sync,
) -> Result<()> {
    if identification == Identification::Off {
        bail!("Select an embedding model for enrollment");
    }
    let snapshot_run = super::runs::current_run(pool, meeting).await?;
    let group_version = snapshot_run
        .clone()
        .unwrap_or_else(|| "legacy_unknown".into());
    let linked:bool=sqlx::query_scalar("SELECT EXISTS(SELECT 1 FROM transcripts WHERE meeting_id=? AND speaker=? AND voice_profile_id=?)").bind(meeting).bind(speaker).bind(profile).fetch_one(pool).await?;
    if !linked {
        bail!("Confirm this speaker's identity before learning their voice");
    }
    let mut a = if let Some(a) = load(pool, meeting).await? {
        a
    } else {
        let rows:Vec<(f64,f64)>=sqlx::query_as("SELECT audio_start_time,audio_end_time FROM transcripts WHERE meeting_id=? AND voice_profile_id=? AND audio_end_time>audio_start_time").bind(meeting).bind(profile).fetch_all(pool).await?;
        let duration = rows.iter().map(|r| r.1).fold(0., f64::max);
        serde_json::from_value(
            serde_json::json!({"version":1,"duration":duration,"language":null,"segments":[],"words":[],"turns":rows.into_iter().map(|(s,e)|serde_json::json!({"start":s,"end":e,"speaker":speaker})).collect::<Vec<_>>(),"overlaps":[],"embeddings":[]}),
        )?
    };
    let attributed:Vec<(f64,f64)>=sqlx::query_as("SELECT audio_start_time,audio_end_time FROM transcripts WHERE meeting_id=? AND speaker=? AND voice_profile_id=? AND audio_end_time>audio_start_time").bind(meeting).bind(speaker).bind(profile).fetch_all(pool).await?;
    let attribution_snapshot:Vec<(f64,f64,Option<String>,Option<String>)>=sqlx::query_as("SELECT audio_start_time,audio_end_time,speaker,voice_profile_id FROM transcripts WHERE meeting_id=? AND audio_end_time>audio_start_time ORDER BY audio_start_time,audio_end_time,id").bind(meeting).fetch_all(pool).await?;
    let other_spans: Vec<_> = attribution_snapshot
        .iter()
        .filter(|(_, _, label, id)| {
            label.as_deref() != Some(speaker) || id.as_deref() != Some(profile)
        })
        .map(|(s, e, _, _)| (*s, *e))
        .collect();
    let groups = source_groups(&a, &attributed, speaker);
    if groups.is_empty() {
        bail!("Source transcript boundaries no longer identify one speaker group; re-detect speakers before enrollment");
    }
    {
        let mut conn = pool.acquire().await?;
        require_confirmed_groups(&mut conn, meeting, &group_version, &groups, profile).await?;
    }
    let model = match identification {
        Identification::Campplus => "campplus",
        Identification::Wespeaker => "wespeaker",
        _ => {
            if a.settings["config"]["speaker_pipeline"] == "community1"
                || a.settings["diarization"] == "community"
            {
                "wespeaker"
            } else {
                "campplus"
            }
        }
    };
    let target = super::config::get_target(pool, target_id).await?;
    let folder: String = sqlx::query_scalar("SELECT folder_path FROM meetings WHERE id=?")
        .bind(meeting)
        .fetch_one(pool)
        .await?;
    let audio = crate::audio::retranscription::find_audio_file(std::path::Path::new(&folder))?;
    let hash = super::runs::audio_hash(audio.clone()).await?;
    if a.audio_sha256
        .as_deref()
        .or_else(|| a.settings["audio_sha256"].as_str())
        .is_some_and(|old| old != hash)
    {
        bail!("Source recording changed; re-detect speakers before enrollment");
    }

    let intervals = confirmed_clean_intervals(&a, &attributed, &other_spans, &groups);
    if intervals.is_empty() {
        bail!("Insufficient clean, non-overlapping audio; split or re-detect this speaker group");
    }
    let v=super::service::extract_embeddings(&target,serde_json::json!({"version":1,"operation":"extract_embeddings","audio_path":audio,"audio_sha256":hash,"duration":a.duration,"source_intervals":intervals,"embedding_model":model,"config":{"identification":model,"speaker_pipeline":if model=="campplus"{"sherpa"}else{"community1"}}}),cancelled,progress).await?;
    a.embeddings = serde_json::from_value(v.get("analysis").unwrap_or(&v)["embeddings"].clone())?;
    for e in &a.embeddings {
        super::identification::validate_embedding(e)?;
        let space = e
            .embedding_space
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Extracted evidence has no exact embedding space"))?;
        let expected_model = space.model == model
            || (model == "wespeaker" && space.model == "community1/bundled-wespeaker");
        if !expected_model
            || !groups.contains(&e.speaker)
            || e.source_intervals.is_empty()
            || e.source_intervals.iter().any(|source| {
                source.speaker != e.speaker
                    || !intervals.iter().any(|allowed| {
                        allowed.speaker == source.speaker
                            && allowed.channel == source.channel
                            && source.start >= allowed.start - 0.0001
                            && source.end <= allowed.end + 0.0001
                    })
            })
        {
            bail!("Extractor returned unauthorized speaker, model or audio evidence; no enrollment was published");
        }
    }
    if a.embeddings.is_empty() {
        bail!("No sufficiently clean voice evidence was extracted");
    }
    if super::runs::audio_hash(audio.clone()).await? != hash {
        bail!("Source recording changed during extraction; no enrollment was published");
    }
    let mut tx = pool.begin().await?;
    let current: Option<String> = sqlx::query_scalar(
        "SELECT current_run_id FROM meeting_processing_state WHERE meeting_id=?",
    )
    .bind(meeting)
    .fetch_optional(&mut *tx)
    .await?
    .flatten();
    if current != snapshot_run {
        bail!("Speaker groups changed during extraction; confirm the current speaker before enrolling");
    }
    let current_folder: Option<String> =
        sqlx::query_scalar("SELECT folder_path FROM meetings WHERE id=?")
            .bind(meeting)
            .fetch_optional(&mut *tx)
            .await?
            .flatten();
    if current_folder.as_deref() != Some(folder.as_str()) {
        bail!("Source recording location changed during extraction");
    }
    let current_spans:Vec<(f64,f64)>=sqlx::query_as("SELECT audio_start_time,audio_end_time FROM transcripts WHERE meeting_id=? AND speaker=? AND voice_profile_id=? AND audio_end_time>audio_start_time ORDER BY audio_start_time,audio_end_time").bind(meeting).bind(speaker).bind(profile).fetch_all(&mut *tx).await?;
    let mut snapshot_spans = attributed;
    snapshot_spans.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.total_cmp(&b.1)));
    if current_spans != snapshot_spans {
        bail!("Speaker attribution changed during extraction; no enrollment was published");
    }
    let current_attributions:Vec<(f64,f64,Option<String>,Option<String>)>=sqlx::query_as("SELECT audio_start_time,audio_end_time,speaker,voice_profile_id FROM transcripts WHERE meeting_id=? AND audio_end_time>audio_start_time ORDER BY audio_start_time,audio_end_time,id").bind(meeting).fetch_all(&mut *tx).await?;
    if current_attributions != attribution_snapshot {
        bail!("Speaker attribution or overlap context changed during extraction; no enrollment was published");
    }
    require_confirmed_groups(&mut tx, meeting, &group_version, &groups, profile).await?;
    if cancelled.load(std::sync::atomic::Ordering::SeqCst) {
        bail!("Enrollment cancelled");
    }
    for e in &a.embeddings {
        store_evidence_tx(
            &mut tx,
            profile,
            meeting,
            snapshot_run.as_deref(),
            Some(&hash),
            e,
        )
        .await?;
    }
    tx.commit().await?;
    Ok(())
}

/// A removed attribution revokes every pooled sample that used its audio.
/// Rebuilding never keeps a centroid containing the revoked contribution.
pub async fn revoke_evidence(
    pool: &SqlitePool,
    profile: &str,
    meeting: &str,
    intervals: &[(f64, f64)],
) -> Result<()> {
    let mut tx = pool.begin().await?;
    revoke_evidence_tx(&mut tx, profile, meeting, intervals).await?;
    tx.commit().await?;
    Ok(())
}

pub(crate) async fn revoke_evidence_tx(
    tx: &mut SqliteConnection,
    profile: &str,
    meeting: &str,
    intervals: &[(f64, f64)],
) -> Result<()> {
    let rows:Vec<(String,String,String)>=sqlx::query_as("SELECT id,space_id,intervals_json FROM enrollment_samples WHERE profile_id=? AND source_meeting_id=?").bind(profile).bind(meeting).fetch_all(&mut *tx).await?;
    let mut spaces = std::collections::BTreeSet::new();
    for (id, space, json) in rows {
        let sources: Vec<SourceInterval> = serde_json::from_str(&json)?;
        if sources.iter().any(|s| {
            intervals
                .iter()
                .any(|(start, end)| s.start < *end && s.end > *start)
        }) {
            sqlx::query("DELETE FROM enrollment_samples WHERE id=?")
                .bind(id)
                .execute(&mut *tx)
                .await?;
            spaces.insert(space);
        }
    }
    for space in spaces {
        rebuild_tx(tx, profile, &space).await?;
    }
    Ok(())
}

pub(crate) fn source_groups(
    a: &Analysis,
    spans: &[(f64, f64)],
    speaker: &str,
) -> std::collections::BTreeSet<String> {
    if a.segments.is_empty() {
        return std::collections::BTreeSet::from([speaker.into()]);
    }
    let mut groups = std::collections::BTreeSet::new();
    for (start, end) in spans {
        let labels: std::collections::BTreeSet<_> = a
            .segments
            .iter()
            .filter(|s| s.start == *start && s.end == *end)
            .filter_map(|s| s.speaker.clone())
            .collect();
        if labels.len() != 1 {
            return std::collections::BTreeSet::new();
        }
        groups.extend(labels);
    }
    groups
}
async fn require_confirmed_groups(
    conn: &mut SqliteConnection,
    meeting: &str,
    version: &str,
    groups: &std::collections::BTreeSet<String>,
    profile: &str,
) -> Result<()> {
    let json: Option<String> = sqlx::query_scalar(
        "SELECT analysis_json FROM meeting_transcription_analysis WHERE meeting_id=?",
    )
    .bind(meeting)
    .fetch_optional(&mut *conn)
    .await?;
    let analysis: Option<Analysis> = json.map(|json| serde_json::from_str(&json)).transpose()?;
    for group in groups {
        let decision:Option<(Option<String>,bool)>=sqlx::query_as("SELECT profile_id,confirmed FROM speaker_identity_decisions WHERE meeting_id=? AND group_version=? AND speaker=?").bind(meeting).bind(version).bind(group).fetch_optional(&mut *conn).await?;
        let mut automatic = false;
        if let Some(a) = &analysis {
            for segment in a
                .segments
                .iter()
                .filter(|segment| segment.speaker.as_deref() == Some(group.as_str()))
            {
                automatic|=sqlx::query_scalar::<_,bool>("SELECT EXISTS(SELECT 1 FROM transcripts WHERE meeting_id=? AND audio_start_time=? AND audio_end_time=? AND identity_origin='machine')")
                    .bind(meeting).bind(segment.start).bind(segment.end).fetch_one(&mut *conn).await?;
            }
        } else {
            automatic=sqlx::query_scalar("SELECT EXISTS(SELECT 1 FROM transcripts WHERE meeting_id=? AND speaker=? AND identity_origin='machine')").bind(meeting).bind(group).fetch_one(&mut *conn).await?;
        }
        if automatic || decision != Some((Some(profile.into()), true)) {
            bail!("Confirm this speaker's identity before learning voice evidence; an automatic suggestion cannot enroll itself");
        }
    }
    Ok(())
}

/// Called only by the explicit Confirm identity and enroll user action.
/// It records a human decision before any learning or remote extraction starts.
pub async fn confirm_enrollment_source(
    pool: &SqlitePool,
    profile: &str,
    meeting: &str,
    speaker: &str,
    expected_group_version: Option<&str>,
) -> Result<()> {
    let mut tx = pool.begin().await?;
    let current: Option<String> = sqlx::query_scalar(
        "SELECT current_run_id FROM meeting_processing_state WHERE meeting_id=?",
    )
    .bind(meeting)
    .fetch_optional(&mut *tx)
    .await?
    .flatten();
    if current.as_deref() != expected_group_version {
        bail!("Speaker groups changed while the source picker was open; reload its sources");
    }
    let spans:Vec<(f64,f64)>=sqlx::query_as("SELECT audio_start_time,audio_end_time FROM transcripts WHERE meeting_id=? AND speaker=? AND voice_profile_id=? AND audio_end_time>audio_start_time").bind(meeting).bind(speaker).bind(profile).fetch_all(&mut *tx).await?;
    if spans.is_empty() {
        bail!("This source attribution no longer exists");
    }
    let json: Option<String> = sqlx::query_scalar(
        "SELECT analysis_json FROM meeting_transcription_analysis WHERE meeting_id=?",
    )
    .bind(meeting)
    .fetch_optional(&mut *tx)
    .await?;
    let analysis: Option<Analysis> = json.map(|s| serde_json::from_str(&s)).transpose()?;
    let groups = analysis
        .as_ref()
        .map(|a| source_groups(a, &spans, speaker))
        .unwrap_or_else(|| std::collections::BTreeSet::from([speaker.into()]));
    if groups.is_empty() {
        bail!("Source transcript boundaries no longer identify one group; reload or re-detect speakers");
    }
    let version = current.unwrap_or_else(|| "legacy_unknown".into());
    for group in groups {
        if let Some(a) = &analysis {
            for segment in a
                .segments
                .iter()
                .filter(|s| s.speaker.as_deref() == Some(&group))
            {
                let links:Vec<Option<String>>=sqlx::query_scalar("SELECT voice_profile_id FROM transcripts WHERE meeting_id=? AND audio_start_time=? AND audio_end_time=?").bind(meeting).bind(segment.start).bind(segment.end).fetch_all(&mut *tx).await?;
                if links.iter().any(|id| id.as_deref() != Some(profile)) {
                    bail!("This group contains conflicting or unassigned identities; split or re-detect it before enrollment");
                }
                sqlx::query("UPDATE transcripts SET identity_origin='human' WHERE meeting_id=? AND audio_start_time=? AND audio_end_time=? AND voice_profile_id=?")
                    .bind(meeting).bind(segment.start).bind(segment.end).bind(profile).execute(&mut *tx).await?;
            }
        }
        sqlx::query("INSERT INTO speaker_identity_decisions VALUES(?,?,?,?,1,NULL,datetime('now')) ON CONFLICT(meeting_id,group_version,speaker) DO UPDATE SET profile_id=excluded.profile_id,confirmed=1,updated_at=excluded.updated_at").bind(meeting).bind(&version).bind(&group).bind(profile).execute(&mut *tx).await?;
        sqlx::query("INSERT INTO identity_decision_events VALUES(?,?,?,?,?,'confirmed',datetime('now'),?)").bind(uuid::Uuid::new_v4().to_string()).bind(meeting).bind(&version).bind(group).bind(profile).bind(serde_json::json!({"source":"explicit_enrollment_confirmation","display_label":speaker,"confirmed_intervals":spans}).to_string()).execute(&mut *tx).await?;
    }
    sqlx::query("UPDATE transcripts SET identity_origin='human' WHERE meeting_id=? AND speaker=? AND voice_profile_id=?")
        .bind(meeting).bind(speaker).bind(profile).execute(&mut *tx).await?;
    tx.commit().await?;
    Ok(())
}

pub(crate) fn merge_confirmed_intervals(spans: &[(f64, f64)]) -> Vec<(f64, f64)> {
    let mut spans: Vec<_> = spans
        .iter()
        .copied()
        .filter(|(s, e)| s.is_finite() && e.is_finite() && *s >= 0. && e > s)
        .collect();
    spans.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.total_cmp(&b.1)));
    let mut merged: Vec<(f64, f64)> = vec![];
    for (s, e) in spans {
        if let Some(last) = merged.last_mut().filter(|last| s <= last.1) {
            last.1 = last.1.max(e);
        } else {
            merged.push((s, e));
        }
    }
    merged
}
/// ASR word boundaries need not cover the acoustic turn (including its pauses).
/// Naming every original segment confirms the entire diarization group. Partial
/// edits still only authorize the intersection with the explicitly named words.
fn confirmed_cached_intervals(
    a: &Analysis,
    attributed: &[(f64, f64)],
    other: &[(f64, f64)],
    groups: &std::collections::BTreeSet<String>,
) -> Vec<SourceInterval> {
    let whole: std::collections::BTreeSet<_> = groups.iter().filter(|group| {
        let segments: Vec<_> = a.segments.iter()
            .filter(|s| s.speaker.as_ref() == Some(*group)).collect();
        !segments.is_empty() && segments.iter()
            .all(|s| attributed.contains(&(s.start, s.end)))
    }).cloned().collect();
    let partial = groups.difference(&whole).cloned().collect();
    let mut intervals = confirmed_clean_intervals(a, attributed, other, &partial);
    intervals.extend(super::service::clean_intervals(a).into_iter()
        .filter(|i| whole.contains(&i.speaker)));
    intervals
}

/// Repair missing enrollment from already human-confirmed, current groups.
/// No extraction or guessing: reuse only fully attributed cached clean samples.
/// Enrollment history is a tombstone: deleted/rejected evidence stays deleted.
pub async fn recover_confirmed_evidence(pool: &SqlitePool, profile: &str) -> Result<()> {
    let mut tx = pool.begin().await?;
    recover_confirmed_evidence_tx(&mut tx, profile).await?;
    tx.commit().await?;
    Ok(())
}

pub(crate) async fn recover_confirmed_evidence_tx(
    tx: &mut SqliteConnection,
    profile: &str,
) -> Result<()> {
    // Removing a whole model enrollment opts out of automatic recovery for it.
    let deleted_spaces: Vec<String> = sqlx::query_scalar(
        "SELECT DISTINCT space_id FROM enrollment_versions v WHERE profile_id=? AND NOT EXISTS (SELECT 1 FROM profile_enrollments p WHERE p.profile_id=v.profile_id AND p.space_id=v.space_id)",
    ).bind(profile).fetch_all(&mut *tx).await?;
    // History identifies individual sources, not a blanket ban on new meetings.
    // This preserves removed samples while allowing existing profiles to grow.
    let histories: Vec<(String, String)> = sqlx::query_as(
        "SELECT space_id,summary_json FROM enrollment_versions WHERE profile_id=?",
    ).bind(profile).fetch_all(&mut *tx).await?;
    let mut seen = std::collections::HashSet::new();
    for (space, json) in histories {
        let history: serde_json::Value = serde_json::from_str(&json)?;
        if let Some(samples) = history["samples"].as_array() {
            for sample in samples {
                let intervals: Vec<SourceInterval> = serde_json::from_value(sample["intervals"].clone())?;
                seen.insert(super::evidence::contribution_key(
                    &space, sample["meeting_id"].as_str(), sample["audio_sha256"].as_str(),
                    &intervals, sample["sample_id"].as_str().unwrap_or("unknown"),
                )?);
            }
        }
    }
    let rows: Vec<(String, String, String, String)> = sqlx::query_as(
        "SELECT d.meeting_id,d.group_version,d.speaker,a.analysis_json FROM speaker_identity_decisions d JOIN meeting_processing_state s ON s.meeting_id=d.meeting_id AND s.group_version=d.group_version JOIN meeting_transcription_analysis a ON a.meeting_id=d.meeting_id WHERE d.profile_id=? AND d.confirmed=1",
    ).bind(profile).fetch_all(&mut *tx).await?;
    for (meeting, run, group, json) in rows {
        let a: Analysis = serde_json::from_str(&json)?;
        let spans: Vec<(f64, f64)> = sqlx::query_as(
            "SELECT audio_start_time,audio_end_time FROM transcripts WHERE meeting_id=? AND voice_profile_id=? AND identity_origin='human' AND audio_end_time>audio_start_time",
        ).bind(&meeting).bind(profile).fetch_all(&mut *tx).await?;
        // Require every original segment, not an automatic match or a partial edit.
        let segments: Vec<_> = a.segments.iter().filter(|s| s.speaker.as_deref()==Some(&group)).collect();
        if segments.is_empty() || !segments.iter().all(|s| spans.contains(&(s.start,s.end))) { continue; }
        let clean = super::service::clean_intervals(&a);
        let cached: Vec<String> = sqlx::query_scalar(
            "SELECT embedding_json FROM speaker_group_embeddings WHERE meeting_id=? AND group_version=? AND speaker=?",
        ).bind(&meeting).bind(&run).bind(&group).fetch_all(&mut *tx).await?;
        let mut embeddings = a.embeddings.clone();
        for json in cached {
            embeddings.push(serde_json::from_str(&json)?);
        }
        let hash = a.audio_sha256.as_deref().or_else(|| a.settings["audio_sha256"].as_str());
        for e in embeddings.iter().filter(|e| e.speaker == group) {
            let Some(space) = &e.embedding_space else { continue; };
            if deleted_spaces.contains(&space.id) || e.source_intervals.is_empty()
                || super::identification::validate_embedding(e).is_err()
                || !e.source_intervals.iter().all(|i| clean.iter().any(|c|
                    c.speaker == i.speaker && c.channel == i.channel && i.start >= c.start && i.end <= c.end)) { continue; }
            let key = super::evidence::contribution_key(&space.id, Some(&meeting), hash, &e.source_intervals, "new-source")?;
            if !seen.insert(key) { continue; }
            store_evidence_tx(tx, profile, &meeting, Some(&run), hash, e).await?;
        }
    }
    Ok(())
}

pub(crate) fn confirmed_clean_intervals(
    a: &Analysis,
    attributed: &[(f64, f64)],
    other: &[(f64, f64)],
    groups: &std::collections::BTreeSet<String>,
) -> Vec<SourceInterval> {
    let union = merge_confirmed_intervals(attributed);
    let mut clean: Vec<_> = super::service::clean_intervals(a)
        .into_iter()
        .filter(|i| groups.contains(&i.speaker))
        .collect();
    for (start, end) in other {
        clean = clean
            .into_iter()
            .flat_map(|i| {
                if *end <= i.start || *start >= i.end {
                    vec![i]
                } else {
                    vec![
                        SourceInterval {
                            end: i.end.min(*start),
                            ..i.clone()
                        },
                        SourceInterval {
                            start: i.start.max(*end),
                            ..i
                        },
                    ]
                }
            })
            .filter(|i| i.end > i.start)
            .collect();
    }
    clean
        .into_iter()
        .flat_map(|i| {
            union.iter().filter_map(move |(start, end)| {
                let s = i.start.max(*start);
                let e = i.end.min(*end);
                (e - s >= 3.).then(|| SourceInterval {
                    speaker: i.speaker.clone(),
                    start: s,
                    end: e,
                    channel: i.channel,
                })
            })
        })
        .collect()
}
