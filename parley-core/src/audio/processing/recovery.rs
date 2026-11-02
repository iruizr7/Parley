//! Private reconnect metadata. Connection settings never enter public run history.
use super::{config::ProcessingTarget, service::ProcessingOptions};
use anyhow::{anyhow, bail, Result};
use serde::{Deserialize, Serialize};
use sqlx::SqlitePool;

#[derive(Clone, Serialize, Deserialize)]
pub(crate) struct RecoveryContext {
    version: u32,
    request_id: String,
    meeting_id: String,
    pub target: ProcessingTarget,
    pub options: ProcessingOptions,
    pub operation: String,
    pub started_at: String,
}
fn key(request: &str) -> String {
    format!("processing_recovery_v1:{request}")
}
pub(crate) async fn retain(
    pool: &SqlitePool,
    meeting: &str,
    request: &str,
    target: &ProcessingTarget,
    options: &ProcessingOptions,
    operation: &str,
    started_at: &str,
) -> Result<()> {
    let context = RecoveryContext {
        version: 1,
        request_id: request.into(),
        meeting_id: meeting.into(),
        target: target.clone(),
        options: options.clone(),
        operation: operation.into(),
        started_at: started_at.into(),
    };
    // The request identity is immutable. A reconnect must never replace its
    // connection details with a newer target carrying the same display ID.
    let result = sqlx::query("INSERT INTO app_settings(key,value,updated_at) VALUES(?,?,datetime('now')) ON CONFLICT(key) DO NOTHING")
        .bind(key(request)).bind(serde_json::to_string(&context)?).execute(pool).await?;
    if result.rows_affected() == 0 {
        bail!("This processing request already has an original execution snapshot");
    }
    Ok(())
}
pub(crate) async fn load(
    pool: &SqlitePool,
    meeting: &str,
    request: &str,
) -> Result<RecoveryContext> {
    let json: Option<String> = sqlx::query_scalar("SELECT value FROM app_settings WHERE key=?")
        .bind(key(request))
        .fetch_optional(pool)
        .await?;
    let context: RecoveryContext = serde_json::from_str(&json.ok_or_else(|| anyhow!(
        "The original target setup for this historical request is unavailable; start a new processing run explicitly"
    ))?)?;
    if context.version != 1 || context.meeting_id != meeting || context.request_id != request {
        bail!("Recovery metadata does not belong to this meeting and request");
    }
    Ok(context)
}
pub(crate) async fn forget(pool: &SqlitePool, request: &str) -> Result<()> {
    sqlx::query("DELETE FROM app_settings WHERE key=?")
        .bind(key(request))
        .execute(pool)
        .await?;
    Ok(())
}
