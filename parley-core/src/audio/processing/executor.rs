//! Versioned local/SSH worker transport. Inference owns no database state.
//!
//! OpenSSH retains host verification and existing credentials. The remote
//! supervisor owns jobs after disconnection; callers can resume by request ID.
use super::config::{PipelineConfig, ProcessingTarget, TargetKind};
use anyhow::{anyhow, bail, Context, Result};
use base64::Engine;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::path::Path;
use std::process::Stdio;
use std::sync::atomic::{AtomicBool, Ordering};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::process::Command;

const MAX_RESULT_BYTES: usize = 64 * 1024 * 1024;
const MAX_CONTROL_BYTES: usize = 90 * 1024 * 1024;
const BOOTSTRAP: &str = r#"import sys,json,os,pathlib,hashlib,tempfile
p=json.load(sys.stdin)
namespace={'__name__':'parley_bootstrap_protocol'}
exec(p['files']['protocol.py'],namespace)
try:
 root=namespace['private_root'](p['root'])
except ValueError as error:
 print(json.dumps({'version':1,'error':'worker_root_rejected','message':str(error)}));sys.exit(0)
except OSError:
 print(json.dumps({'version':1,'error':'worker_root_unavailable'}));sys.exit(0)
digest=hashlib.sha256(json.dumps(p['files'],sort_keys=True).encode()).hexdigest()
folder=root/'programs'/digest
folder.mkdir(parents=True,exist_ok=True,mode=0o700)
for name,content in p['files'].items():
 if pathlib.Path(name).name!=name: raise ValueError('Invalid worker filename')
 path=folder/name
 if path.is_symlink(): raise ValueError('Invalid worker file link')
 if not path.is_file() or path.read_text()!=content:
  with tempfile.NamedTemporaryFile(mode='w',encoding='utf-8',dir=folder,prefix='.'+name+'.',suffix='.partial',delete=False) as stream:
   tmp=pathlib.Path(stream.name)
   try:
    os.fchmod(stream.fileno(),0o600);stream.write(content);stream.flush();tmp.replace(path)
   finally: tmp.unlink(missing_ok=True)
print(json.dumps({'version':1,'program':str(folder/'supervisor.py'),'root':str(root)}))
"#;

fn files() -> Value {
    json!({
        "worker.py": include_str!("../../../../tools/offline_transcription/worker.py"),
        "protocol.py": include_str!("../../../../tools/offline_transcription/protocol.py"),
        "assets.py": include_str!("../../../../tools/offline_transcription/assets.py"),
        "qualification.py": include_str!("../../../../tools/offline_transcription/qualification.py"),
        "resources.py": include_str!("../../../../tools/offline_transcription/resources.py"),
        "supervisor.py": include_str!("../../../../tools/offline_transcription/supervisor.py"),
        "setup_runtime.py": include_str!("../../../../tools/offline_transcription/setup_runtime.py"),
        "requirements.txt": include_str!("../../../../tools/offline_transcription/requirements.txt")
    })
}

fn quoted(value: &str) -> String {
    format!("'{}'", value.replace('\'', "'\"'\"'"))
}

fn command(target: &ProcessingTarget, arguments: &[String]) -> Result<Command> {
    match target.kind {
        TargetKind::Local => {
            let mut cmd = Command::new("python3");
            cmd.args(arguments);
            Ok(cmd)
        }
        TargetKind::Ssh => {
            let host = target
                .ssh_host
                .as_deref()
                .context("Choose an OpenSSH host alias for this target")?;
            if host.is_empty()
                || host.starts_with('-')
                || host.chars().any(char::is_whitespace)
                || host.contains('\0')
            {
                bail!("Invalid OpenSSH destination");
            }
            let mut cmd = Command::new("ssh");
            cmd.args([
                "-T",
                "-o",
                "BatchMode=yes",
                "-o",
                "StrictHostKeyChecking=yes",
                "-o",
                "ConnectTimeout=10",
                "-o",
                "ServerAliveInterval=5",
                "-o",
                "ServerAliveCountMax=2",
                "--",
                host,
            ]);
            // SSH executes a remote shell; quote every individual argument.
            // No user data is interpolated into executable Python source.
            cmd.arg(
                std::iter::once("python3".to_string())
                    .chain(arguments.iter().map(|a| quoted(a)))
                    .collect::<Vec<_>>()
                    .join(" "),
            );
            Ok(cmd)
        }
    }
}

async fn communicate(mut cmd: Command, input: Vec<u8>) -> Result<Value> {
    cmd.stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .kill_on_drop(true);
    let mut child = cmd
        .spawn()
        .context("Cannot start the target's worker transport")?;
    let mut stdin = child.stdin.take().context("Worker stdin unavailable")?;
    let stdout = child.stdout.take().context("Worker stdout unavailable")?;
    let writer = tokio::spawn(async move {
        stdin.write_all(&input).await?;
        stdin.shutdown().await
    });
    let reader = tokio::spawn(async move {
        let mut output = Vec::new();
        stdout
            .take((MAX_CONTROL_BYTES + 1) as u64)
            .read_to_end(&mut output)
            .await?;
        Ok::<_, std::io::Error>(output)
    });
    let status = tokio::time::timeout(std::time::Duration::from_secs(60), child.wait())
        .await
        .context(
            "Worker transport timed out; the detached job can be resumed by its request ID",
        )??;
    let written = writer.await?;
    let output = reader.await??;
    if output.len() > MAX_CONTROL_BYTES {
        bail!("Worker response exceeds its protocol size limit");
    }
    if !status.success() || written.is_err() {
        // Do not echo SSH stderr, private paths, credentials or model logs.
        bail!("Target transport disconnected or rejected the operation; detached jobs retain their request ID for reconnection");
    }
    let value: Value =
        serde_json::from_slice(&output).context("Invalid worker control response")?;
    if value["error"] == "worker_root_rejected" {
        // Only known, path-free validation reasons may enter the interface.
        let reason = value["message"].as_str().filter(|message| matches!(*message,
            "Choose a dedicated worker subdirectory, not a home or shared system directory"
            | "Worker directory must be a private directory owned by the current user; its permissions were not changed"
            | "Worker directory ownership marker does not match"
            | "Existing directory is not a verified worker root; choose a new dedicated subdirectory"
            | "Legacy worker release must belong to this worker directory"
            | "Existing worker release has no verified complete code snapshot"
            | "Existing worker native snapshot failed verification"
        )).unwrap_or("The selected directory could not be verified as an owned worker root");
        bail!("Worker directory setup required: {reason}. Choose a dedicated private worker subdirectory.");
    }
    if value["error"] == "worker_root_unavailable" {
        bail!("Cannot access the selected worker directory. Choose a writable dedicated private worker subdirectory.");
    }
    let job_status = value["protocol_version"] == 1
        && value["request_id"]
            .as_str()
            .is_some_and(|id| identifier(id).is_ok())
        && matches!(
            value["state"].as_str(),
            Some(
                "created" | "queued" | "running" | "succeeded" | "failed" | "cancelled" | "cleaned"
            )
        );
    if value.get("error").is_some() && !job_status {
        bail!("Worker rejected the requested operation; check this target's setup and owned job status");
    }
    Ok(value)
}

struct Connection {
    program: String,
    root: String,
}
async fn connect(target: &ProcessingTarget) -> Result<Connection> {
    let root = target
        .worker_path
        .as_deref()
        .filter(|p| !p.trim().is_empty())
        .map(str::to_owned)
        .unwrap_or_else(|| {
            if target.kind == TargetKind::Local {
                dirs::data_local_dir()
                    .unwrap_or_else(std::env::temp_dir)
                    .join("parley/offline-worker")
                    .to_string_lossy()
                    .into_owned()
            } else {
                "~/.local/share/parley/offline-worker".into()
            }
        });
    if root.contains('\0') {
        bail!("Invalid worker directory");
    }
    let payload = serde_json::to_vec(&json!({"root":root,"files":files()}))?;
    let result = communicate(command(target, &["-c".into(), BOOTSTRAP.into()])?, payload).await?;
    Ok(Connection {
        program: result["program"]
            .as_str()
            .context("Missing worker program")?
            .into(),
        root: result["root"]
            .as_str()
            .context("Missing worker root")?
            .into(),
    })
}

async fn control(
    target: &ProcessingTarget,
    connection: &Connection,
    message: Value,
) -> Result<Value> {
    communicate(
        command(
            target,
            &[
                connection.program.clone(),
                "--root".into(),
                connection.root.clone(),
            ],
        )?,
        serde_json::to_vec(&message)?,
    )
    .await
}

fn identifier(identity: &str) -> Result<()> {
    if !(8..=80).contains(&identity.len())
        || !identity
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_')
    {
        bail!("Invalid processing request identifier");
    }
    Ok(())
}

async fn hash_audio(path: &Path) -> Result<String> {
    let path = path.to_owned();
    tokio::task::spawn_blocking(move || -> Result<String> {
        use std::io::Read;
        let mut file = std::fs::File::open(path)?;
        let mut digest = Sha256::new();
        let mut block = vec![0u8; 1024 * 1024];
        loop {
            let count = file.read(&mut block)?;
            if count == 0 {
                break;
            }
            digest.update(&block[..count]);
        }
        Ok(format!("{:x}", digest.finalize()))
    })
    .await?
}

fn decode_result(envelope: &Value) -> Result<Value> {
    let encoded = envelope["result_base64"]
        .as_str()
        .context("Missing terminal result bytes")?;
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(encoded)
        .context("Invalid terminal result encoding")?;
    if bytes.len() > MAX_RESULT_BYTES {
        bail!("Terminal result exceeds its size limit");
    }
    let digest = format!("{:x}", Sha256::digest(&bytes));
    if envelope["result_sha256"] != digest {
        bail!("Returned result checksum mismatch");
    }
    serde_json::from_slice(&bytes).context("Invalid terminal result JSON")
}

/// Count bytes accepted by the transport, without buffering the recording.
/// Receipt/checksum acknowledgement is a separate subtask after this copy.
async fn copy_upload(
    source: &mut (impl tokio::io::AsyncRead + Unpin),
    destination: &mut (impl tokio::io::AsyncWrite + Unpin),
    total: u64,
    label: &str,
    progress: &(impl Fn(&str, u32, &str) + Send + Sync),
) -> Result<()> {
    let started = std::time::Instant::now();
    let mut last_update = started;
    let mut last_percent = 0;
    let mut sent = 0u64;
    let mut buffer = vec![0u8; 64 * 1024];
    let message = |sent: u64| {
        let mib = 1024. * 1024.;
        let rate = if sent == 0 { String::new() } else {
            format!(" · {:.2} MiB/s", sent as f64 / mib / started.elapsed().as_secs_f64().max(0.001))
        };
        format!("Uploading {label}: {:.1} / {:.1} MiB{rate}", sent as f64 / mib, total as f64 / mib)
    };
    progress("uploading", 0, &message(0));
    loop {
        let count = source.read(&mut buffer).await?;
        if count == 0 { break; }
        destination.write_all(&buffer[..count]).await?;
        sent += count as u64;
        let percent = ((sent as u128 * 100 / total.max(1) as u128).min(99)) as u32;
        if percent != last_percent || last_update.elapsed() >= std::time::Duration::from_millis(250) {
            progress("uploading", percent, &message(sent));
            last_percent = percent;
            last_update = std::time::Instant::now();
        }
    }
    if sent != total { bail!("Upload source size changed during transfer"); }
    destination.shutdown().await?;
    progress("uploading", 100, &message(sent));
    Ok(())
}

async fn upload(
    target: &ProcessingTarget,
    connection: &Connection,
    identity: &str,
    digest: &str,
    audio: &Path,
    cancelled: &AtomicBool,
    kind: &str,
    on_progress: &(impl Fn(&str, u32, &str) + Send + Sync),
) -> Result<()> {
    let started = std::time::Instant::now();
    let mut cmd = command(
        target,
        &[
            connection.program.clone(),
            "--root".into(),
            connection.root.clone(),
            "--receive".into(),
            identity.into(),
            "--sha256".into(),
            digest.into(),
            "--receive-kind".into(),
            kind.into(),
        ],
    )?;
    cmd.stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .kill_on_drop(true);
    let mut child = cmd
        .spawn()
        .context("Cannot upload recording to the selected target")?;
    let mut input = child.stdin.take().context("Upload stdin unavailable")?;
    let mut source = tokio::fs::File::open(audio).await?;
    let total = source.metadata().await?.len();
    let label = if kind == "native_bundle" { "native helper bundle" } else { "recording" };
    let transfer = async {
        copy_upload(&mut source, &mut input, total, label, on_progress).await?;
        drop(input);
        on_progress("upload_verifying", 0, "Waiting for the target to confirm receipt and verify the checksum");
        // Includes the final SSH buffer drain and remote acknowledgement. Keep
        // cancellation available here as well as during the byte transfer.
        child.wait_with_output().await.context("Cannot confirm recording transfer")
    };
    tokio::pin!(transfer);
    let output = loop {
        tokio::select! {
            result = &mut transfer => break result?,
            _ = tokio::time::sleep(std::time::Duration::from_millis(100)) => {
                if cancelled.load(Ordering::Relaxed) {
                    // Dropping the pending transfer kills its owned child.
                    bail!("Processing cancelled while transferring audio");
                }
            }
        }
    };
    if !output.status.success() {
        bail!("Recording transfer failed checksum verification or disconnected");
    }
    let result: Value = serde_json::from_slice(&output.stdout)?;
    if result["audio_sha256"] != digest {
        bail!("Recording transfer checksum mismatch");
    }
    control(target,connection,json!({"action":"record_transfer","request_id":identity,"kind":"upload","seconds":started.elapsed().as_secs_f64()})).await?;
    Ok(())
}

/// Submit or resume an immutable request. Successful results remain available
/// by ID until the caller publishes them atomically and calls [`cleanup`].
pub async fn execute(
    target: &ProcessingTarget,
    mut request: Value,
    cancelled: &AtomicBool,
    on_progress: impl Fn(&str, u32, &str) + Send + Sync,
) -> Result<Value> {
    if cancelled.load(Ordering::Relaxed) {
        bail!("Processing cancelled");
    }
    if !request.is_object() {
        bail!("Processing request must be an object");
    }
    request["version"] = json!(1);
    let identity = request["request_id"]
        .as_str()
        .map(str::to_owned)
        .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());
    identifier(&identity)?;
    request["request_id"] = json!(identity);
    request["device"] = json!(target.device);
    if request.get("speech_device").is_none() {
        request["speech_device"] = json!(target.device);
    }
    if request.get("speaker_device").is_none() {
        request["speaker_device"] = json!(if target.device == "cuda"
            && request["config"]["speaker_pipeline"] == "community1"
        {
            "cuda"
        } else {
            "cpu"
        });
    }
    if let Some(assets) = &target.assets_dir {
        request["assets_dir"] = json!(assets);
    }
    let operation = request["operation"]
        .as_str()
        .unwrap_or("process")
        .to_string();
    let has_audio = !["probe", "prepare", "install_runtime"].contains(&operation.as_str());
    let audio = if has_audio {
        let path = request["audio_path"]
            .as_str()
            .context("Processing request needs its source recording")?
            .to_owned();
        let digest = hash_audio(Path::new(&path)).await?;
        if request["audio_sha256"]
            .as_str()
            .is_some_and(|expected| expected != digest)
        {
            bail!("Input recording changed before submission");
        }
        request["audio_sha256"] = json!(digest);
        Some((path, digest))
    } else {
        None
    };
    let native_bundle = if operation == "install_runtime" {
        if let Some(path) = request["native_bundle_path"].as_str().map(str::to_owned) {
            let digest = hash_audio(Path::new(&path)).await?;
            request["native_bundle"] = json!(true);
            request["native_sha256"] = json!(digest);
            request
                .as_object_mut()
                .unwrap()
                .remove("native_bundle_path");
            Some((path, digest))
        } else {
            None
        }
    } else {
        None
    };
    let connection = connect(target).await?;
    let state = control(
        target,
        &connection,
        json!({"action":"create","request":request}),
    )
    .await?;
    if state["state"] == "created" {
        if let Some((path, digest)) = audio {
            if let Err(error) = upload(
                target,
                &connection,
                &identity,
                &digest,
                Path::new(&path),
                cancelled,
                "audio",
                &on_progress,
            )
            .await
            {
                if cancelled.load(Ordering::Relaxed) {
                    let _ = control(
                        target,
                        &connection,
                        json!({"action":"cancel","request_id":identity}),
                    )
                    .await;
                    let _ = control(
                        target,
                        &connection,
                        json!({"action":"cleanup","request_id":identity}),
                    )
                    .await;
                }
                return Err(error);
            }
        }
        if let Some((path, digest)) = native_bundle {
            upload(
                target,
                &connection,
                &identity,
                &digest,
                Path::new(&path),
                cancelled,
                "native_bundle",
                &on_progress,
            )
            .await?;
        }
        if cancelled.load(Ordering::Relaxed) {
            control(
                target,
                &connection,
                json!({"action":"cancel","request_id":identity}),
            )
            .await?;
            let _ = control(
                target,
                &connection,
                json!({"action":"cleanup","request_id":identity}),
            )
            .await;
            bail!("Processing cancelled before submission");
        }
        control(
            target,
            &connection,
            json!({"action":"submit","request_id":identity,"python":target.python}),
        )
        .await?;
    }
    wait_result(target, &connection, &identity, cancelled, &on_progress).await
}

/// Map only documented failure codes to public text. Worker diagnostics can
/// contain credentials and private paths, so never display their raw strings.
fn worker_failure(target: &ProcessingTarget, state: &Value, identity: &str) -> String {
    let artifact = match state["error_artifact"].as_str() {
        Some("community1") => Some("Community-1"),
        Some(
            "large-v3-turbo"
            | "large-v3-turbo-q5_0"
            | "large-v3-turbo-fp16"
            | "large-v3-turbo-int8"
            | "large-v3-turbo-mlx-fp16",
        ) => Some("The selected speech model"),
        Some("campplus" | "sherpa-segmentation") => Some("The selected speaker model"),
        _ => None,
    };
    if state["error_code"] == "model_access_denied" {
        if let Some(artifact) = artifact {
            let location = match target.kind {
                TargetKind::Local => "this computer",
                TargetKind::Ssh => "the selected remote target, as its SSH user",
            };
            let conditions = if state["error_artifact"] == "community1" {
                "1. Sign in at https://huggingface.co/pyannote/speaker-diarization-community-1 and accept the model's access conditions."
            } else {
                "1. Check that your Hugging Face account has access to the selected model and has accepted any required access conditions."
            };
            return format!("{artifact} download failed: Hugging Face denied access (authentication or model permission required).\n\n{conditions}\n2. On {location}, run `hf auth login` with a token allowed to read this model. Use the Hugging Face CLI in the worker environment: current/bin/hf under the worker folder in Target setup.\n3. Retry Download selected models.\n\nRequest ID: {identity}");
        }
    }
    format!("The target job failed. Its private diagnostic log contains the details; existing meeting data is unchanged. Request ID: {identity}")
}

async fn wait_result(
    target: &ProcessingTarget,
    connection: &Connection,
    identity: &str,
    cancelled: &AtomicBool,
    on_progress: &(impl Fn(&str, u32, &str) + Send + Sync),
) -> Result<Value> {
    let mut previous = Value::Null;
    let mut previous_state = String::new();
    let mut cancel_sent = false;
    loop {
        if cancelled.load(Ordering::Relaxed) && !cancel_sent {
            control(
                target,
                connection,
                json!({"action":"cancel","request_id":identity}),
            )
            .await?;
            cancel_sent = true;
        }
        let state = control(
            target,
            connection,
            json!({"action":"status","request_id":identity}),
        )
        .await
        .map_err(|error| anyhow!("{error} (request ID: {identity})"))?;
        if state["state"] == "queued" && previous_state != "queued" {
            on_progress(
                "queued",
                0,
                "Waiting for the target's previous job to finish",
            );
        }
        previous_state = state["state"].as_str().unwrap_or("").into();
        if state["progress"].is_object() && state["progress"] != previous {
            let p = &state["progress"];
            on_progress(
                p["stage"].as_str().unwrap_or("processing"),
                p["percent"].as_u64().unwrap_or(0).min(100) as u32,
                p["message"].as_str().unwrap_or("Processing recording"),
            );
            previous = p.clone();
        }
        match state["state"].as_str() {
            Some("succeeded") => {
                let downloaded=std::time::Instant::now();
                let envelope = control(target,connection,json!({"action":"result","request_id":identity,"result_format":"base64"})).await?;
                if envelope["request_id"] != identity { bail!("Result request identifier mismatch"); }
                let mut value = decode_result(&envelope)?;
                if value.get("segments").is_some() {
                    let analysis: super::types::Analysis = serde_json::from_value(value.clone()).context("Worker analysis schema is invalid")?;
                    analysis.validate()?;
                    if value["request_id"] != identity { bail!("Analysis does not belong to this processing request"); }
                    if value["audio_sha256"] != state["audio_sha256"] { bail!("Analysis does not belong to the submitted audio"); }
                    let observed=control(target,connection,json!({"action":"record_transfer","request_id":identity,"kind":"result_download","seconds":downloaded.elapsed().as_secs_f64()})).await?;
                    if !value["provenance"].is_object() {value["provenance"]=json!({});}
                    value["provenance"]["transport"]=observed["transport"].clone();
                    value["provenance"]["transport"]["source_result_sha256"]=envelope["result_sha256"].clone();
                }
                return Ok(value);
            }
            Some("cancelled") => {
                let _ = control(target,connection,json!({"action":"cleanup","request_id":identity})).await;
                bail!("Processing cancelled");
            }
            Some("failed") => return Err(anyhow!(worker_failure(target, &state, identity))),
            Some("created") => bail!("Upload stopped before job submission; resume with the original source recording to verify and retry its transfer"),
            Some("queued"|"running") => (),
            _ => return Err(anyhow!("Unknown worker job state")),
        }
        tokio::time::sleep(std::time::Duration::from_millis(
            if target.kind == TargetKind::Ssh {
                1200
            } else {
                250
            },
        ))
        .await;
    }
}

/// Continue observing a detached job after an SSH interruption.
pub async fn resume(
    target: &ProcessingTarget,
    identity: &str,
    cancelled: &AtomicBool,
    on_progress: impl Fn(&str, u32, &str) + Send + Sync,
) -> Result<Value> {
    identifier(identity)?;
    let connection = connect(target).await?;
    wait_result(target, &connection, identity, cancelled, &on_progress).await
}

/// Recover an interrupted upload without changing the immutable job request.
/// Running/terminal jobs are observed normally; created jobs accept only the
/// original checksum and are submitted under the same owned request ID.
pub async fn resume_with_audio(
    target: &ProcessingTarget,
    identity: &str,
    audio: &Path,
    cancelled: &AtomicBool,
    on_progress: impl Fn(&str, u32, &str) + Send + Sync,
) -> Result<Value> {
    identifier(identity)?;
    let connection = connect(target).await?;
    let state = control(
        target,
        &connection,
        json!({"action":"status","request_id":identity}),
    )
    .await?;
    if state["state"] == "created" {
        if cancelled.load(Ordering::Relaxed) {
            control(
                target,
                &connection,
                json!({"action":"cancel","request_id":identity}),
            )
            .await?;
            let _ = control(
                target,
                &connection,
                json!({"action":"cleanup","request_id":identity}),
            )
            .await;
            bail!("Processing cancelled before submission");
        }
        let expected = state["audio_sha256"]
            .as_str()
            .context("The interrupted request has no resumable audio checksum")?;
        if hash_audio(audio).await? != expected {
            bail!("The source recording changed; the interrupted upload was not resumed");
        }
        if let Err(error) = upload(
            target,
            &connection,
            identity,
            expected,
            audio,
            cancelled,
            "audio",
            &on_progress,
        )
        .await
        {
            if cancelled.load(Ordering::Relaxed) {
                control(
                    target,
                    &connection,
                    json!({"action":"cancel","request_id":identity}),
                )
                .await?;
                let _ = control(
                    target,
                    &connection,
                    json!({"action":"cleanup","request_id":identity}),
                )
                .await;
                return Err(error);
            }
            let latest = control(
                target,
                &connection,
                json!({"action":"status","request_id":identity}),
            )
            .await?;
            if latest["state"] == "created" {
                return Err(error);
            }
        }
        if cancelled.load(Ordering::Relaxed) {
            control(
                target,
                &connection,
                json!({"action":"cancel","request_id":identity}),
            )
            .await?;
            let _ = control(
                target,
                &connection,
                json!({"action":"cleanup","request_id":identity}),
            )
            .await;
            bail!("Processing cancelled before submission");
        }
        control(
            target,
            &connection,
            json!({"action":"submit","request_id":identity,"python":target.python}),
        )
        .await?;
    }
    wait_result(target, &connection, identity, cancelled, &on_progress).await
}

pub async fn status(target: &ProcessingTarget, identity: &str) -> Result<Value> {
    identifier(identity)?;
    control(
        target,
        &connect(target).await?,
        json!({"action":"status","request_id":identity}),
    )
    .await
}
pub async fn cancel(target: &ProcessingTarget, identity: &str) -> Result<Value> {
    identifier(identity)?;
    control(
        target,
        &connect(target).await?,
        json!({"action":"cancel","request_id":identity}),
    )
    .await
}
/// Removes only terminal jobs carrying a matching worker ownership marker.
pub async fn cleanup(target: &ProcessingTarget, identity: &str) -> Result<Value> {
    identifier(identity)?;
    control(
        target,
        &connect(target).await?,
        json!({"action":"cleanup","request_id":identity}),
    )
    .await
}
pub async fn probe(target: &ProcessingTarget) -> Result<Value> {
    maintenance(target, json!({"operation":"probe"}), |_, _, _| {}).await
}
/// Read-only platform discovery works before an optional runtime is installed.
pub async fn platform_info(target: &ProcessingTarget) -> Result<Value> {
    communicate(command(target,&["-c".into(),"import json,platform;print(json.dumps({'system':platform.system(),'architecture':platform.machine()}))".into()])?,vec![]).await
}
/// Qualify only the requested embedding adapter, without loading ASR models or
/// requiring the other speaker package's segmentation artifacts.
pub async fn probe_identification(target: &ProcessingTarget, model: &str) -> Result<Value> {
    if !["campplus", "wespeaker"].contains(&model) {
        bail!("Choose a supported identification model");
    }
    maintenance(
        target,
        json!({"operation":"probe","probe_scope":"embedding","embedding_model":model,
        "speaker_device":if model=="campplus"{"cpu"}else{"auto"}}),
        |_, _, _| {},
    )
    .await
}
pub async fn probe_config(target: &ProcessingTarget, config: &PipelineConfig) -> Result<Value> {
    maintenance(
        target,
        json!({"operation":"probe","config":config}),
        |_, _, _| {},
    )
    .await
}
pub async fn prepare(
    target: &ProcessingTarget,
    config: &PipelineConfig,
    on_progress: impl Fn(&str, u32, &str) + Send + Sync,
) -> Result<Value> {
    maintenance(
        target,
        json!({"operation":"prepare","config":config}),
        on_progress,
    )
    .await
}
/// Prepare only the requested curated optional artifacts. Local native assets
/// are owned by the application's existing model managers.
pub async fn prepare_artifacts(
    target: &ProcessingTarget,
    config: &PipelineConfig,
    ids: &[String],
    on_progress: impl Fn(&str, u32, &str) + Send + Sync,
) -> Result<Value> {
    maintenance(
        target,
        json!({"operation":"prepare","config":config,"artifact_ids":ids}),
        on_progress,
    )
    .await
}
pub async fn install(
    target: &ProcessingTarget,
    on_progress: impl Fn(&str, u32, &str) + Send + Sync,
) -> Result<Value> {
    maintenance(target, json!({"operation":"install_runtime"}), on_progress).await
}
/// Restore a previously verified code/dependency/native release atomically.
pub async fn rollback(
    target: &ProcessingTarget,
    on_progress: impl Fn(&str, u32, &str) + Send + Sync,
) -> Result<Value> {
    maintenance(
        target,
        json!({"operation":"install_runtime","rollback":true}),
        on_progress,
    )
    .await
}
/// Install a user-selected target-compatible archive, verifying its bytes on
/// both sides of SSH before isolated activation. The local path stays local.
pub async fn install_bundle(
    target: &ProcessingTarget,
    bundle: &Path,
    on_progress: impl Fn(&str, u32, &str) + Send + Sync,
) -> Result<Value> {
    maintenance(
        target,
        json!({"operation":"install_runtime","native_bundle_path":bundle}),
        on_progress,
    )
    .await
}

async fn maintenance(
    target: &ProcessingTarget,
    mut request: Value,
    on_progress: impl Fn(&str, u32, &str) + Send + Sync,
) -> Result<Value> {
    let identity = uuid::Uuid::new_v4().to_string();
    request["request_id"] = json!(identity);
    let result = execute(target, request, &AtomicBool::new(false), on_progress).await?;
    let _ = cleanup(target, &identity).await;
    Ok(result)
}

/// Synthetic transport fixture: real controller and immutable release, but no
/// model dependencies or private recordings. The probe succeeds; inference fails.
#[cfg(all(test, unix))]
pub(crate) fn test_failure_target(root: &Path) -> Result<ProcessingTarget> {
    use std::os::unix::fs::symlink;
    std::fs::create_dir_all(root)?;
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(root, std::fs::Permissions::from_mode(0o700))?;
    }
    let release = root.join("runtimes/fixture-release");
    std::fs::create_dir_all(release.join("bin"))?;
    std::fs::create_dir_all(release.join("worker"))?;
    let python = which::which("python3")?;
    symlink(python, release.join("bin/python"))?;
    for (name, source) in files().as_object().context("Missing bundled test files")? {
        std::fs::write(
            release.join("worker").join(name),
            source.as_str().context("Invalid bundled source")?,
        )?;
    }
    let fake = r#"import argparse,json,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--request');p.add_argument('--response');a=p.parse_args()
r=json.load(open(a.request));c=r.get('config',{})
if r.get('operation')=='probe':
 value=dict(version=1,protocol_version=1,operations=['process','speech_recognition','speaker_analysis','extract_embeddings','probe','prepare'],engines=[dict(id='faster_whisper',installed=True,available=True,verified=True,device='cpu',compute_type='int8_float32',model=c.get('speech_model','large-v3-turbo-int8'))],speaker_pipelines=[dict(id='community1',installed=True,available=True,verified=True,device='cpu')],embedding_models=[dict(id=model,installed=True,available=True,verified=True,device='cpu') for model in ('campplus','wespeaker')],artifacts=[dict(id=c.get('speech_model','large-v3-turbo-int8'),ready=True)],resolved=dict(speech_device='cpu',speaker_device='cpu',compute_type='int8_float32'),issues=[])
 Path(a.response).write_text(json.dumps(value));sys.exit(0)
raise RuntimeError('Intentional synthetic inference failure')
"#;
    std::fs::write(release.join("worker/worker.py"), fake)?;
    let result=std::process::Command::new("python3").args(["-c","import sys,json;sys.path.insert(0,sys.argv[1]);from protocol import bundle_description;print(json.dumps(bundle_description(sys.argv[1])))"]).arg(release.join("worker")).output()?;
    if !result.status.success() {
        bail!("Cannot hash synthetic worker release");
    }
    let mut description: Value = serde_json::from_slice(&result.stdout)?;
    description["version"] = json!(1);
    description["runtime_id"] = json!("fixture-release");
    description["runtime_verified"] = json!(true);
    std::fs::write(
        release.join("runtime.json"),
        serde_json::to_vec(&description)?,
    )?;
    symlink(&release, root.join("current"))?;
    Ok(ProcessingTarget {
        id: "fixture-worker".into(),
        name: "Synthetic worker".into(),
        kind: TargetKind::Local,
        worker_path: Some(root.to_string_lossy().into()),
        device: "cpu".into(),
        ..Default::default()
    })
}

/// Seed a completed synthetic result into the same failed owned request. This
/// exists only in tests; resume still uses the normal checksum/transport path.
#[cfg(all(test, unix))]
pub(crate) async fn seed_test_result(
    target: &ProcessingTarget,
    identity: &str,
    analysis: Value,
) -> Result<()> {
    identifier(identity)?;
    let connection = connect(target).await?;
    let program = Path::new(&connection.program)
        .parent()
        .context("Missing test controller directory")?;
    let code = r#"import sys,json
from pathlib import Path
sys.path.insert(0,sys.argv[1])
from supervisor import owned_job,update
from protocol import load_json,atomic_json,sha256
folder=owned_job(Path(sys.argv[2]),sys.argv[3]);request=load_json(folder/'request.json')
analysis=json.load(sys.stdin)
analysis['request_id']=request['request_id'];analysis['audio_sha256']=request['audio_sha256']
if not isinstance(analysis.get('settings'),dict):analysis['settings']={}
analysis['settings']['config']=request.get('config',{})
analysis['settings']['worker_operation']=request.get('operation','process')
atomic_json(folder/'result.json',analysis)
update(folder,state='succeeded',result_sha256=sha256(folder/'result.json'))
print(json.dumps({'version':1,'seeded':True}))
"#;
    communicate(
        command(
            target,
            &[
                "-c".into(),
                code.into(),
                program.to_string_lossy().into(),
                connection.root.clone(),
                identity.into(),
            ],
        )?,
        serde_json::to_vec(&analysis)?,
    )
    .await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[tokio::test]
    async fn upload_stream_reports_intermediate_bytes_and_preserves_contents() {
        let bytes = vec![42u8; 1024 * 1024];
        let mut source = bytes.as_slice();
        // A small transport buffer forces partial writes, as SSH backpressure does.
        let (mut sender, mut receiver) = tokio::io::duplex(1024);
        let events = std::sync::Mutex::new(Vec::new());
        let progress = |stage: &str, percent, message: &str| {
            events.lock().unwrap().push((stage.to_owned(), percent, message.to_owned()));
        };
        let read = async {
            let mut received = Vec::new();
            receiver.read_to_end(&mut received).await.unwrap();
            received
        };
        let (result, received) = tokio::join!(
            copy_upload(&mut source, &mut sender, bytes.len() as u64, "recording", &progress), read);
        result.unwrap();
        assert_eq!(received, bytes);
        let events = events.into_inner().unwrap();
        assert_eq!(events.first().unwrap().1, 0);
        assert_eq!(events.last().unwrap().1, 100);
        assert!(events.iter().any(|(_, p, _)| *p > 0 && *p < 100));
        assert!(events.windows(2).all(|e| e[0].1 <= e[1].1));
        assert!(events.last().unwrap().2.contains("1.0 / 1.0 MiB"));
        assert!(events.last().unwrap().2.contains("MiB/s"));
    }

    #[tokio::test]
    async fn changed_or_disconnected_upload_does_not_report_completion() {
        for disconnected in [false, true] {
            let mut source = b"fictional audio".as_slice();
            let (mut sender, mut receiver) = tokio::io::duplex(1024);
            receiver.shutdown().await.unwrap();
            let events = std::sync::Mutex::new(Vec::new());
            let progress = |_: &str, percent, _: &str| events.lock().unwrap().push(percent);
            if disconnected { drop(receiver); }
            let result = copy_upload(&mut source, &mut sender, 100, "recording", &progress).await;
            assert!(result.is_err());
            assert!(!events.into_inner().unwrap().contains(&100));
        }
    }

    #[tokio::test]
    async fn upload_can_be_cancelled_while_waiting_for_remote_confirmation() {
        let dir = tempfile::tempdir().unwrap();
        let program = dir.path().join("waiting_receiver.py");
        std::fs::write(&program, "import sys,time\nsys.stdin.buffer.read()\ntime.sleep(30)\n").unwrap();
        let audio = dir.path().join("audio.bin");
        std::fs::write(&audio, b"fictional audio").unwrap();
        let connection = Connection { program: program.to_string_lossy().into(), root: dir.path().to_string_lossy().into() };
        let cancelled = AtomicBool::new(false);
        let progress = |stage: &str, _: u32, _: &str| {
            if stage == "upload_verifying" { cancelled.store(true, Ordering::Relaxed); }
        };
        let target = ProcessingTarget::default();
        let result = tokio::time::timeout(std::time::Duration::from_secs(3),
            upload(&target, &connection, "fictional-upload", "unused", &audio, &cancelled, "audio", &progress))
            .await.expect("Cancellation must not wait for a remote acknowledgement");
        assert!(result.unwrap_err().to_string().contains("cancelled"));
    }

    #[test]
    fn model_access_failure_explains_authentication_on_the_selected_target() {
        let state = json!({"error_code":"model_access_denied", "error_artifact":"community1",
            "error":"fictional-secret-token and /private/diagnostics"});
        let mut target = ProcessingTarget::default();
        let local = worker_failure(&target, &state, "fictional-request");
        assert!(local.contains("Community-1 download failed"));
        assert!(local.contains("https://huggingface.co/pyannote/speaker-diarization-community-1"));
        assert!(local.contains("On this computer"));
        assert!(local.contains("hf auth login"));
        assert!(local.contains("Retry Download selected models"));
        assert!(!local.contains("reconnect"));
        assert!(!local.contains("fictional-secret-token"));
        assert!(!local.contains("/private/diagnostics"));
        target.kind = TargetKind::Ssh;
        let remote = worker_failure(&target, &state, "fictional-request");
        assert!(remote.contains("selected remote target, as its SSH user"));
        assert!(!remote.contains("On this computer"));
    }

    #[test]
    fn unknown_worker_failures_do_not_expose_diagnostics_or_suggest_reconnection() {
        for state in [
            json!({"error":"fictional-secret-token"}),
            json!({"error_code":"fictional-secret-token", "error_artifact":"community1"}),
            json!({"error_code":"model_access_denied", "error_artifact":"fictional-secret-token"}),
        ] {
            let message = worker_failure(&ProcessingTarget::default(), &state, "fictional-request");
            assert!(message.contains("job failed"));
            assert!(message.contains("fictional-request"));
            assert!(!message.contains("fictional-secret-token"));
            assert!(!message.contains("hf auth login"));
            assert!(!message.contains("reconnect"));
        }
    }

    #[test]
    fn remote_arguments_cannot_change_shell_syntax() {
        assert_eq!(quoted("a'b;$(false)"), "'a'\"'\"'b;$(false)'");
        assert!(identifier("../../other-job").is_err());
        assert!(identifier("fictional-job-123").is_ok());
    }
    #[test]
    fn returned_result_checksum_is_checked_before_parsing() {
        let bytes = br#"{"version":1,"fixture":true}"#;
        let mut envelope = json!({"result_base64":base64::engine::general_purpose::STANDARD.encode(bytes),
            "result_sha256":format!("{:x}",Sha256::digest(bytes))});
        assert_eq!(decode_result(&envelope).unwrap()["fixture"], true);
        envelope["result_sha256"] = json!("0".repeat(64));
        assert!(decode_result(&envelope).is_err());
    }
    #[cfg(unix)]
    #[tokio::test]
    async fn bootstrap_rejects_unrelated_existing_directory_without_changing_it() {
        use std::os::unix::fs::PermissionsExt;
        let temporary = tempfile::tempdir().unwrap();
        let root = temporary.path().join("unrelated");
        std::fs::create_dir(&root).unwrap();
        std::fs::set_permissions(&root, std::fs::Permissions::from_mode(0o755)).unwrap();
        std::fs::write(root.join("keep.txt"), "fictional unrelated contents").unwrap();
        let before = std::fs::metadata(&root).unwrap().permissions().mode();
        let target = ProcessingTarget {
            worker_path: Some(root.to_string_lossy().into()),
            ..Default::default()
        };
        let error = connect(&target)
            .await
            .err()
            .expect("Unrelated directory must be rejected")
            .to_string();
        assert!(error.contains("dedicated private worker subdirectory"));
        assert!(!error.contains("disconnected"));
        assert!(!error.contains(root.to_str().unwrap()));
        assert_eq!(
            std::fs::metadata(&root).unwrap().permissions().mode(),
            before
        );
        assert_eq!(
            std::fs::read_to_string(root.join("keep.txt")).unwrap(),
            "fictional unrelated contents"
        );
        assert_eq!(std::fs::read_dir(&root).unwrap().count(), 1);
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn parallel_cold_bootstraps_publish_one_complete_owned_bundle() {
        let temporary = tempfile::tempdir().unwrap();
        for round in 0..3 {
            let root = temporary.path().join(format!("cold-{round}"));
            let target = ProcessingTarget {
                worker_path: Some(root.to_string_lossy().into()),
                ..Default::default()
            };
            let mut calls = tokio::task::JoinSet::new();
            for _ in 0..12 {
                let target = target.clone();
                calls.spawn(async move { connect(&target).await.map(|c| c.program) });
            }
            let mut program = None;
            while let Some(result) = calls.join_next().await {
                let returned = result.unwrap().expect("Parallel bootstrap must succeed");
                if let Some(expected) = &program {
                    assert_eq!(&returned, expected);
                }
                program = Some(returned);
            }
            let folder = Path::new(program.as_ref().unwrap()).parent().unwrap();
            for (name, content) in files().as_object().unwrap() {
                assert_eq!(
                    std::fs::read_to_string(folder.join(name)).unwrap(),
                    content.as_str().unwrap()
                );
            }
            assert!(std::fs::read_dir(folder).unwrap().all(|p| !p
                .unwrap()
                .file_name()
                .to_string_lossy()
                .ends_with(".partial")));
        }
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn failed_model_download_keeps_actionable_error_through_rust_status_transport() {
        let temporary = tempfile::tempdir().unwrap();
        let program = temporary.path().join("program");
        let root = temporary.path().join("owned-runtime");
        std::fs::create_dir(&program).unwrap();
        std::fs::write(
            program.join("protocol.py"),
            include_str!("../../../../tools/offline_transcription/protocol.py"),
        )
        .unwrap();
        std::fs::write(
            program.join("supervisor.py"),
            include_str!("../../../../tools/offline_transcription/supervisor.py"),
        )
        .unwrap();
        std::fs::write(
            program.join("worker.py"),
            r#"import sys
from pathlib import Path
from protocol import atomic_json
response = Path(sys.argv[sys.argv.index('--response') + 1])
atomic_json(response.with_name(response.name + '.failure.json'), {
    'protocol_version': 1, 'error_code': 'model_access_denied', 'error_artifact': 'community1'
})
raise RuntimeError('fictional-secret-token in private failure')
"#,
        )
        .unwrap();
        let target = ProcessingTarget::default();
        let connection = Connection {
            program: program.join("supervisor.py").to_string_lossy().into(),
            root: root.to_string_lossy().into(),
        };
        let identity = "fictional-failed-worker";
        control(&target,&connection,json!({"action":"create","request":{"version":1,"operation":"probe","request_id":identity}})).await.unwrap();
        control(
            &target,
            &connection,
            json!({"action":"submit","request_id":identity}),
        )
        .await
        .unwrap();
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        loop {
            let state = control(
                &target,
                &connection,
                json!({"action":"status","request_id":identity}),
            )
            .await
            .unwrap();
            if state["state"] == "failed" {
                assert!(state["error"].is_string());
                assert_eq!(state["error_code"], "model_access_denied");
                break;
            }
            assert!(
                std::time::Instant::now() < deadline,
                "Fake worker did not terminate"
            );
            tokio::time::sleep(std::time::Duration::from_millis(40)).await;
        }
        let error = wait_result(
            &target,
            &connection,
            identity,
            &AtomicBool::new(false),
            &|_, _, _| {},
        )
        .await
        .unwrap_err()
        .to_string();
        assert!(error.contains("Community-1 download failed"));
        assert!(error.contains("hf auth login"));
        assert!(!error.contains("reconnect"));
        assert!(!error.contains("fictional-secret-token"));
        control(
            &target,
            &connection,
            json!({"action":"cleanup","request_id":identity}),
        )
        .await
        .unwrap();
    }
}
