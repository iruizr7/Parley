//! `whisper-helper`: a stdio JSON-lines whisper.cpp transcription sidecar.
//!
//! Native saved-recording adapter, originally introduced for issue #56.
//! Modelled directly on
//! `llama-helper/src/main.rs`'s request loop and lifecycle: read one
//! [`whisper_protocol::Request`] per line from stdin, reply with exactly one
//! terminal [`whisper_protocol::Response`] line on stdout, serve until stdin
//! closes or `Shutdown` arrives. Params-building and confidence scoring are
//! shared with the in-process engine via the `whisper-core` crate rather
//! than duplicated here — see that crate's doc comment.
//!
//! The saved-recording worker selects a build for its requested CPU, CUDA,
//! Vulkan or Metal backend. GPU builds verify actual device initialization
//! after decoding and reject silent CPU fallback. Local live recording keeps
//! its existing in-process engine.

use std::io::{self, BufRead, Write};
use std::sync::Mutex;
use std::time::Instant;

use anyhow::{anyhow, Result};
use base64::Engine;
use whisper_core::{DecodeConfig, TranscribeOptions};
use whisper_protocol::{Request, Response, Segment};
use whisper_rs::{WhisperContext, WhisperContextParameters};

struct GpuInitialization {
    device: Option<String>,
    failed: bool,
}
static GPU_INITIALIZATION: Mutex<GpuInitialization> = Mutex::new(GpuInitialization {
    device: None,
    failed: false,
});

fn observe_backend_log(message: &str, state: &mut GpuInitialization) {
    for line in message.lines() {
        if let Some(device) = line
            .strip_prefix("whisper_backend_init_gpu: using ")
            .and_then(|s| s.strip_suffix(" backend"))
        {
            state.device = Some(device.to_owned());
        }
        if line.starts_with("whisper_backend_init_gpu: failed to initialize ")
            || line == "whisper_backend_init_gpu: no GPU found"
        {
            state.failed = true;
        }
    }
}

unsafe extern "C" fn native_log(
    _level: whisper_rs::whisper_rs_sys::ggml_log_level,
    text: *const std::ffi::c_char,
    _user: *mut std::ffi::c_void,
) {
    if text.is_null() {
        return;
    }
    // whisper.cpp owns this NUL-terminated string for the callback duration.
    let message = unsafe { std::ffi::CStr::from_ptr(text) }.to_string_lossy();
    if let Ok(mut state) = GPU_INITIALIZATION.lock() {
        observe_backend_log(&message, &mut state);
    }
    let _ = io::stderr().write_all(message.as_bytes());
}

fn initialized_device() -> Result<String> {
    if compiled_backend() == "cpu" {
        return Ok("cpu".into());
    }
    let state = GPU_INITIALIZATION
        .lock()
        .map_err(|_| anyhow!("Device verification unavailable"))?;
    match (&state.device, state.failed) {
        (Some(device), false) => {
            // Validate the observed device belongs to the requested backend:
            // workspace feature unification can compile multiple backends.
            unsafe {
                use whisper_rs::whisper_rs_sys as sys;
                for index in 0..sys::ggml_backend_dev_count() {
                    let handle = sys::ggml_backend_dev_get(index);
                    let name = std::ffi::CStr::from_ptr(sys::ggml_backend_dev_name(handle))
                        .to_string_lossy();
                    if name.as_ref() == device {
                        let registry = sys::ggml_backend_dev_backend_reg(handle);
                        let backend =
                            std::ffi::CStr::from_ptr(sys::ggml_backend_reg_name(registry))
                                .to_string_lossy()
                                .to_lowercase();
                        if backend == compiled_backend() {
                            return Ok(device.clone());
                        }
                    }
                }
            }
            Err(anyhow!(
                "Initialized device does not match the requested {} backend",
                compiled_backend()
            ))
        }
        _ => Err(anyhow!(
            "Requested {} GPU initialization was not confirmed; refusing CPU fallback",
            compiled_backend()
        )),
    }
}

#[cfg(test)]
mod device_tests {
    use super::*;
    #[test]
    fn compiled_or_discovered_gpu_is_not_initialization_evidence() {
        let mut state = GpuInitialization {
            device: None,
            failed: false,
        };
        observe_backend_log(
            "whisper_backend_init_gpu: found GPU device 0: Example (type: 1, cnt: 0)",
            &mut state,
        );
        assert!(state.device.is_none());
        observe_backend_log(
            "whisper_backend_init_gpu: using Example backend\n",
            &mut state,
        );
        assert_eq!(state.device.as_deref(), Some("Example"));
        observe_backend_log(
            "whisper_backend_init_gpu: failed to initialize Example backend\n",
            &mut state,
        );
        assert!(state.failed);
    }
    #[test]
    fn cpu_fallback_is_recorded_as_failure() {
        let mut state = GpuInitialization {
            device: None,
            failed: false,
        };
        observe_backend_log("whisper_backend_init_gpu: no GPU found\n", &mut state);
        assert!(state.failed);
    }
}

/// Which backend this binary was compiled for — self-reported in `Probe`
/// responses and used by `backend_probe` (the future selection logic) to
/// label results without having to parse the binary's filename.
fn compiled_backend() -> &'static str {
    if cfg!(feature = "cuda") {
        "cuda"
    } else if cfg!(feature = "vulkan") {
        "vulkan"
    } else if cfg!(feature = "metal") {
        "metal"
    } else {
        "cpu"
    }
}

struct State {
    ctx: Option<WhisperContext>,
    decode_config: DecodeConfig,
}

impl State {
    fn new() -> Self {
        Self {
            ctx: None,
            decode_config: DecodeConfig::default(),
        }
    }

    fn load_model(&mut self, path: &str) -> Result<()> {
        if let Ok(mut observed) = GPU_INITIALIZATION.lock() {
            observed.device = None;
            observed.failed = false;
        }
        let context_param = WhisperContextParameters {
            use_gpu: cfg!(any(feature = "cuda", feature = "vulkan", feature = "metal")),
            gpu_device: 0,
            ..Default::default()
        };
        let ctx = WhisperContext::new_with_params(path, context_param)
            .map_err(|e| anyhow!("failed to load model {path}: {e}"))?;
        self.ctx = Some(ctx);
        Ok(())
    }

    fn transcribe(
        &self,
        samples: Vec<f32>,
        language: Option<&str>,
        context_prompt: Option<&str>,
        max_threads: Option<i32>,
        greedy: bool,
    ) -> Result<whisper_core::TranscriptionOutcome> {
        let ctx = self
            .ctx
            .as_ref()
            .ok_or_else(|| anyhow!("no model loaded"))?;
        let result = whisper_core::transcribe_pcm16k(
            ctx,
            samples,
            language,
            context_prompt,
            self.decode_config,
            TranscribeOptions {
                max_threads,
                greedy,
            },
        )?;
        initialized_device()?;
        Ok(result)
    }
}

fn send_response(response: &Response) -> Result<()> {
    let json = serde_json::to_string(response)?;
    println!("{json}");
    io::stdout().flush()?;
    Ok(())
}

/// A short burst of near-silent samples (1s @ 16kHz) used by `Probe` when no
/// real model/audio is available — enough to exercise `full()` end-to-end
/// (model load + a decode pass) without needing a recorded clip on disk.
fn synthetic_probe_samples() -> Vec<f32> {
    vec![0.0f32; whisper_core::WHISPER_MIN_INPUT_SAMPLES]
}

fn handle_request(state: &mut State, request: Request) -> Result<Response> {
    match request {
        Request::LoadModel { path } => {
            let start = Instant::now();
            match state.load_model(&path) {
                Ok(()) => Ok(Response::Loaded {
                    error: None,
                    load_ms: start.elapsed().as_millis() as u64,
                }),
                Err(e) => Ok(Response::Loaded {
                    error: Some(e.to_string()),
                    load_ms: start.elapsed().as_millis() as u64,
                }),
            }
        }
        Request::Transcribe {
            samples_b64,
            language,
            context_prompt,
            max_threads,
            greedy,
        } => {
            let bytes = base64::engine::general_purpose::STANDARD
                .decode(samples_b64)
                .map_err(|e| anyhow!("invalid base64 samples: {e}"))?;
            if bytes.len() % 4 != 0 {
                return Ok(Response::Transcribed {
                    text: String::new(),
                    confidence: 0.0,
                    is_partial: false,
                    segments: vec![],
                    error: Some("sample buffer length is not a multiple of 4 bytes (f32le)".into()),
                    decode_ms: 0,
                });
            }
            let samples: Vec<f32> = bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();

            let start = Instant::now();
            match state.transcribe(
                samples,
                language.as_deref(),
                context_prompt.as_deref(),
                max_threads,
                greedy.unwrap_or(false),
            ) {
                Ok(outcome) => Ok(Response::Transcribed {
                    text: outcome.text,
                    confidence: outcome.confidence,
                    is_partial: outcome.is_partial,
                    segments: outcome
                        .segments
                        .into_iter()
                        .map(|s| Segment {
                            text: s.text,
                            confidence: s.confidence,
                            start_ms: s.start_ms,
                            end_ms: s.end_ms,
                        })
                        .collect(),
                    error: None,
                    decode_ms: start.elapsed().as_millis() as u64,
                }),
                Err(e) => Ok(Response::Transcribed {
                    text: String::new(),
                    confidence: 0.0,
                    is_partial: false,
                    segments: vec![],
                    error: Some(e.to_string()),
                    decode_ms: start.elapsed().as_millis() as u64,
                }),
            }
        }
        Request::Unload => {
            state.ctx = None;
            Ok(Response::Unloaded)
        }
        Request::Ping => Ok(Response::Pong),
        Request::Probe { model_path } => {
            let backend = compiled_backend().to_string();
            match model_path {
                None => Ok(Response::ProbeResult {
                    backend,
                    initialized_device: None,
                    device_verified: false,
                    decode_ok: None,
                    detail: "no model available; probe verified process spawn + ping only".into(),
                }),
                Some(path) => {
                    let mut probe_state = State::new();
                    let load_result = probe_state.load_model(&path);
                    if let Err(e) = load_result {
                        return Ok(Response::ProbeResult {
                            backend,
                            initialized_device: None,
                            device_verified: false,
                            decode_ok: Some(false),
                            detail: format!("model load failed: {e}"),
                        });
                    }
                    match probe_state.transcribe(synthetic_probe_samples(), None, None, None, true)
                    {
                        Ok(_) => Ok(Response::ProbeResult {
                            backend,
                            initialized_device: initialized_device().ok(),
                            device_verified: initialized_device().is_ok(),
                            decode_ok: Some(initialized_device().is_ok()),
                            detail: initialized_device()
                                .map(|_| {
                                    "1s synthetic decode and device initialization succeeded".into()
                                })
                                .unwrap_or_else(|e| e.to_string()),
                        }),
                        Err(e) => Ok(Response::ProbeResult {
                            backend,
                            initialized_device: None,
                            device_verified: false,
                            decode_ok: Some(false),
                            detail: format!("synthetic decode failed: {e}"),
                        }),
                    }
                }
            }
        }
        Request::Shutdown => Ok(Response::Goodbye),
    }
}

fn main() -> Result<()> {
    // See llama-helper's main.rs: writing to a stdout whose reader has gone
    // away must not panic on EPIPE.
    unsafe {
        libc::signal(libc::SIGPIPE, libc::SIG_DFL);
        whisper_rs::whisper_rs_sys::whisper_log_set(Some(native_log), std::ptr::null_mut());
    }

    eprintln!("whisper-helper ({}) starting", compiled_backend());

    let mut state = State::new();
    let stdin = io::stdin();
    let mut stdin_lock = stdin.lock();
    let mut buffer = String::new();

    loop {
        buffer.clear();
        match stdin_lock.read_line(&mut buffer) {
            Ok(0) => {
                eprintln!("EOF received, shutting down");
                break;
            }
            Ok(_) => {
                let line = buffer.trim();
                if line.is_empty() {
                    continue;
                }
                let is_shutdown;
                match serde_json::from_str::<Request>(line) {
                    Ok(request) => {
                        is_shutdown = matches!(request, Request::Shutdown);
                        match handle_request(&mut state, request) {
                            Ok(response) => send_response(&response)?,
                            Err(e) => send_response(&Response::Error {
                                message: e.to_string(),
                            })?,
                        }
                    }
                    Err(e) => {
                        is_shutdown = false;
                        eprintln!("failed to parse request: {e}");
                        send_response(&Response::Error {
                            message: format!("invalid request: {e}"),
                        })?;
                    }
                }
                if is_shutdown {
                    break;
                }
            }
            Err(e) => {
                eprintln!("error reading stdin: {e}");
                break;
            }
        }
    }

    eprintln!("whisper-helper exiting");
    Ok(())
}
