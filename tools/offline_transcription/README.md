# Saved-recording worker

The application invokes this versioned worker through
`audio::processing::executor`. It supports local and SSH targets for saved
recordings. Live recognition remains in Parley's local audio pipeline.
Dependencies and models are installed only through explicit target setup.

New media imports extract audio before saving the meeting copy, so remote jobs
upload audio rather than the original video. Existing compressed audio streams
are copied without re-encoding. Supported integer PCM imports use lossless FLAC
without changing sample rate or channels; float and 32-bit PCM are preserved
without reducing precision. The saved audio is the source for playback, content
hashes and retries. Previously saved recordings are not rewritten automatically.

## Engines and packaged speaker analysis

| Stage | Explicit execution devices | Artifacts |
|---|---|---|
| whisper.cpp through `whisper-helper` | CPU; CUDA, Vulkan or Metal when the matching helper is installed and its decode probe passes | Turbo floating-point GGML or Q5_0 |
| faster-whisper 1.2.1 / CTranslate2 4.7.1 | CPU or CUDA | Turbo FP16 or INT8 weights |
| MLX Whisper 0.4.3 / MLX 0.29.3 | Apple GPU | Turbo FP16; Q4 remains disabled |
| Sherpa through `parley_native_worker` | CPU | Existing segmentation, CAMPPlus and native clustering package |
| Community-1 / pyannote.audio 4.0.4 | CPU, CUDA, or artifact/version-qualified Apple MPS | Bundled segmentation, WeSpeaker and VBx/PLDA package |

These are adapter paths, not a claim that every platform/driver combination has
passed quality qualification. A setup probe reports `verified: true` only after
actual inference with installed model weights on the requested device. An import
check alone does not enable a choice. MPS requires an explicitly accepted CPU parity
qualification for the exact Community artifact, pinned packages, OS family and
architecture. The tested OS release remains recorded as provenance; macOS updates
do not invalidate that model/package approval. Execution probes still test the
selected GPU on the current OS, and a failed MPS probe leaves it unavailable
rather than silently substituting CPU.
Without that qualification the probe reports CPU explicitly. Requesting MPS
without a matching qualification fails; inference never changes its requested device.
The application shares availability results for its current session, keyed by
target connection/runtime settings and pipeline choices. Opening settings or an
import dialog reuses the startup check, including one already in progress.
Changing those choices checks the new combination. Explicit availability checks,
model preparation and worker changes invalidate the target's saved checks;
restarting the application starts a new session. Meeting-specific identity
evidence is still read afresh. Runtime-only resource probes are not cached.
Speech and speaker models load sequentially, and a target-wide execution lock
prevents simultaneous worker jobs from competing for GPU memory. Other services
are never stopped or reconfigured.

Stored weights and execution precision are distinct. Faster-whisper requests
`float16` or `int8_float16` on CUDA and `float32` or `int8_float32` on CPU, and checks the
runtime's resolved compute type. Native backend computation is recorded as
backend-dependent rather than mislabeled FP16. MLX uses greedy decoding; its
speed is not equivalent to beam-5 recognition. MLX disables previous-text
conditioning to avoid long-form alignment failure loops. If an aligner returns
a zero-width word, its original recognized text remains in its actual parent
segment; no word boundaries are invented. No vocabulary prompts or saved
substitutions are applied.

## Install/update and rollback

The application's Install/update action deploys the scripts and runs the same
installer as this standalone command:

```sh
python3 setup_runtime.py --directory /absolute/path/private-worker --device cpu
# Other explicit target devices: cuda, metal, vulkan.
python3 setup_runtime.py --directory /absolute/path/private-worker --rollback
```

`worker_path` is a **directory**, not a Python script pathname. SSH destinations
are OpenSSH aliases or `ssh://user@example.invalid:2222` URIs. The transport uses
argument-quoted SSH commands and stdin uploads, retains host-key verification,
and reuses existing OpenSSH credentials. No SSH keys are copied into the worker.
The worker root must be a dedicated private directory. A fresh leaf and any
missing parents are created with mode 0700 and an ownership marker bound to the
canonical folder path and user. Device numbers and inodes are not persisted in
new markers. Old markers migrate automatically when their owner and directory
inode still match, even if a reboot changed the device number. Migration keeps
the installed runtime, models and jobs. Setup never changes an existing directory's permissions. Home, filesystem
root and shared system directories are rejected, as are unrelated existing
folders. An unmarked legacy root is adopted only when its current release is
contained under its own `runtimes` directory and its complete code/native hashes
verify. An incomplete legacy folder requires a new dedicated worker directory.

`auto` is a setup/probe policy: it resolves verified stage devices before a job
starts. Inference rejects unresolved Auto stages. Auto installation selects a
platform runtime explicitly and reports that choice.

The installer bootstraps pinned uv from its published wheel with SHA256
verification, creates a managed Python 3.12 environment, installs pinned platform
libraries and a packaged FFmpeg executable, records every installed dependency,
and runs device checks. It never installs global packages. Each successful
release snapshots worker code, its independent environment and native helpers;
activation changes `current` atomically and retains `previous`. Submitted jobs
keep their exact interpreter and script snapshot when that link changes.
Rollback verifies the previous code/native hashes before activation. Older
releases without a saved code snapshot explicitly report that full rollback
cannot be proven. An installation failure keeps the existing release.
Worker snapshots declare a versioned exact filename/checksum manifest. Current
releases use the eight-file version-2 bundle. Existing complete seven-file
snapshots retain the original known version-1 digest scheme, so updating does
not invalidate their rollback proof. Unknown file sets or changed hashes fail
verification; the installer never guesses a snapshot's contents.
Models use a shared cache, so rollback does not download them again. Top-level
pins live in `setup_runtime.py`; each release's `installed-requirements.txt`
records the complete resolved dependency set.

Native helpers must also be packaged for the target architecture. Build
`parley_native_worker` with `cargo build --locked --release -p parley-native-worker`
from the same Parley source and the appropriate
`whisper-helper` backend. Include their shared libraries. The installer accepts
an explicitly supplied flat tar archive containing regular files:

```sh
python3 setup_runtime.py --directory /absolute/path/private-worker --device cpu \
  --native-bundle /absolute/path/helpers.tar --native-sha256 EXPECTED_SHA256
```

Helpers are named `parley_native_worker` and `whisper-helper-cpu`,
`whisper-helper-cuda`, `whisper-helper-vulkan` or `whisper-helper-metal`.
Manage targets also offers a local native-bundle picker. `install_bundle` streams
that selected archive through an owned job and verifies its SHA256 on both sides.
Before activation, the installer executes the speaker helper library/protocol
probe and speech helper ping/backend protocol. These checks prove runnability;
full model/device inference must still pass Check setup.

A missing helper stays an explicit setup issue; installing Python does not
silently enable a native engine. Probe results distinguish installed runtime
libraries, prepared artifacts and verified inference.

## Model preparation

Prepare/download resolves curated immutable revisions in `assets.py` and records
content hashes. Normal jobs are offline and reject missing or changed artifacts.
Hugging Face model-license acceptance and credentials must be configured outside
job requests. Credentials never belong in target settings, manifests or logs.

Existing model assets may be registered and reused without modification:

```text
assets/
  models/precise/model.bin
  models/fast/model.bin
  pyannote/config.yaml
  pyannote/embedding/pytorch_model.bin
  ...
```

`assets_dir` points to `assets`. A preparation request can instead supply exact
`model_paths` by artifact ID. `artifact_ids` limits preparation to particular
curated artifacts, for example `['community1']` for an embedding-only workflow.
Default preparation downloads the selected speech model, speaker package and any
additional selected identity model. Missing artifacts in that explicit Prepare
operation may be downloaded; inference never downloads implicitly.

## Protocol version 1

A request is a JSON object with `version: 1`, an immutable `request_id`, an
operation, pipeline `config`, and explicit `speech_device` / `speaker_device`.
The executor supplies audio SHA256, checks the upload and snapshots the target.

| Operation | Inputs and behavior |
|---|---|
| `process` | Audio, language and expected speaker count; runs speaker analysis and speech recognition |
| `speech_recognition` | Audio and language; recognition only, without speaker models, regrouping or annotation |
| `speaker_analysis` | Audio plus existing segments; speaker analysis without ASR |
| `extract_embeddings` | Audio, selected model, disjoint clean `source_intervals`, optional overlap masks; no ASR or regrouping |
| `prepare` | Configuration or explicit artifact IDs; downloads/registers models |
| `probe` | Checks installed versions, real device execution and available artifacts |

The aliases `transcribe`, `diarize`, and `embed` are accepted for protocol
migration. Results retain Analysis version 1 with additive runtime provenance,
request/audio identity, exact embedding-space metadata and source intervals.
Output validates finite numbers, timeline bounds, ordering, speaker references,
vector normalization, per-space dimensions and requested source containment
before import. Result and terminal status are written atomically.

`probe.operations` advertises the operations supported by that installed worker
release. Protocol version 1 alone does not imply support for later additive
operations. A local faster-whisper or MLX job with the app's linked Sherpa
package requires `speech_recognition`; an older immutable worker that omits it
must be updated explicitly in target setup before processing. The app never
substitutes another engine or silently rewrites that installed release.

`platform_info` reads only OS and architecture using bootstrap Python; it does
not install a runtime or load models. This lets selectors offer compatible
installable engines before their artifacts exist. Selected-engine readiness
still requires actual installed inference.

`probe_scope: "embedding"` with `embedding_model: "campplus"` or `"wespeaker"`
checks only that standalone extractor on synthetic audio. Its `embedding_models`
entry reports the actual resolved device and exact space, without publishing a
vector or requiring ASR/segmentation weights. CAMPPlus executes on CPU even on
a CUDA target; WeSpeaker uses CPU, CUDA or the specifically qualified MPS path.
This is numerical/device readiness, not proof of identification accuracy.

For this local combination, the worker returns ASR data and
`settings.worker_operation: "speech_recognition"`; the linked native package
then analyzes speakers on the same computer. A portable native helper is not
required. Degenerate word alignment retains the original segment text and
timing. Conversely, linked native ASR can supply `upstream_provenance`,
`upstream_recognition_device`, `upstream_compute_type`, and
`upstream_recognition_seconds` to a speaker-only worker request. Its terminal
result retains those fields, including after reconnection, and includes upstream
time exactly once; `provenance.timings_seconds.worker_elapsed_seconds` records
the worker's own time separately.

Community-1 preserves its regular overlap-capable speaker timeline. Voice
extraction uses only sufficiently long intervals outside overlap, through the
package's embedding adapter. The first interval is numerically checked against
its `get_embeddings` reference path on each extraction. The resulting space is
separate from pooled diarization centroids. Its checkpoint, preprocessing,
normalization and aggregation policy are versioned. CAMPPlus delegates to the
same native extractor used by local Parley; any trims are recorded as actual
source intervals. Insufficient clean evidence yields no vector.

## Cancellation, disconnection and recovery

Each job owns a private directory and an ownership marker. A detached supervisor
retains its terminal result when SSH disconnects. `executor::status` and
`executor::resume` reconnect using the same request ID. Repeating a request with
that ID returns the existing result; different bytes or settings are rejected.
If transfer stopped before submission, the owned job remains `created`.
`resume_with_audio` verifies the original checksum and retries that upload under
the same identifier; observing without source audio returns an actionable error.
Cancellation before submission removes media without starting inference.
The application publishes successful results atomically and then calls
`executor::cleanup`. Result SHA256 is verified before retrieval. Failed jobs keep
private diagnostic logs and never replace the previous transcript.

Cancellation signals the held worker process group, including FFmpeg and native
helpers. Queued jobs observe the same cancellation marker. Uploaded and
normalized media are removed on every terminal outcome. Cleanup accepts only
owned, terminal jobs; it never follows a supplied path or removes a live job.
A disconnected cancellation request reports the interruption and retains the ID
for explicit recovery. There is no target, engine or precision fallback.

## Resource and transfer measurements

Analysis preserves measurements under `provenance.resources` and
`provenance.transport`; the application also retains them in processing history.
Host RSS high-water marks are sampled through result assembly. Worker-process
RSS and POSIX `RUSAGE_CHILDREN` are separate platform-defined measurements; the
latter covers waited-for FFmpeg/native children and is not an isolated helper
peak or a simultaneous process-tree total. `native_helper_peak_rss_bytes` remains
unknown. Do not sum these values or interpret RSS as GPU memory: unified/device
mappings may overlap host RSS.

GPU stage measurements identify their allocator and scope. PyTorch CUDA
allocated/reserved peaks and the MLX allocator peak are collected when their APIs
exist. Reserved memory contains allocated memory. CTranslate2, native backends
and MPS allocator peaks remain explicitly unknown where no supported peak API is
available; unknown is never reported as zero. These counters are not whole-device
VRAM usage and must not be added to RSS or other allocator/process peaks.

Setup probes report a host-global available-memory estimate from Linux
`MemAvailable` or Apple's free/inactive/speculative pages. CUDA reports a global
free/total device snapshot when available. Apple unified memory has no equivalent
global free-GPU measurement, so that value remains unknown and host availability
is reported separately. These snapshots do not reserve memory.

Transport records elapsed wall-clock seconds for the first completed audio upload
and first result download, including the transport session and validation work.
Interrupted attempts are excluded. Later reconnects retain those same first
measurements; fetching a result never rewrites its immutable raw worker bytes.

## Validation and comparisons

```sh
python3 -m unittest discover -s tools/offline_transcription -v
```

Synthetic tests cover schema rejection, source intervals, rollback, failed
activation, real detached process lifecycle, reconnect/retry, process-group
cancellation and owned cleanup without downloading models.

`compare.py` consumes private result files and an optional manually corrected
reference to calculate word error rate. Its stdout uses a public-safe allowlist of
numeric metrics and curated choices with generic run labels. An explicitly
requested report file retains private settings and must stay outside the checkout. Keep recordings, transcripts, embeddings,
reference annotations and raw benchmark reports outside the checkout. Compare
speaker pipelines with recognition fixed; compare recognition with the same
checkpoint and document decoder differences. NoScribe output is a comparison,
not ground truth. `benchmark_mlx.py` is retained as an old standalone measurement
entry point; application execution uses the worker, not that benchmark.

## Qualify Apple speaker acceleration

`qualify_speaker_device.py` runs the same pinned adapter without ASR. Its MPS
path is experimental and requires CPU fallback to be disabled. Compare the
same audio, model artifacts and speaker count against an existing CPU result:

```sh
python qualify_speaker_device.py --audio PRIVATE_AUDIO --assets MODEL_ASSETS \
  --cache-dir MODEL_CACHE --device mps --baseline PRIVATE_CPU_RESULT \
  --response PRIVATE_MPS_RESULT --comparison PRIVATE_COMPARISON
python accept_mps_qualification.py --assets MODEL_ASSETS --cache-dir MODEL_CACHE \
  --comparison PRIVATE_COMPARISON --analysis PRIVATE_MPS_RESULT
```

The second command is an explicit acceptance action after inspecting the private
comparison. It requires matching speaker counts, tight permutation-independent
timeline/overlap parity, and fixed-clean-interval embedding parity against the
same CPU extractor. It records an artifact/package/OS-specific attestation in
the private cache. The product probe then verifies actual MPS execution again.
CPU agreement does not establish ground-truth diarization accuracy; review the
speaker/overlap output before accepting a comparison. Model, package, adapter
or OS changes invalidate the attestation.
