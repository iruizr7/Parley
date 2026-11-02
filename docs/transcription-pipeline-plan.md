# Configurable transcription pipelines and speaker identities

Status: implementation complete, with initial acceptance checks and bounded
human-reviewed quality comparisons. Updated 2026-10-28. Capability tables describe
supported implementations, not a quality certification for every combination.
Identity matching remains review-only until its thresholds are calibrated.

## Product decisions

Replace the current Transcription settings controls with four pipeline fields:
**Target**, **Speech engine**, **Speech model**, and **Speaker pipeline**.
Speaker detection, its embedding model, and within-meeting clustering form one
validated package: initially **Sherpa-ONNX** or **Community-1**.

Add a separate **Speaker identification** selector below the pipeline. It controls
matching meeting speakers to saved identities and can be changed independently.
Extend the existing Speakers page to manage enrollment evidence for each embedding
model and play the source audio used to learn a voice.

Preserve recording and local live transcription. Introduce remote processing for
saved recordings first. Keep language and expected speaker count in the meeting's
processing dialog. Keep the recording-only toggle in Recording settings.

## Speech models: equivalence and precision

Three separate properties must be recorded and explained:

1. **Model family/checkpoint:** for example, Whisper large-v3-turbo.
2. **Stored weights:** FP16, INT8, Q5, or an engine-specific quantization.
3. **Execution precision:** what the runtime actually uses on the selected device.

The inspected noScribe 0.7.1 distribution documents Precise as large-v3-turbo
converted to CTranslate2 with FP16 weights, and Fast as the same model quantized
to INT8. Neither is FP8. Its application passes a configurable compute type,
defaulting to `default`; do not infer actual execution precision from the menu
label. The Precise conversion is documented by its
[model publisher](https://huggingface.co/dropbox-dash/faster-whisper-large-v3-turbo),
formerly published under `mobiuslabsgmbh`.

CTranslate2 separates stored precision from computation and can convert weights
on load. Its documented CPU fallback for FP16 is FP32. On supported NVIDIA GPUs,
FP16 and mixed INT8/FP16 execution are available. Expanding INT8 weights to FP32
does not recover information lost during quantization.
[CTranslate2 precision documentation](https://opennmt.net/CTranslate2/quantization.html).

### Model equivalence table

These rows identify counterparts from the same underlying checkpoint. They do
not promise identical words, timestamps, memory use, or accuracy across engines.

| Choice | faster-whisper / CTranslate2 | whisper.cpp | MLX Whisper on Apple Silicon |
| --- | --- | --- | --- |
| large-v3-turbo, non-quantized baseline | FP16 artifact; typically FP16 on CUDA or FP32 on CPU. Counterpart to noScribe Precise. | Non-quantized `ggml-large-v3-turbo.bin`, using floating-point weights and backend-specific computation. | `mlx-community/whisper-large-v3-turbo`; inspected floating tensors are FP16. |
| large-v3-turbo, compact | INT8 artifact; `int8_float16` on compatible CUDA hardware, or supported INT8 CPU computation. Counterpart to noScribe Fast. | `ggml-large-v3-turbo-q5_0.bin`. A compact counterpart, **not** an INT8-equivalent representation. | `mlx-community/whisper-large-v3-turbo-q4`, a published 4-bit variant. Candidate until validated with the pinned worker. |
| Full large-v3, optional future comparison | A separate, larger Whisper checkpoint, available through faster-whisper. | A separate large-v3 model. | Requires a corresponding MLX conversion. Outside the initial curated menu. |

The [MLX Turbo artifact](https://huggingface.co/mlx-community/whisper-large-v3-turbo)
was checked at revision `a4aaeec0636e6fef84abdcbe3544cb2bf7e9f6fb`: its
Safetensors header contains 586 F16 tensors and one I64 tensor. The
[MLX Q4 configuration](https://huggingface.co/mlx-community/whisper-large-v3-turbo-q4/blob/main/config.json)
specifies 4-bit quantization with group size 64. Q4 is not FP8 or INT8.
Whisper.cpp documents its model conversion and quantization separately in its
[README](https://github.com/ggml-org/whisper.cpp).

Start with Turbo floating-point and the validated compact variant for each
engine. Use explicit labels such as `Whisper large-v3-turbo — FP16 weights`,
`Whisper large-v3-turbo — INT8`, and `Whisper large-v3-turbo — Q5_0`.
Avoid a universal “Precise/Fast” label: it conceals material differences and does
not guarantee speed or quality. Show resolved execution precision underneath.

### Apple Silicon and other execution targets

| Target capabilities | whisper.cpp | faster-whisper | MLX Whisper |
| --- | --- | --- | --- |
| CPU-only supported host | CPU | CPU | Not offered for this target class |
| Linux with Vulkan-capable GPU | Vulkan, when built and probed successfully | CPU; no Vulkan path in the proposed worker | Unavailable |
| Apple Silicon | Metal or CPU, including floating-point Turbo and Q5 variants | CPU in the initial supported matrix | Apple GPU, initially FP16 Turbo; Q4 after validation |
| Linux NVIDIA | CUDA when the installed helper supports it; CPU fallback only if selected explicitly | CUDA or CPU | Unavailable |

Apple Silicon does not require MLX: whisper.cpp also supports Apple GPU execution.
Conversely, selecting faster-whisper does not imply CPU-only processing; it uses
CTranslate2's supported accelerators. These are the initial compatibility rules,
not an exhaustive inventory of every upstream backend.
[whisper.cpp hardware support](https://github.com/ggml-org/whisper.cpp),
[CTranslate2 hardware support](https://opennmt.net/CTranslate2/hardware_support.html),
[MLX Whisper implementation](https://github.com/ml-explore/mlx-examples/tree/main/whisper).

Probe actual installed builds, model formats, available memory, and device
execution. A machine's label is not a capability guarantee. Speech and speaker
analysis may use different devices on the same target. Community-1's CPU and
CUDA paths are the initial reference configurations; Apple MPS acceleration must
pass a version-specific smoke test and quality comparison before being offered
as supported. MLX speech acceleration does not automatically accelerate pyannote.

## Transcription settings and meeting workflow

| Field | Choices and behavior |
| --- | --- |
| Target | Local by default; named remote targets; Manage targets action. |
| Speech engine | whisper.cpp, faster-whisper, MLX Whisper, filtered by target capabilities. |
| Speech model | Curated compatible Turbo artifacts, with inline Download/Ready status. |
| Speaker pipeline | Sherpa-ONNX or Community-1, with their packaged embedding and clustering configuration. |
| Speaker identification | Same as speaker pipeline (default), Off, CAMPPlus (Sherpa), or WeSpeaker (Community-1). |

Remove the old active-model control, catalog UI, and custom-vocabulary UI. Retain
the internal artifact registry. Stop applying saved vocabulary substitutions and
prompts; preserve their stored data without silently using it. Move runtime paths
and environment diagnostics into target setup, rather than adding more main fields.

Changing a field recalculates compatible choices. Keep the current downstream
choice if valid; otherwise clearly mark it as needing selection. Never silently
substitute an engine or quantization. Show a concise readiness summary including
each stage's actual device and any required downloads. Disable processing until
the requested configuration is available.

Settings define defaults for future jobs. Process recording and Reprocess offer
meeting-specific overrides, language, and expected speaker count. Snapshot these
choices when a job starts. Subsequent settings changes cannot affect that job.

Local live transcription remains available as a clearly labeled preview. If the
chosen final pipeline differs, show that distinction before recording and run the
final pipeline after stopping, according to the user's processing preference.
Record preview and final provenance separately. Recording-only mode skips live
inference and opens the processing dialog afterward. No remote streaming in this
first implementation.

## Speaker pipelines and independent identity matching

| Speaker pipeline | Embedding package | Within-meeting clustering |
| --- | --- | --- |
| Sherpa-ONNX | Existing CAMPPlus checkpoint and preprocessing | Existing native clustering/refinement, versioned with the pipeline |
| Community-1 | Its bundled WeSpeaker checkpoint and preprocessing | Its calibrated VBx/PLDA assets and parameters |

Community-1 is a complete diarization pipeline. Preserve its overlap-capable
timeline; its exclusive timeline may assist primary-speaker assignment but must
not replace overlap information. Overlap detection does not separate mixed audio
or guarantee recognition of every simultaneous utterance.
[Community-1 model documentation](https://huggingface.co/pyannote/speaker-diarization-community-1).

Keep three operations distinct in the UI and implementation:

- **Re-detect speakers:** changes who spoke when and can regroup anonymous speakers.
  Runs the selected speaker pipeline against audio, without requiring new ASR.
- **Identify speakers:** assigns saved identities to existing speaker groups.
  Does not alter words, timestamps, or group boundaries.
- **Enroll/relearn voice:** changes the evidence stored for a named person's profile.
  Requires confirmed identity; an automatic guess must not train itself.

Identification can run automatically after a transcription completes, or later
through an Identify speakers action beside the meeting transcript. It produces
reviewable identity suggestions; explicit acceptance applies a name to the
transcript and enrolls eligible cached clean voice samples from that meeting.
Repeated confirmations do not duplicate evidence; removed samples stay removed.
Automatic suggestions never train profiles. That action offers the same identification selector and a target
selector when inference is needed. Changing identification alone never reruns
speech recognition.

Pending suggestions are exposed through the meeting's Confirm identities button.
The speaker editor offers a searchable saved-speaker picker for both one line
and all lines with that speaker label in the meeting. Reassignment keeps profile
names intact and removes evidence attributed to the previous person from the
corrected audio. When the latest
identification run saved comparable scores, profiles are ordered by cosine
similarity; these are group-level comparisons, not per-segment probabilities.
Unscored profiles remain available alphabetically. Older runs without rankings
can be refreshed through Identify speakers; opening the picker never runs inference.

### Using a different identification model

This is feasible, with additional audio processing. For example, a meeting
diarized with Sherpa can be identified with WeSpeaker:

1. Retain its existing anonymous speaker groups and select clean, sufficiently
   long, non-overlapping audio intervals for each group.
2. Extract WeSpeaker embeddings from those intervals using a validated adapter.
3. Match against confirmed WeSpeaker evidence in saved profiles.
4. Present identity suggestions and preserve existing human-confirmed assignments.

Reusing existing embeddings requires matching checkpoint, preprocessing,
normalization, aggregation policy, and compatible extraction version. Otherwise
extract again. Do not assume a pipeline's pooled vector is interchangeable with
a newly implemented standalone extractor. Validate standalone extraction against
the package's reference extraction before enabling cross-pipeline identification.

If clean audio is unavailable, abstain and explain why. An identity matcher cannot
repair a diarization group containing two people; the user must split it or run
Re-detect speakers. Overlap masks and audio channels must remain attached to the
evidence used for extraction.

Store each embedding in a versioned **embedding space**, not merely by dimension
or a broad name such as “WeSpeaker.” Never compare or average across incompatible
spaces. A single named person can have CAMPPlus and WeSpeaker enrollments. Reusing
the same model on another machine does not inherently create a new space, but its
numerical compatibility must be validated and runtime provenance retained.

Qualify model-specific thresholds and ambiguity margins on held-out, confirmed
identities; abstain on weak or conflicting evidence. Until that calibration is
available, conservative heuristic scores must remain suggestions, not automatic
name assignments. Anonymous diarization groups are not identity ground truth,
and similarity is not a calibrated probability. Consolidate
the existing attendee/self-voice policies into one matching service and version
the policy used for each identification run. Retain the resolved attendee/self
context, thresholds and ambiguity margin so a past decision can be explained.
Check ambiguity against every comparable profile, not only the runner-up after
attendee weighting. Keep confirmed mappings separate
from machine suggestions, and require explicit replacement of confirmed labels.
Replacement acceptance must check both the current speaker-group version and
the confirmed identity shown during review so stale proposals cannot overwrite
a subsequent correction.

Persist whether a transcript identity came from automatic processing or explicit
human confirmation. Live guesses may remain visible during recording, but must
not become protected human assignments when final processing starts. Confirming
an unchanged name still records a human decision. Preserve historical rows as
legacy/unknown rather than inventing their origin.

## Speakers page: enrollment and playable evidence

Extend the existing Speakers page rather than creating a competing identity list.
Add a profile detail view with one enrollment row per embedding space:

| Display | Meaning |
| --- | --- |
| Model and revision | CAMPPlus or WeSpeaker, with exact embedding-space details available. |
| Enrollment status | Ready, insufficient clean evidence, legacy/unknown, or model unavailable. |
| Evidence | Confirmed sample count, clean duration, and contributing recordings. |
| Updated | When this enrollment was last built and which extraction version was used. |
| Samples | Play the actual excerpts used for enrollment; open the source meeting. |
| Actions | Enroll another model, relearn, remove a sample, delete one enrollment, merge profiles. |

An embedding is a numeric representation, not a playable recording. Playback must
use the actual enrollment audio. Store the source recording identity, audio hash,
channel, exact intervals, source processing run, and confirmation provenance for
each contributing sample. A centroid may represent several excerpts: show those
excerpts rather than imply there is one audio file “inside” the embedding.

Initially reference source recordings instead of silently copying clips. If the
recording is removed, show Sample unavailable. The existing vector can still be
used for matching, but replay and regeneration are unavailable. Retaining a
separate enrollment clip after source deletion can be a later explicit option.

Removing a sample rebuilds its affected enrollment from remaining stored sample
vectors or available confirmed audio. If neither is sufficient, mark it unusable;
do not retain a centroid that still contains supposedly removed evidence. Deleting
an enrollment removes that model's recognition evidence without rewriting old
transcripts. Profile merges combine only compatible spaces and retain sample
provenance. Existing enrollments without source references must be labeled honestly;
do not invent playback links or claim they were trained on current transcript spans.

Repeated enrollment of the same audio intervals in the same embedding space must
not increase its weight, clean duration or contribution count. Profile merges
retain source provenance but count duplicate audio evidence only once; show which
retained samples do not contribute. Distinct recordings, intervals and embedding
spaces remain separate evidence. Removing one sample removes its duplicate
references within that enrollment, so a previously ignored copy cannot silently
restore the removed contribution.

Allow enrollment from confirmed speakers in older meetings whenever their audio
is available. This removes the current dependency on evidence retained only in
the running application's session. Reusing confirmed source audio can enroll the
same person in another embedding model; existing vectors cannot be converted into
equivalent vectors for another model.

## Execution architecture and remote provisioning

Define shared, versioned request/result types for transcription, speaker analysis,
embedding extraction, and identification. Adapters return data; the existing
repositories and speaker services own persistence, profile matching, enrollment,
and merge policy. Refactor the large batch integration module accordingly and
keep benchmark code outside production execution paths.

Reuse `whisper-core`, `whisper-protocol`, and `whisper-helper` where suitable;
extend their capability and timing contracts rather than adding another decoder
implementation. The current linked native backend is build-dependent: selecting
a target cannot enable CUDA or Metal in a binary that lacks that backend.

The implementation separates configuration, execution, history, identification
and enrollment under `parley-core/src/audio/processing/`. Import and reprocessing
delegate to these shared services. `speaker-core` contains the portable native
speaker algorithms used by both the application and `parley-native-worker`;
moving them does not introduce a second implementation. The former monolithic
batch module and its duplicate execution paths are removed.

Use an on-demand SSH worker for remote saved-recording jobs. All inference stages
of a job run on the selected target initially. Profile comparisons and database
updates remain local; send only the recording material needed for extraction,
not the entire profile database. Do not split stages across targets in this phase.

Target setup stores a display name, SSH host alias, and worker location. Reuse
OpenSSH credentials and host verification. Check setup probes protocol versions,
engines, model artifacts, available memory, and real device execution.

Provide an explicit Install/update worker operation with isolated, versioned
environments, pinned platform dependencies, reusable model caches, and rollback
to the previous worker. Use Python environments for faster-whisper, MLX, and
pyannote, and native helpers for whisper.cpp/Sherpa. Check FFmpeg, compilers when
needed, and GPU drivers without silently modifying global system packages.
Model-license acceptance and credentials stay outside job manifests and logs.

Jobs use private temporary directories, audio checksums, structured progress,
bounded concurrency, and a terminal result manifest. Verify returned timestamps,
speaker references, finite vectors, dimensions, and schema before importing data.
Cancellation must stop child processes and remove temporary media. Handle SSH
disconnects with explicit job status/reconnection and owned-job cleanup; retries
must not duplicate database results. Never silently fall back to a different
target, engine, or precision. Sequential model loading is the initial memory-safe
policy for constrained GPUs.

## Persistence and reproducibility

Add typed target/configuration records and immutable processing-run records.
Preserve existing migrations; introduce additive migrations and explicit legacy
configuration mapping. Migration must not download models, initiate remote work,
or discard recordings. Unknown historical provenance stays unknown.

Record each transcription or speaker-analysis run with the requested configuration
and resolved target, engine versions, model revisions/hashes, stored quantization,
actual compute types/devices, preprocessing/decoder/clustering parameters, input
audio identity, language, speaker count, timing, and outcome. Preserve previous
runs and a pointer to the current successful result. Failed or cancelled jobs
leave the existing transcript untouched; publish successful results atomically.

Identification gets its own immutable run referencing the particular diarization
run/group version, embedding space, extraction target/runtime, profile enrollment
revisions, matching policy, suggestions, and accepted decisions. This lets a
meeting say “diarized with Sherpa; identified with WeSpeaker” without falsifying
its original pipeline. Re-diarization creates new groups and invalidates stale
identity suggestions; human-confirmed decisions require explicit reconciliation.

Enrollment samples and profile versions need their own provenance. Preserve
enough historical metadata to explain past matches without retaining deleted
biometric vectors or audio solely for history. Target renaming/deletion must not
erase the display snapshot on old runs. Never include SSH credentials or private
connection paths in processing metadata. Redact private target details from
exports by default.

## Delivery stages and acceptance criteria

| Stage | Deliverable | Acceptance criterion |
| --- | --- | --- |
| 1. Shared contracts and refactor | Pipeline configuration, embedding-space identifiers, capability resolver, reuse of existing profile services | No duplicate enrollment/matching policy in the batch adapter; existing local recording remains functional. |
| 2. Local settings and processing | Four pipeline fields plus identification selector, curated downloads, removal of vocabulary behavior | Valid combinations run; invalid/missing combinations are explained; no silent substitution. |
| 3. Run history and independent identification | Meeting overrides, immutable provenance, identification-only jobs | Can change identification without ASR or regrouping; confirmed labels remain protected; cancelled jobs preserve results. |
| 4. Enrollment management | Model-specific profile details, replayable sample references, enrollment from older meetings | Playback matches actual evidence; incompatible vectors never mix; unavailable/deleted sources are handled correctly. |
| 5. Remote targets | SSH worker, Apple/NVIDIA provisioning, progress and recovery | Successful end-to-end jobs plus cancellation, disconnect, retry, and cleanup checks on each supported target. |
| 6. Quality qualification | Controlled speech, diarization, and identity comparisons | Published support matrix reflects tested combinations; experimental combinations remain labeled or disabled. |

Use a consented private test recording and a manually reviewed reference. Hold
recognition fixed when comparing speaker pipelines. Then compare speech engines
with the same checkpoint and matched decoding parameters where supported; report
unavoidable differences. Measure word errors, speaker attribution, missed/false
overlap, speaker-count errors, wall time, peak memory, and transfer overhead.
NoScribe output is a comparison, not ground truth.

Evaluate identification separately on held-out meetings and different recording
conditions, with both enrolled and unknown speakers. Do not enroll and evaluate
on the same excerpts. Check false identifications, missed matches, abstention,
and conflicting evidence for both embedding spaces. A wrong confident identity
is more harmful than leaving a speaker unnamed.

Automated checks should cover compatibility filtering, migration behavior,
embedding-space isolation, independent identification, confirmed-label protection,
atomic result replacement, evidence deletion/rebuild, source playback intervals,
remote cancellation/retry, and export redaction. Use fictional fixtures. Keep
recordings, transcripts, embeddings, credentials, private target configurations,
and raw benchmark results outside the public repository.

## Explicitly deferred

Remote live streaming, arbitrary embedding/clustering combinations, audio source
separation, automatic enrollment from unconfirmed matches, distributed stages,
general model catalogs, and vocabulary redesign are outside the initial scope.
MLX Q4 availability is verified upstream; its inclusion in Parley's supported
menu still depends on pinned-runtime and quality validation.
