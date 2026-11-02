# Portable native speaker worker

Build on the destination operating system and architecture:

```sh
cargo build --locked --release -p parley-native-worker
```

The binary is `target/release/parley_native_worker`. It depends only on the
portable `speaker-core` algorithms and Sherpa's native runtime; it does not
compile the application, PipeWire, a GUI, SQLite, or whisper.cpp. The build needs
a Rust toolchain, a native C/C++ linker, and network access for the locked Sherpa
runtime distribution when it is not cached. Package the matching Sherpa C API
and ONNX Runtime shared libraries beside the executable. The installer accepts
flat regular files: dereference library symlinks into files named for the
required loader/SONAME names rather than placing symlinks in the bundle. The executable searches its own directory for these libraries on
Linux and macOS. Build the existing `whisper-helper` separately for CPU, CUDA,
Vulkan, or Metal speech recognition.

The CLI accepts `--request <private.json> --response <private.json>`. Version 1
requests support `speaker_analysis` and `extract_embeddings`, with explicit
native model paths and normalized mono 16 kHz PCM16 WAV input. CPU is the
qualified speaker device. Results retain exact extraction intervals and a
checkpoint-hashed embedding space. The same implementations and extraction
policy power the application's local native pipeline.

Do not put recordings, embeddings, credentials, private model files, or real
request/result manifests into this repository. The Python supervisor owns
private job directories and cancellation of this child process.
