"""Explicit model preparation and immutable local artifact records."""
import hashlib
import os
from pathlib import Path
from protocol import atomic_json, load_json, sha256, PublicWorkerError

# Curated immutable revisions. Inference is offline and never resolves a moving
# model name. Updating these pins is a reviewable worker release change.
REVISIONS = {
    'dropbox-dash/faster-whisper-large-v3-turbo': '0a363e9161cbc7ed1431c9597a8ceaf0c4f78fcf',
    'mukowaty/faster-whisper-int8': 'db1cdb5990df067cf53670d86241330c7ea454cf',
    'mlx-community/whisper-large-v3-turbo': 'a4aaeec0636e6fef84abdcbe3544cb2bf7e9f6fb',
    'ggerganov/whisper.cpp': '5359861c739e955e79d9a303bcbc70fb988958b1',
    'csukuangfj/sherpa-onnx-pyannote-segmentation-3-0': '9403a6902bb58e3d5ae8c7e77c3422de279db2e0',
    'csukuangfj/speaker-embedding-models': '0743f301363dec56491a490f6d6cbc9d67f9a3bf',
    'pyannote/speaker-diarization-community-1': '3533c8cf8e369892e6b79ff1bf80f7b0286a54ee',
}
SPEECH = {
    'large-v3-turbo-fp16': ('dropbox-dash/faster-whisper-large-v3-turbo', None),
    'large-v3-turbo-int8': ('mukowaty/faster-whisper-int8', 'faster-whisper-large-v3-turbo-int8'),
    'large-v3-turbo-mlx-fp16': ('mlx-community/whisper-large-v3-turbo', None),
    'large-v3-turbo': ('ggerganov/whisper.cpp', 'ggml-large-v3-turbo.bin'),
    'large-v3-turbo-q5_0': ('ggerganov/whisper.cpp', 'ggml-large-v3-turbo-q5_0.bin'),
}
NATIVE = {
    'sherpa-segmentation': ('csukuangfj/sherpa-onnx-pyannote-segmentation-3-0', 'model.onnx'),
    'campplus': ('csukuangfj/speaker-embedding-models', '3dspeaker_speech_campplus_sv_en_voxceleb_16k.onnx'),
}


def model_access_denied(error):
    """Recognize HF authorization failures only, without parsing error text."""
    from huggingface_hub.errors import GatedRepoError, HfHubHTTPError
    seen = set()
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        if isinstance(error, GatedRepoError):
            return True
        if (isinstance(error, HfHubHTTPError)
                and getattr(getattr(error, 'response', None), 'status_code', None) in (401, 403)):
            return True
        error = error.__cause__ if error.__cause__ is not None else error.__context__
    return False


def cache_root(request):
    return Path(request.get('cache_dir') or os.environ.get('PARLEY_WORKER_CACHE') or
                Path.home() / '.cache/parley/offline-models').expanduser().resolve()


def tree_digest(path):
    path = Path(path)
    if path.is_file():
        return sha256(path)
    digest = hashlib.sha256()
    for file in sorted(p for p in path.rglob('*') if p.is_file()):
        digest.update(str(file.relative_to(path)).encode())
        digest.update(bytes.fromhex(sha256(file)))
    return digest.hexdigest()


def registry(request):
    file = cache_root(request) / 'artifacts-v1.json'
    return load_json(file) if file.is_file() else {}


def resolve(request, artifact, verify=True):
    snapshot=request.get('_resolved_artifacts',{}).get(artifact)
    if snapshot:
        if not Path(snapshot['path']).exists() or (verify and tree_digest(snapshot['path'])!=snapshot['sha256']):
            raise ValueError('The snapshotted model artifact changed during this job')
        return snapshot
    explicit = request.get('model_paths', {}).get(artifact)
    if not explicit and request.get('assets_dir'):
        root = Path(request['assets_dir'])
        legacy = {'large-v3-turbo-fp16': root/'models/precise',
                  'large-v3-turbo-int8': root/'models/fast',
                  'community1': root/'pyannote'}
        candidate = legacy.get(artifact, root/artifact)
        if candidate.exists():
            explicit = str(candidate)
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if not path.exists():
            raise ValueError('The selected existing artifact is unavailable')
        digest = tree_digest(path)
        return dict(id=artifact, path=str(path), revision='content-sha256:'+digest,
                    sha256=digest, ready=True, source='existing')
    record = registry(request).get(artifact)
    if not record or not Path(record['path']).exists():
        raise ValueError('Model requires explicit Prepare/download: '+artifact)
    if verify and tree_digest(record['path']) != record['sha256']:
        raise ValueError('Model content checksum changed: '+artifact)
    return record


def prepare(request, progress):
    from huggingface_hub import snapshot_download, hf_hub_download
    root = cache_root(request)
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    records = registry(request)
    config = request['config']
    if 'artifact_ids' in request:
        ids=request['artifact_ids']
        if not isinstance(ids,list) or not ids or any(a not in (set(SPEECH)|set(NATIVE)|{'community1'}) for a in ids):
            raise ValueError('Select known curated artifacts for preparation')
    else:
        ids = [config['speech_model'], 'community1' if config['speaker_pipeline'] == 'community1' else 'sherpa-segmentation']
        if config['speaker_pipeline'] == 'sherpa' or config.get('identification') == 'campplus':
            ids.append('campplus')
        if config.get('identification') == 'wespeaker' and 'community1' not in ids:
            ids.append('community1')
    for i, artifact in enumerate(dict.fromkeys(ids)):
        progress('preparing', int(100*i/len(ids)), 'Preparing '+artifact)
        if request.get('model_paths', {}).get(artifact) or request.get('assets_dir'):
            try:
                records[artifact] = resolve(request, artifact)
                continue
            except ValueError:
                if request.get('model_paths', {}).get(artifact):
                    raise
        if artifact == 'community1':
            repo, member = 'pyannote/speaker-diarization-community-1', None
        else:
            repo, member = (SPEECH | NATIVE)[artifact]
        revision = request.get('model_revisions', {}).get(artifact) or REVISIONS[repo]
        if not revision or len(revision) != 40:
            raise ValueError('Model preparation must resolve an immutable revision')
        try:
            if member and (member.endswith('.bin') or member.endswith('.onnx')):
                path = hf_hub_download(repo_id=repo, filename=member, revision=revision, cache_dir=str(root/'hub'))
            else:
                path = snapshot_download(repo_id=repo, revision=revision, cache_dir=str(root/'hub'),
                                         allow_patterns=(member+'/*') if member else None)
                if member:
                    path = str(Path(path)/member)
        except Exception as error:
            if model_access_denied(error):
                raise PublicWorkerError('model_access_denied', artifact) from error
            raise
        records[artifact] = dict(id=artifact, path=path, repo=repo, revision=revision,
                                 sha256=tree_digest(path), ready=True)
    atomic_json(root/'artifacts-v1.json', records)
    return dict(version=1, artifacts=[{k:v for k,v in r.items() if k != 'path'} for r in records.values()])


def public_record(record):
    return {k:v for k,v in record.items() if k not in ('path', 'source')}
