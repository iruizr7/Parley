"""Versioned offline protocol and validation. No ML imports or network access."""
import hashlib
import json
import math
import os
import stat
from pathlib import Path
import re

VERSION = 1
MAX_JSON_BYTES = 64 * 1024 * 1024
OPERATIONS = {'process', 'speech_recognition', 'speaker_analysis', 'extract_embeddings', 'probe', 'prepare'}
ALIASES = {'transcribe': 'process', 'diarize': 'speaker_analysis', 'embed': 'extract_embeddings'}
LEGACY_BUNDLE_FILES=('worker.py','protocol.py','assets.py','qualification.py','supervisor.py','setup_runtime.py','requirements.txt')
BUNDLE_FILES=('worker.py','protocol.py','assets.py','qualification.py','resources.py','supervisor.py','setup_runtime.py','requirements.txt')
BUNDLE_VERSION=2
BUNDLE_VERSIONS={1:LEGACY_BUNDLE_FILES,2:BUNDLE_FILES}
MODEL_ARTIFACT_IDS = frozenset({
    'large-v3-turbo-fp16', 'large-v3-turbo-int8', 'large-v3-turbo-mlx-fp16',
    'large-v3-turbo', 'large-v3-turbo-q5_0', 'sherpa-segmentation', 'campplus', 'community1',
})


def public_failure(value):
    """Accept only fixed public error identifiers, never worker exception text."""
    if (not isinstance(value, dict)
            or set(value) != {'protocol_version', 'error_code', 'error_artifact'}
            or type(value['protocol_version']) is not int or value['protocol_version'] != VERSION
            or value['error_code'] != 'model_access_denied'
            or not isinstance(value['error_artifact'], str)
            or value['error_artifact'] not in MODEL_ARTIFACT_IDS):
        return {}
    return {key: value[key] for key in ('error_code', 'error_artifact')}


class PublicWorkerError(Exception):
    """A worker failure with a validated, non-sensitive public explanation."""
    def __init__(self, code, artifact):
        self.public_error = public_failure(dict(protocol_version=VERSION,
            error_code=code, error_artifact=artifact))
        if not self.public_error:
            raise ValueError('Unknown public worker error')
        super().__init__('Model preparation requires access authorization')


def failure_path(response):
    response = Path(response)
    return response.with_name(response.name + '.failure.json')


def read_public_failure(response):
    """The sidecar is untrusted; malformed/oversized failures stay generic."""
    path = failure_path(response)
    try:
        if path.is_symlink() or not path.is_file() or path.stat().st_size > 4096:
            return {}
        return public_failure(load_json(path))
    except (OSError, ValueError, TypeError):
        return {}


def load_json(path):
    path = Path(path)
    if path.stat().st_size > MAX_JSON_BYTES:
        raise ValueError('Protocol document exceeds size limit')
    return json.loads(path.read_text(encoding='utf-8'),
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError('Non-finite JSON number')))


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.partial')
    with temporary.open('w', encoding='utf-8') as stream:
        import os
        os.fchmod(stream.fileno(), 0o600)
        json.dump(value, stream, ensure_ascii=False, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()



ROOT_MARKER='.parley-worker-root.json'


def private_root(path):
    """Create a dedicated private leaf, or verify an already owned worker root.

    Existing directories are never chmodded. Unmarked legacy roots are adopted
    only with a verified immutable installed release, not from their name alone.
    """
    root=Path(os.path.abspath(os.path.expanduser(os.fspath(path))))
    shared={'/','/tmp','/var/tmp','/private/tmp','/private/var/tmp','/private',
            '/var','/private/var','/usr','/usr/local','/opt','/home','/Users','/root',
            '/etc','/run','/var/run','/mnt','/media','/srv','/dev','/proc','/sys',
            '/bin','/sbin','/lib','/lib64','/System','/Library','/Applications'}
    if str(root) in shared or root.resolve()==Path.home().resolve():
        raise ValueError('Choose a dedicated worker subdirectory, not a home or shared system directory')
    # Directory advisory locking creates no lock file and never changes the
    # parent's mode/content. Serialize first publication so another bootstrap
    # cannot observe our freshly created leaf before its ownership marker.
    missing=[];parent=root.parent
    while not parent.exists():
        missing.append(parent);parent=parent.parent
    for directory in reversed(missing):
        try: directory.mkdir(mode=0o700)
        except FileExistsError: pass
    descriptor=os.open(root.parent,os.O_RDONLY|getattr(os,'O_DIRECTORY',0))
    try:
        import fcntl
        fcntl.flock(descriptor,fcntl.LOCK_EX)
        return _private_root_locked(root)
    finally: os.close(descriptor)


def _private_root_locked(root):
    created=False
    try: root.lstat()
    except FileNotFoundError:
        try: root.mkdir(mode=0o700);created=True
        except FileExistsError: pass
    status=root.lstat()
    if not stat.S_ISDIR(status.st_mode) or status.st_uid!=os.getuid() or status.st_mode&0o077:
        raise ValueError('Worker directory must be a private directory owned by the current user; its permissions were not changed')
    marker=root/ROOT_MARKER
    # Filesystem device numbers can change across boots/remounts. Persist the
    # canonical location, not the kernel's temporary device/inode identity.
    identity=dict(version=2,kind='parley-offline-worker',uid=os.getuid(),
                  directory_path=str(root.resolve()))
    if marker.exists() or marker.is_symlink():
        if marker.is_symlink() or not marker.is_file():
            raise ValueError('Worker directory ownership marker does not match')
        saved=load_json(marker)
        if saved!=identity:
            # Version 1 had no path. Its unchanged inode and owner let us
            # recognize the original folder even when st_dev changed. A marker
            # copied to an unrelated directory must not be adopted.
            legacy=(isinstance(saved,dict)
                    and set(saved)=={'version','kind','uid','directory_device','directory_inode'}
                    and type(saved['version']) is int and saved['version']==1
                    and saved['kind']==identity['kind']
                    and type(saved['uid']) is int and saved['uid']==identity['uid']
                    and type(saved['directory_inode']) is int and saved['directory_inode']==status.st_ino
                    and type(saved['directory_device']) is int and saved['directory_device']>=0)
            if not legacy:
                raise ValueError('Worker directory ownership marker does not match')
            # The parent directory lock above serializes migrations. Use a
            # unique private temporary file so an interrupted write leaves the
            # previous marker intact and cannot follow a pre-existing link.
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w',encoding='utf-8',dir=root,
                                             prefix=ROOT_MARKER+'.',delete=False) as stream:
                temporary=Path(stream.name)
                try:
                    json.dump(identity,stream);stream.flush();os.fsync(stream.fileno())
                    temporary.replace(marker)
                finally: temporary.unlink(missing_ok=True)
    else:
        if not created:
            # A legacy worker must have its actual current release contained in
            # runtimes, with complete code/native checksum proof. Merely having
            # a current link or a similarly named directory is insufficient.
            releases=root/'runtimes';current=root/'current'
            if releases.is_symlink() or not releases.is_dir() or not current.is_symlink():
                raise ValueError('Existing directory is not a verified worker root; choose a new dedicated subdirectory')
            release=current.resolve()
            if release.parent!=releases.resolve() or not release.is_dir():
                raise ValueError('Legacy worker release must belong to this worker directory')
            manifest=load_json(release/'runtime.json')
            if (manifest.get('runtime_id')!=release.name or manifest.get('runtime_verified') is not True
                    or not manifest.get('worker_bundle_sha256')
                    or bundle_sha256(release/'worker',manifest)!=manifest['worker_bundle_sha256']):
                raise ValueError('Existing worker release has no verified complete code snapshot')
            for name,digest in manifest.get('native_files',{}).items():
                native=release/'native'/name
                if Path(name).name!=name or native.is_symlink() or not native.is_file() or sha256(native)!=digest:
                    raise ValueError('Existing worker native snapshot failed verification')
        # Exclusive creation avoids overwriting any unrelated file or link.
        with marker.open('x',encoding='utf-8') as stream:
            os.fchmod(stream.fileno(),0o600);json.dump(identity,stream)
    return root


def bundle_sha256(folder,manifest=None):
    files=BUNDLE_FILES
    declared=None
    if manifest is not None:
        version=manifest.get('worker_bundle_version')
        declared=manifest.get('worker_bundle_files')
        if version is None and declared is None:
            # The first complete immutable releases recorded this exact known
            # seven-file digest. Never infer a different set from directory contents.
            files=LEGACY_BUNDLE_FILES
        else:
            if isinstance(version,bool) or version not in BUNDLE_VERSIONS or not isinstance(declared,dict):
                raise ValueError('Unsupported or incomplete worker bundle manifest')
            files=BUNDLE_VERSIONS[version]
            if set(declared)!=set(files) or any(not isinstance(v,str) or not re.fullmatch('[0-9a-f]{64}',v) for v in declared.values()):
                raise ValueError('Invalid worker bundle filenames or checksums')
        if {p.name for p in Path(folder).iterdir() if p.is_file()}!=set(files):
            raise ValueError('Unexpected or missing files in immutable worker snapshot')
    digest=hashlib.sha256()
    for name in files:
        file=Path(folder)/name
        if not file.is_file(): raise ValueError('The versioned worker bundle is incomplete')
        actual=sha256(file)
        if declared is not None and actual!=declared[name]:
            raise ValueError('Worker snapshot file checksum changed')
        digest.update(name.encode());digest.update(bytes.fromhex(actual))
    return digest.hexdigest()


def bundle_description(folder,version=BUNDLE_VERSION):
    if isinstance(version,bool) or version not in BUNDLE_VERSIONS: raise ValueError('Unsupported worker bundle version')
    description=dict(worker_bundle_version=version,worker_bundle_files={name:sha256(Path(folder)/name) for name in BUNDLE_VERSIONS[version]})
    description['worker_bundle_sha256']=bundle_sha256(folder,description)
    return description


def job_id(value):
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9_-]{8,80}', value):
        raise ValueError('Invalid job identifier')
    return value


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def interval(value, duration):
    return (isinstance(value, dict) and finite(value.get('start')) and finite(value.get('end'))
            and 0 <= value['start'] < value['end'] <= duration + .1)


def normalize_request(request):
    """Keep the original optional-worker envelope readable during migration."""
    request = dict(request)
    if request.get('version', 1) != VERSION:
        raise ValueError('Unsupported worker protocol version')
    request['version'] = VERSION
    op = request.get('operation', 'process')
    request['operation'] = ALIASES.get(op, op)
    if request['operation'] not in OPERATIONS:
        raise ValueError('Unsupported worker operation')
    config = dict(request.get('config') or {})
    config.setdefault('speech_engine', {'native': 'whisper_cpp', 'faster': 'faster_whisper', 'mlx': 'mlx_whisper'}.get(request.get('recognizer', 'native')))
    config.setdefault('speaker_pipeline', {'native': 'sherpa', 'community': 'community1'}.get(request.get('diarization', 'native')))
    config.setdefault('speech_model', 'large-v3-turbo-int8' if request.get('model') == 'fast' else 'large-v3-turbo-fp16')
    config.setdefault('identification', 'same')
    if config['speech_engine'] not in ('whisper_cpp', 'faster_whisper', 'mlx_whisper'):
        raise ValueError('Unknown speech engine')
    if config['speaker_pipeline'] not in ('sherpa', 'community1'):
        raise ValueError('Unknown speaker pipeline')
    if config['identification'] not in ('same', 'off', 'campplus', 'wespeaker'):
        raise ValueError('Unknown identification model')
    if 'q4' in config['speech_model']:
        raise ValueError('MLX Q4 remains disabled pending qualification')
    request['config'] = config
    # Auto was an old optional-worker preference. Resolve only for those old
    # requests; new pipeline requests must snapshot an explicit device.
    device = request.get('device', 'cpu')
    request.setdefault('speech_device', device)
    request.setdefault('speaker_device', 'cpu' if device in ('metal', 'vulkan') else device)
    if (request['operation'] not in ('probe','prepare')
            and ('auto' in (request['speech_device'],request['speaker_device']))):
        raise ValueError('Choose an explicit target device before submitting a job')
    count = request.get('num_speakers', 0) or 0
    if not isinstance(count, int) or isinstance(count, bool) or not 0 <= count <= 64:
        raise ValueError('Speaker count must be auto or 1–64')
    request['num_speakers'] = count
    if request.get('request_id'):
        job_id(request['request_id'])
    digest = request.get('audio_sha256')
    if digest is not None and not re.fullmatch('[0-9a-f]{64}', digest):
        raise ValueError('Invalid audio checksum')
    return request


def validate_result(result, request=None):
    """Reject malformed model output before it reaches persistence."""
    if result.get('version') != VERSION or not finite(result.get('duration')) or result['duration'] <= 0:
        raise ValueError('Invalid result version/duration')
    duration = result['duration']
    for key in ('segments', 'words', 'turns', 'overlaps', 'embeddings'):
        if not isinstance(result.get(key), list) or len(result[key]) > 2_000_000:
            raise ValueError('Missing or oversized result array')
    speakers = {t.get('speaker') for t in result['turns']}
    if any(not isinstance(s, str) or not s or len(s) > 256 for s in speakers):
        raise ValueError('Invalid speaker identifier')
    operation=(request or {}).get('operation')
    operation=ALIASES.get(operation,operation)
    requested_intervals = (request or {}).get('source_intervals', [])
    if operation == 'extract_embeddings':
        speakers |= {i.get('speaker') for i in requested_intervals}
    for key in ('segments', 'words', 'turns', 'overlaps'):
        previous = -1.
        for item in result[key]:
            if not interval(item, duration):
                raise ValueError('Invalid result timestamp')
            if key in ('segments', 'words'):
                if item['start'] < previous or not isinstance(item.get('text'), str):
                    raise ValueError('Invalid text/time ordering')
                previous = item['start']
            if item.get('speaker') is not None and item['speaker'] not in speakers:
                raise ValueError('Unknown result speaker reference')
            refs = item.get('speakers', item.get('overlapping_speakers', []))
            if not isinstance(refs, list) or any(s not in speakers for s in refs):
                raise ValueError('Unknown overlap speaker reference')
            if key == 'overlaps' and len(set(refs)) < 2:
                raise ValueError('Overlap requires two different speakers')
            p = item.get('probability')
            if p is not None and (not finite(p) or not 0 <= p <= 1):
                raise ValueError('Invalid word probability')
    dimensions = {}
    for embedding in result['embeddings']:
        vector = embedding.get('vector')
        space = embedding.get('model_id')
        if (embedding.get('speaker') not in speakers or not isinstance(space, str) or not space
                or not isinstance(vector, list) or not 1 <= len(vector) <= 4096
                or any(not finite(x) for x in vector)
                or not .9 <= sum(x*x for x in vector) <= 1.1
                or not finite(embedding.get('clean_seconds')) or embedding['clean_seconds'] < 3
                or embedding['clean_seconds'] > duration + .1
                or not isinstance(embedding.get('sample_count', 1), int)
                or embedding.get('sample_count', 1) < 1):
            raise ValueError('Invalid speaker embedding')
        metadata = embedding.get('embedding_space')
        if metadata is not None:
            if (not isinstance(metadata, dict) or not isinstance(metadata.get('id'),str) or not metadata['id']
                    or metadata.get('dimension') != len(vector)
                    or any(not isinstance(metadata.get(k), str) or not metadata[k]
                           for k in ('model','revision','preprocessing','normalization','aggregation','extraction_version'))):
                raise ValueError('Incomplete or incompatible embedding-space identity')
        identity=metadata['id'] if metadata else space
        if identity in dimensions and dimensions[identity] != len(vector):
            raise ValueError('Inconsistent embedding dimension')
        dimensions[identity] = len(vector)
        for span in embedding.get('source_intervals', []):
            if not interval(span, duration) or span.get('channel', 0) != 0:
                raise ValueError('Invalid embedding source interval')
        if operation == 'extract_embeddings':
            spans = embedding.get('source_intervals', [])
            if not spans or any(not any(span.get('speaker')==allowed.get('speaker')
                    and span.get('channel',0)==allowed.get('channel',0)
                    and allowed['start']<=span['start']<span['end']<=allowed['end']
                    for allowed in requested_intervals) for span in spans):
                raise ValueError('Extraction must retain the exact used intervals within requested clean sources')
    if request:
        for key in ('request_id', 'audio_sha256'):
            if request.get(key) and result.get(key) != request[key]:
                raise ValueError('Result does not belong to the requested audio/job')
    return result


def clean_intervals(turns, simultaneous, minimum=3.):
    """Union each speaker's turns and subtract all known simultaneous speech."""
    result = []
    for speaker in sorted({t['speaker'] for t in turns}):
        spans = []
        for start, end in sorted((t['start'], t['end']) for t in turns if t['speaker'] == speaker):
            if spans and start <= spans[-1][1]:
                spans[-1][1] = max(end, spans[-1][1])
            else:
                spans.append([start, end])
        for overlap in simultaneous:
            spans = [(s, min(e, overlap['start'])) for s, e in spans if s < overlap['start']] + [
                (max(s, overlap['end']), e) for s, e in spans if e > overlap['end']]
        result += [dict(speaker=speaker, start=s, end=e, channel=0) for s, e in spans if e-s >= minimum]
    return sorted(result, key=lambda span: (span['start'], span['speaker']))
