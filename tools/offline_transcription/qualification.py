"""Version/artifact-specific Community-1 MPS parity qualification."""
import datetime
import importlib.metadata
import os
import platform
from assets import cache_root,resolve
from protocol import atomic_json,load_json,sha256

ADAPTER_VERSION='community1-4.0.4-clean-three10s-v1'
PACKAGES={'pyannote.audio':'4.0.4','torch':'2.8.0','torchaudio':'2.8.0'}


def installed_packages():
    try: return {name:importlib.metadata.version(name).split('+')[0] for name in PACKAGES}
    except importlib.metadata.PackageNotFoundError: return {}


def platform_identity():
    return dict(system=platform.system(),architecture=platform.machine(),release=platform.release())


def compatible_platform(record):
    # Keep the tested OS release as provenance, not as an acceleration switch.
    # The model/package parity approval survives OS updates on the same platform;
    # the worker's execution probes separately verify MPS on the current OS.
    return (isinstance(record, dict)
            and record.get('system') == platform.system()
            and record.get('architecture') == platform.machine())


def mps_qualified(request):
    if (platform.system()!='Darwin' or platform.machine()!='arm64'
            or os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK')=='1' or installed_packages()!=PACKAGES):
        return False
    try:
        artifact=resolve(request,'community1')
        file=cache_root(request)/'mps-qualifications-v1.json'
        records=load_json(file) if file.is_file() else {}
        record=records.get(artifact['sha256'],{})
        return (record.get('version')==1 and record.get('adapter_version')==ADAPTER_VERSION
            and record.get('packages')==PACKAGES and compatible_platform(record.get('platform'))
            and record.get('artifact',{}).get('sha256')==artifact['sha256']
            and record.get('artifact',{}).get('revision')==artifact['revision'])
    except (ValueError,OSError,KeyError): return False


def accept_mps(request,comparison_file,analysis_file):
    """Explicitly accept an inspected private parity report; never run inference."""
    report=load_json(comparison_file);analysis=load_json(analysis_file)
    if (platform.system()!='Darwin' or platform.machine()!='arm64'
            or installed_packages()!=PACKAGES or os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK')=='1'):
        raise ValueError('Qualification requires the exact pinned Apple runtime with CPU fallback disabled')
    timeline=report.get('timeline',{})
    vectors=report.get('fixed_clean_interval_embeddings',[])
    if (report.get('reference_device')!='cpu' or report.get('candidate_device')!='mps'
            or analysis.get('diarization_device')!='mps'
            or timeline.get('reference_speakers')!=timeline.get('candidate_speakers')
            or timeline.get('reference_speaker_seconds',0)<30
            or not 0<=timeline.get('speaker_time_error_fraction',1)<=.001
            or not .999<=timeline.get('overlap_intersection_over_union',0)<=1.000001
            or not vectors or any(v.get('compatible') is not True
                or not .9999<=v.get('cosine_similarity',0)<=1.000001
                or not 0<=v.get('max_absolute_difference',1)<=.0001 for v in vectors)):
        raise ValueError('The CPU/MPS comparison does not meet this adapter qualification contract')
    for runtime in (analysis.get('provenance',{}).get('runtime',{}),report.get('fixed_reference_runtime',{})):
        actual={name:str(runtime.get('packages',{}).get(name,'')).split('+')[0] for name in PACKAGES}
        if actual!=PACKAGES: raise ValueError('The parity report used a different package runtime')
    if analysis.get('provenance',{}).get('embedding_reference_verified') is not True:
        raise ValueError('The candidate did not verify package-reference embedding extraction')
    artifact=resolve(request,'community1')
    if analysis.get('provenance',{}).get('speaker_artifact',{}).get('sha256')!=artifact['sha256']:
        raise ValueError('The accepted comparison used a different Community-1 artifact')
    file=cache_root(request)/'mps-qualifications-v1.json'
    records=load_json(file) if file.is_file() else {}
    record=dict(version=1,adapter_version=ADAPTER_VERSION,packages=PACKAGES,platform=platform_identity(),
        artifact={k:artifact[k] for k in ('id','revision','sha256')},
        accepted_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),report_sha256=sha256(comparison_file),
        scope='CPU parity and device compatibility; not ground-truth speaker accuracy')
    records[artifact['sha256']]=record
    file.parent.mkdir(parents=True,exist_ok=True,mode=0o700);atomic_json(file,records)
    return dict(version=1,accepted=True,adapter_version=ADAPTER_VERSION,artifact_sha256=artifact['sha256'])
