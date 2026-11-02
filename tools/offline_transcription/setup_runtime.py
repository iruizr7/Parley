#!/usr/bin/env python3
"""Explicit, isolated worker install/update. Never changes system packages.

The bootstrap uses only the existing Python standard library. Each release has
its own managed Python environment and installed-package lock. Activation is an
atomic symlink swap only after runtime/device checks pass; rollback is explicit.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import signal
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.request
import zipfile
from protocol import atomic_json, sha256,load_json,bundle_sha256,bundle_description,BUNDLE_FILES,private_root

UV_VERSION='0.9.5'
PYTHON_VERSION='3.12.12'
RUNTIME_VERSION='1'
PACKAGES=['faster-whisper==1.2.1','ctranslate2==4.7.1','pyannote.audio==4.0.4',
          'torchcodec==0.7.0','numpy==2.3.4','huggingface-hub==0.36.0','imageio-ffmpeg==0.6.0']
APPLE_PACKAGES=['mlx-whisper==0.4.3','mlx==0.29.3']


def event(stage,percent,message):
    print(json.dumps(dict(stage=stage,percent=percent,message=message)),flush=True)


def uv_executable(root):
    # Do not depend on a globally installed uv (or alter it). PyPI's release
    # metadata supplies the immutable wheel filename and published SHA256.
    destination=root/'bootstrap'/('uv-'+UV_VERSION)
    if destination.is_file(): return str(destination)
    event('installing',2,'Preparing isolated package manager')
    system=platform.system();machine=platform.machine().lower()
    if system not in ('Linux','Darwin') or machine not in ('x86_64','amd64','arm64','aarch64'):
        raise ValueError('This installer supports Linux and Apple Silicon targets')
    architecture='aarch64' if machine in ('arm64','aarch64') else 'x86_64'
    with urllib.request.urlopen('https://pypi.org/pypi/uv/'+UV_VERSION+'/json') as stream:
        metadata=json.load(stream)
    def matching(record):
        name=record['filename']
        if system=='Darwin': return name.endswith('.whl') and 'macosx' in name and (('arm64' in name) if architecture=='aarch64' else ('x86_64' in name))
        return name.endswith('.whl') and 'manylinux' in name and name.endswith(architecture+'.whl')
    candidates=[record for record in metadata['urls'] if matching(record)]
    if len(candidates)!=1: raise ValueError('No unique pinned package-manager wheel for this platform')
    selected=candidates[0]
    destination.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
    with tempfile.TemporaryDirectory(dir=destination.parent) as folder:
        wheel=Path(folder)/'uv.whl'
        urllib.request.urlretrieve(selected['url'],wheel)
        if sha256(wheel)!=selected['digests']['sha256']: raise ValueError('Package-manager wheel checksum mismatch')
        with zipfile.ZipFile(wheel) as archive:
            members=[n for n in archive.namelist() if n.endswith('/uv') and '.data/scripts/' in n]
            if not members: members=[n for n in archive.namelist() if n.endswith('/uv')]
            if len(members)!=1: raise ValueError('Unexpected package-manager wheel layout')
            temporary=destination.with_suffix('.partial');temporary.write_bytes(archive.read(members[0]));temporary.chmod(0o700)
            temporary.replace(destination)
        atomic_json(destination.with_suffix('.json'),dict(version=UV_VERSION,wheel=selected['filename'],sha256=selected['digests']['sha256']))
    return str(destination)


def run(command,environment):
    # Installer subprocess logs remain private in the worker root. Stdout is
    # reserved for structured progress; credentials never enter commands.
    subprocess.run(command,check=True,env=environment,stdout=sys.stderr,stderr=sys.stderr)


def activate(root,release):
    current=root/'current';previous=root/'previous'
    if current.is_symlink():
        old=current.resolve()
        temp=root/'previous.partial';temp.unlink(missing_ok=True);temp.symlink_to(old);temp.replace(previous)
    temp=root/'current.partial';temp.unlink(missing_ok=True);temp.symlink_to(release);temp.replace(current)


def rollback(root):
    root=private_root(root)
    previous=root/'previous'
    if not previous.is_symlink() or not previous.resolve().is_dir(): raise ValueError('No previous installed runtime is available')
    destination=previous.resolve()
    manifest=load_json(destination/'runtime.json') if (destination/'runtime.json').is_file() else {}
    if not manifest.get('worker_bundle_sha256'):
        raise ValueError('Previous legacy runtime has no worker-code snapshot; full rollback cannot be proven')
    if bundle_sha256(destination/'worker',manifest)!=manifest['worker_bundle_sha256']:
        raise ValueError('The previous worker-code snapshot changed; rollback was not activated')
    for name,digest in manifest.get('native_files',{}).items():
        if sha256(destination/'native'/name)!=digest:
            raise ValueError('The previous native helper snapshot changed; rollback was not activated')
    activate(root,destination)
    return dict(version=1,state='rolled_back',runtime_id=destination.name,worker_snapshot_verified=True)


def snapshot_worker(destination):
    folder=destination/'worker';folder.mkdir(mode=0o700)
    for name in BUNDLE_FILES:
        source=Path(__file__).with_name(name)
        if not source.is_file(): raise ValueError('Install/update requires the complete worker bundle')
        target=folder/name;target.write_bytes(source.read_bytes());target.chmod(0o600)
    return bundle_sha256(folder)


def install_native(bundle,digest,destination):
    if not digest or sha256(bundle)!=digest: raise ValueError('Native bundle requires its exact SHA256')
    native=destination/'native';native.mkdir(mode=0o700)
    with tarfile.open(bundle) as archive:
        for member in archive.getmembers():
            if not member.isfile() or Path(member.name).name!=member.name or member.size>1024**3:
                raise ValueError('Native bundles contain only flat regular helper/library files')
            with archive.extractfile(member) as source,(native/member.name).open('wb') as target:
                shutil.copyfileobj(source,target)
            (native/member.name).chmod(0o700 if member.mode&0o111 else 0o600)


def verify_native(destination,environment):
    """Verify architecture, dynamic libraries and protocol before activation."""
    native=destination/'native'
    if not native.is_dir(): return {}
    environment=dict(environment)
    variable='DYLD_LIBRARY_PATH' if platform.system()=='Darwin' else 'LD_LIBRARY_PATH'
    libraries=[str(native)]+[str(p) for p in destination.glob('lib/python*/site-packages/nvidia/*/lib')]
    environment[variable]=os.pathsep.join(libraries+[environment.get(variable,'')])
    verified={}
    speaker=native/'parley_native_worker'
    if speaker.is_file():
        probe=subprocess.run([str(speaker),'--probe'],capture_output=True,text=True,env=environment,timeout=30,check=True)
        value=json.loads(probe.stdout)
        if value.get('protocol_version')!=1 or value.get('kind')!='native_speaker' or value.get('runtime_ready') is not True:
            raise ValueError('Native speaker helper protocol is incompatible')
        verified[speaker.name]=dict(protocol_version=1,runnable=True,runtime=value.get('runtime'),model_verified=False)
    for backend in ('cpu','cuda','vulkan','metal'):
        helper=native/('whisper-helper-'+backend)
        if not helper.is_file(): continue
        messages='\n'.join(json.dumps(v) for v in [dict(type='ping'),dict(type='probe',model_path=None),dict(type='shutdown')])+'\n'
        probe=subprocess.run([str(helper)],input=messages,capture_output=True,text=True,env=environment,timeout=30,check=True)
        values=[json.loads(line) for line in probe.stdout.splitlines() if line.strip()]
        if (len(values)!=3 or values[0].get('type')!='pong' or values[1].get('type')!='probe_result'
                or values[1].get('backend')!=backend or values[2].get('type')!='goodbye'):
            raise ValueError('Native speech helper protocol/backend is incompatible')
        verified[helper.name]=dict(protocol_version=1,runnable=True,backend=backend,model_verified=False)
    if not verified:
        raise ValueError('The native bundle contains no supported executable helper')
    return verified


def cuda_driver_available():
    try:
        import ctypes
        driver=ctypes.CDLL('libcuda.so.1')
        count=ctypes.c_int()
        return driver.cuInit(0)==0 and driver.cuDeviceGetCount(ctypes.byref(count))==0 and count.value>0
    except (OSError,AttributeError): return False


def resolve_install_device(policy):
    if policy!='auto': return policy
    if platform.system()=='Darwin' and platform.machine()=='arm64': return 'metal'
    if platform.system()=='Linux' and cuda_driver_available(): return 'cuda'
    return 'cpu'


def install(root,device,native_bundle=None,native_sha256=None):
    requested_device=device;device=resolve_install_device(device)
    if requested_device=='auto': event('installing',0,'Auto selected explicit '+device+' platform runtime')
    root=private_root(root)
    apple=platform.system()=='Darwin' and platform.machine()=='arm64'
    if device=='metal' and not apple: raise ValueError('Apple GPU installation requires Apple Silicon')
    if device=='cuda' and platform.system()!='Linux': raise ValueError('The qualified CUDA worker requires Linux')
    releases=root/'runtimes';releases.mkdir(exist_ok=True,mode=0o700)
    # Model caches are shared across releases and survive rollback.
    (root/'cache').mkdir(exist_ok=True,mode=0o700)
    uv=uv_executable(root)
    environment=dict(os.environ,UV_CACHE_DIR=str(root/'package-cache'),UV_PYTHON_INSTALL_DIR=str(root/'python'),
        UV_PYTHON_PREFERENCE='only-managed',UV_NO_PROGRESS='1',HF_HOME=str(root/'cache/huggingface'))
    identity='v'+RUNTIME_VERSION+'-'+device+'-'+str(time.time_ns())
    staging=releases/identity
    try:
        event('installing',10,'Creating managed Python runtime')
        run([uv,'venv','--python',PYTHON_VERSION,str(staging)],environment)
        python=staging/'bin/python'
        torch_suffix='+cu128' if device=='cuda' else '' if apple else '+cpu'
        if apple:
            torch_args=['torch==2.8.0','torchaudio==2.8.0']
        else:
            torch_args=['torch==2.8.0'+torch_suffix,'torchaudio==2.8.0'+torch_suffix,
                '--index-url','https://download.pytorch.org/whl/'+('cu128' if device=='cuda' else 'cpu')]
        event('installing',25,'Installing pinned platform inference libraries')
        run([uv,'pip','install','--python',str(python),*torch_args],environment)
        event('installing',50,'Installing pinned speech and speaker packages')
        run([uv,'pip','install','--python',str(python),*PACKAGES,*(['torch==2.8.0','torchaudio==2.8.0'] if apple else ['torch==2.8.0'+torch_suffix,'torchaudio==2.8.0'+torch_suffix]),*(APPLE_PACKAGES if apple else [])],environment)
        if native_bundle: install_native(native_bundle,native_sha256,staging)
        elif (root/'current/native').is_dir(): shutil.copytree(root/'current/native',staging/'native')
        # Retain the exact dependency resolution as a release lock/provenance.
        frozen=subprocess.run([uv,'pip','freeze','--python',str(python)],env=environment,check=True,capture_output=True,text=True).stdout
        (staging/'installed-requirements.txt').write_text(frozen,encoding='utf-8')
        event('installing',80,'Verifying runtime, FFmpeg and the selected device')
        check="""import json,torch,ctranslate2,imageio_ffmpeg,subprocess
+from pyannote.audio import Pipeline
+from faster_whisper import WhisperModel
+subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-version'],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
+device=__import__('sys').argv[1]
+if device=='cuda':
+ assert torch.cuda.is_available() and ctranslate2.get_cuda_device_count()>0, 'CUDA runtime unavailable'
+ torch.ones((2,2),device='cuda').matmul(torch.ones((2,2),device='cuda')).cpu()
+if device=='metal':
+ import mlx.core as mx
+ import mlx_whisper
+ assert mx.metal.is_available(), 'Apple GPU runtime unavailable'
+ mx.eval(mx.ones((2,2)) @ mx.ones((2,2)))
+print(json.dumps({'device':device,'runtime_verified':True}))
+""".replace('\n+','\n')
        run([str(python),'-c',check,device],environment)
        native_verification=verify_native(staging,environment)
        worker_hash=snapshot_worker(staging)
        run([str(python),'-c','import sys;sys.path.insert(0,sys.argv[1]);import worker,supervisor',str(staging/'worker')],environment)
        manifest=dict(version=1,runtime_id=identity,platform=dict(system=platform.system(),architecture=platform.machine()),
            device=device,requested_device=requested_device,python_version=PYTHON_VERSION,uv_version=UV_VERSION,requirements_sha256=sha256(staging/'installed-requirements.txt'),
            package_pins=PACKAGES+(APPLE_PACKAGES if apple else []),runtime_verified=True,
            native_helpers_ready=bool(native_verification),native_helpers=native_verification,models_prepared=False)
        manifest.update(bundle_description(staging/'worker'))
        if manifest['worker_bundle_sha256']!=worker_hash: raise ValueError('Worker snapshot changed during installation')
        manifest['native_files']={p.name:sha256(p) for p in (staging/'native').glob('*') if p.is_file()}
        atomic_json(staging/'runtime.json',manifest)
        event('installing',95,'Activating verified runtime; retaining the previous release')
        activate(root,staging)
        event('installing',100,'Runtime installed; prepare models and check pipeline setup')
        return manifest
    except BaseException:
        shutil.rmtree(staging,ignore_errors=True)
        raise


def main():
    os.umask(0o077)
    def interrupted(signum,frame):
        raise KeyboardInterrupt('Installation cancelled')
    signal.signal(signal.SIGTERM,interrupted)
    parser=argparse.ArgumentParser()
    parser.add_argument('--directory',type=Path,required=True)
    parser.add_argument('--device',choices=('auto','cpu','cuda','metal','vulkan'),default='cpu')
    parser.add_argument('--rollback',action='store_true')
    parser.add_argument('--native-bundle',type=Path)
    parser.add_argument('--native-sha256')
    parser.add_argument('--response',type=Path)
    args=parser.parse_args()
    result=rollback(args.directory) if args.rollback else install(args.directory,args.device,args.native_bundle,args.native_sha256)
    if args.response: atomic_json(args.response,result)
    else: print(json.dumps(result,allow_nan=False))


if __name__=='__main__': main()
