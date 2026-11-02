#!/usr/bin/env python3
"""Owned, reconnectable jobs. SSH sessions never own the inference process.

Control accepts one JSON object on stdin and emits one JSON object on stdout.
Only this program creates/removes jobs; each directory has an ownership marker.
"""
import argparse
import base64
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import selectors
import shutil
import signal
import subprocess
import sys
import threading
import time
from protocol import (atomic_json, load_json, sha256, job_id, validate_result,bundle_sha256,private_root,
                      failure_path, read_public_failure)

TERMINAL = {'succeeded', 'failed', 'cancelled'}
MAX_AUDIO_BYTES = 20 * 1024**3
MAX_RESULT_BYTES = 64 * 1024**2


def owned_job(root, identity):
    folder = root/'jobs'/job_id(identity)
    if folder.is_symlink() or not folder.is_dir():
        raise ValueError('Unknown owned job')
    marker = load_json(folder/'owner.json')
    if marker != dict(protocol_version=1, request_id=identity, uid=os.getuid()):
        raise ValueError('Job ownership marker does not match')
    return folder


def fingerprint(request):
    # A reconnect may use a new local source pathname, but never different
    # bytes or different inference choices with the same request identifier.
    value = {k:v for k,v in request.items() if k not in ('audio_path','work_dir','request_id')}
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def create(root, request):
    identity = job_id(request['request_id'])
    folder = root/'jobs'/identity
    (root/'jobs').mkdir(exist_ok=True,mode=0o700)
    try:
        folder.mkdir(mode=0o700)
    except FileExistsError:
        folder=owned_job(root,identity)
        if load_json(folder/'status.json')['request_fingerprint'] != fingerprint(request):
            raise ValueError('This job identifier already belongs to different input/configuration')
        return load_json(folder/'status.json')
    atomic_json(folder/'owner.json',dict(protocol_version=1,request_id=identity,uid=os.getuid()))
    atomic_json(folder/'request.json',request)
    status=dict(protocol_version=1,request_id=identity,state='created',request_fingerprint=fingerprint(request),
                audio_sha256=request.get('audio_sha256'),created_at=time.time())
    atomic_json(folder/'status.json',status)
    return status


def receive(root,identity,expected,kind='audio'):
    folder=owned_job(root,identity)
    with (folder/'upload.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        return receive_locked(root,identity,expected,kind)


def receive_locked(root, identity, expected,kind='audio'):
    folder=owned_job(root,identity)
    if load_json(folder/'status.json')['state']!='created':
        raise ValueError('Audio upload is allowed only before job submission')
    request=load_json(folder/'request.json')
    if kind not in ('audio','native_bundle'):
        raise ValueError('Unknown owned upload kind')
    checksum_key='native_sha256' if kind=='native_bundle' else 'audio_sha256'
    if expected!=request.get(checksum_key):
        raise ValueError('Upload checksum does not match the immutable job request')
    digest=hashlib.sha256()
    total=0
    temporary=folder/('native-bundle.partial' if kind=='native_bundle' else 'input.partial')
    try:
        with temporary.open('wb') as stream:
            os.fchmod(stream.fileno(),0o600)
            for block in iter(lambda:sys.stdin.buffer.read(1024*1024),b''):
                total+=len(block)
                if total>(2*1024**3 if kind=='native_bundle' else MAX_AUDIO_BYTES): raise ValueError('Owned upload exceeds its size limit')
                digest.update(block);stream.write(block)
        if digest.hexdigest()!=expected: raise ValueError('Uploaded audio checksum mismatch')
        temporary.replace(folder/('native-bundle.tar' if kind=='native_bundle' else 'input.audio'))
    finally:
        temporary.unlink(missing_ok=True)
    return dict(version=1,request_id=identity,sha256=expected,audio_sha256=expected,bytes=total)


def selected_python(root, override=None):
    if override:
        return override
    current=root/'current/bin/python'
    return str(current) if current.is_file() else sys.executable


def snapshot_python(root,override=None):
    selected=selected_python(root,override)
    if '/' not in selected:
        return selected
    path=Path(selected).expanduser().absolute()
    # Resolve the release directory (current may change), but never resolve the
    # bin/python symlink itself: that would run the base interpreter and lose
    # this virtual environment's site-packages and sys.prefix.
    return str(path.parent.resolve()/path.name)


def snapshot_program(python):
    runtime=Path(python).parent.parent
    worker=runtime/'worker/worker.py'
    if worker.is_file():
        manifest=load_json(runtime/'runtime.json')
        if not manifest.get('worker_bundle_sha256') or bundle_sha256(worker.parent,manifest)!=manifest['worker_bundle_sha256']:
            raise ValueError('The selected runtime worker snapshot failed verification')
        return str(worker),True
    return str(Path(__file__).with_name('worker.py')),False


def submit(root,identity,python=None):
    folder=owned_job(root,identity)
    with (folder/'submit.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        status=load_json(folder/'status.json')
        if status['state']!='created': return status
        request=load_json(folder/'request.json')
        if request.get('operation') not in ('probe','prepare','install_runtime'):
            if not (folder/'input.audio').is_file() or sha256(folder/'input.audio')!=request.get('audio_sha256'):
                raise ValueError('Upload and verify audio before submitting the job')
        if request.get('operation')=='install_runtime' and request.get('native_bundle'):
            bundle=folder/'native-bundle.tar'
            if not bundle.is_file() or sha256(bundle)!=request.get('native_sha256'):
                raise ValueError('Upload and verify the native helper bundle before installation')
            request['native_bundle']=str(bundle)
        request['audio_path']=str(folder/'input.audio')
        request['work_dir']=str(folder/'work')
        request.setdefault('cache_dir',str(root/'cache'))
        (folder/'work').mkdir(exist_ok=True,mode=0o700)
        atomic_json(folder/'request.json',request)
        # The bootstrap Python runs supervision; the selected isolated runtime
        # runs inference. Updating current cannot alter an already submitted job.
        python=snapshot_python(root,python)
        worker,snapshot_verified=(str(Path(__file__).with_name('worker.py')),False) if request.get('operation')=='install_runtime' else snapshot_program(python)
        status.update(state='queued',runtime_python=python,runtime_worker=worker,worker_snapshot_verified=snapshot_verified)
        atomic_json(folder/'status.json',status)
        log=(folder/'supervisor.log').open('ab')
        os.fchmod(log.fileno(),0o600)
        detached=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--root',str(root),'--run',identity],
                         stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True,close_fds=True)
        # A long-lived local caller can reap it; an SSH controller may exit
        # immediately without making the detached job depend on that session.
        threading.Thread(target=detached.wait,daemon=True).start()
        log.close()
    return status


def update(folder,**changes):
    with (folder/'status.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        status=load_json(folder/'status.json');status.update(changes);status['updated_at']=time.time()
        atomic_json(folder/'status.json',status)
        return status


def record_transfer(folder,kind,seconds):
    if kind not in ('upload','result_download') or isinstance(seconds,bool) or not isinstance(seconds,(int,float)) or not math.isfinite(seconds) or not 0<=seconds<=86400:
        raise ValueError('Invalid transfer measurement')
    with (folder/'status.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        status=load_json(folder/'status.json')
        if kind=='result_download' and status['state']!='succeeded':
            raise ValueError('A successful result is required for download timing')
        timings=status.setdefault('transport',{})
        key=kind+'_wall_seconds'
        if key not in timings:
            timings[key]=float(seconds)
            timings['measurement_scope']='First completed client transfer including connection, integrity checks and acknowledgement; not inference time'
            timings['unit']='seconds'
            status['updated_at']=time.time();atomic_json(folder/'status.json',status)
        return dict(protocol_version=1,request_id=status['request_id'],state=status['state'],transport=timings)


def remove_media(folder):
    (folder/'input.audio').unlink(missing_ok=True)
    (folder/'input.partial').unlink(missing_ok=True)
    (folder/'native-bundle.tar').unlink(missing_ok=True)
    (folder/'native-bundle.partial').unlink(missing_ok=True)
    shutil.rmtree(folder/'work',ignore_errors=True)


def supervise(root,identity):
    folder=owned_job(root,identity)
    process=None
    terminal=None
    try:
        # The lock is target-wide, including separate application instances.
        # It preserves existing external services and prevents overlapping jobs.
        with (root/'execution.lock').open('a') as gate:
            while True:
                if (folder/'cancel').exists():
                    terminal=dict(state='cancelled');return
                try:
                    fcntl.flock(gate,fcntl.LOCK_EX|fcntl.LOCK_NB);break
                except BlockingIOError: time.sleep(.2)
            status=update(folder,state='running',started_at=time.time(),supervisor_pid=os.getpid())
            environment=dict(os.environ,PARLEY_WORKER_ROOT=str(root),PARLEY_WORKER_CACHE=str(root/'cache'))
            runtime=Path(status['runtime_python']).parent.parent
            libraries=[str(p) for p in runtime.glob('lib/python*/site-packages/nvidia/*/lib')]
            libraries += [str(runtime/'native'),str(root/'native')]
            # CTranslate2 needs the isolated cuBLAS/cuDNN shared libraries too;
            # changing this child's search path never changes system services.
            if sys.platform=='linux':
                environment['LD_LIBRARY_PATH']=os.pathsep.join(libraries+[environment.get('LD_LIBRARY_PATH','')])
            elif sys.platform=='darwin':
                environment['DYLD_LIBRARY_PATH']=os.pathsep.join(libraries+[environment.get('DYLD_LIBRARY_PATH','')])
            # Worker and its native decoder / FFmpeg children share a dedicated
            # process group. Cancellation addresses this held child, never a
            # stale PID loaded from disk after reconnecting.
            with (folder/'error.log').open('wb') as error, (folder/'progress.jsonl').open('wb') as events:
                os.fchmod(error.fileno(),0o600);os.fchmod(events.fileno(),0o600)
                request=load_json(folder/'request.json')
                command=([sys.executable,str(Path(__file__).with_name('setup_runtime.py')),
                    '--directory',str(root),'--device',request['device'],'--response',str(folder/'result.json')]
                    if request.get('operation')=='install_runtime' else
                    [status['runtime_python'],status.get('runtime_worker',str(Path(__file__).with_name('worker.py'))),
                    '--request',str(folder/'request.json'),'--response',str(folder/'result.json')])
                if request.get('operation')=='install_runtime' and request.get('native_bundle'):
                    command+=['--native-bundle',request['native_bundle'],'--native-sha256',request['native_sha256']]
                if request.get('operation')=='install_runtime' and request.get('rollback'):
                    command+=['--rollback']
                failure_path(folder/'result.json').unlink(missing_ok=True)
                process=subprocess.Popen(command,
                    stdout=subprocess.PIPE,stderr=error,stdin=subprocess.DEVNULL,env=environment,start_new_session=True)
                poller=selectors.DefaultSelector();poller.register(process.stdout,selectors.EVENT_READ)
                pending=b'';written=0
                while process.poll() is None:
                    if (folder/'cancel').exists() or time.time()-status['started_at']>24*3600:
                        os.killpg(process.pid,signal.SIGTERM)
                        try: process.wait(timeout=3)
                        except subprocess.TimeoutExpired:
                            os.killpg(process.pid,signal.SIGKILL);process.wait()
                        terminal=dict(state='cancelled' if (folder/'cancel').exists() else 'failed',error='Job cancelled or exceeded its 24-hour time limit')
                        return
                    for key,_ in poller.select(.2):
                        block=os.read(key.fileobj.fileno(),65536)
                        pending+=block
                        if len(pending)>1024*1024: raise ValueError('Oversized worker progress event')
                        while b'\n' in pending:
                            line,pending=pending.split(b'\n',1)
                            written+=len(line)+1
                            if written>32*1024*1024: raise ValueError('Worker progress exceeds its limit')
                            events.write(line+b'\n');events.flush()
                            try:
                                event=json.loads(line)
                                if isinstance(event,dict) and isinstance(event.get('stage'),str):
                                    update(folder,progress={k:event[k] for k in ('stage','percent','message') if k in event})
                            except (ValueError,TypeError): pass
                poller.close()
            if process.returncode:
                terminal=dict(state='failed',
                    error='Worker failed validation or execution; private diagnostic log is retained',
                    **read_public_failure(folder/'result.json'))
                return
            result_file=folder/'result.json'
            if not result_file.is_file() or result_file.stat().st_size>MAX_RESULT_BYTES:
                raise ValueError('Missing or oversized terminal result')
            request=load_json(folder/'request.json');result=load_json(result_file)
            if request.get('operation') not in ('probe','prepare','install_runtime'): validate_result(result,request)
            terminal=dict(state='succeeded',result_sha256=sha256(result_file),completed_at=time.time())
    except Exception:
        terminal=dict(state='failed',error='Worker failed validation or execution; private diagnostic log is retained')
        import traceback
        traceback.print_exc()
    finally:
        if process is not None:
            try: os.killpg(process.pid,signal.SIGKILL)
            except ProcessLookupError: pass
            process.wait()
        remove_media(folder)
        if terminal:
            if terminal['state']!='succeeded':
                (folder/'result.json.partial').unlink(missing_ok=True)
                (folder/'result.json').unlink(missing_ok=True)
            update(folder,**terminal)


def control(root,message):
    action=message['action']
    if action=='create': return create(root,message['request'])
    identity=job_id(message['request_id']);folder=owned_job(root,identity)
    if action=='submit': return submit(root,identity,message.get('python'))
    if action=='record_transfer': return record_transfer(folder,message.get('kind'),message.get('seconds'))
    status=load_json(folder/'status.json')
    if action=='status':
        # A supervisor dying is explicit. Never turn it into success, infer a
        # different target, or restart computation using an existing identifier.
        if status['state']=='running':
            try: os.kill(status['supervisor_pid'],0)
            except ProcessLookupError: status=update(folder,state='failed',error='Supervisor stopped; inspect this owned job before cleanup')
        return {k:v for k,v in status.items() if k not in ('runtime_python','runtime_worker','supervisor_pid')}
    if action=='cancel':
        if status['state'] not in TERMINAL:
            (folder/'cancel').touch(mode=0o600)
            if status['state']=='created':
                remove_media(folder);status=update(folder,state='cancelled')
        return {k:v for k,v in status.items() if k not in ('runtime_python','runtime_worker','supervisor_pid')}
    if action=='result':
        if status['state']!='succeeded': raise ValueError('The job has no successful terminal result')
        file=folder/'result.json'
        if sha256(file)!=status['result_sha256']: raise ValueError('Terminal result checksum changed')
        if message.get('result_format')=='base64':
            return dict(protocol_version=1,request_id=identity,result_sha256=status['result_sha256'],
                        result_base64=base64.b64encode(file.read_bytes()).decode('ascii'))
        return dict(protocol_version=1,request_id=identity,result_sha256=status['result_sha256'],result=load_json(file))
    if action=='cleanup':
        if status['state'] not in TERMINAL: raise ValueError('Cancel and wait for termination before cleanup')
        shutil.rmtree(folder)
        return dict(protocol_version=1,request_id=identity,state='cleaned')
    raise ValueError('Unknown job control operation')


def main():
    os.umask(0o077)
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True,type=Path)
    parser.add_argument('--run');parser.add_argument('--receive');parser.add_argument('--sha256')
    parser.add_argument('--receive-kind',choices=('audio','native_bundle'),default='audio')
    args=parser.parse_args();root=private_root(args.root)
    if args.run: supervise(root,args.run);return
    try:
        result=receive(root,args.receive,args.sha256,args.receive_kind) if args.receive else control(root,json.load(sys.stdin))
        print(json.dumps(result,allow_nan=False),flush=True)
    except Exception as error:
        print(json.dumps(dict(protocol_version=1,error=type(error).__name__+': '+str(error))),flush=True)
        sys.exit(1)


if __name__=='__main__': main()
