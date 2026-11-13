"""Synthetic protocol, setup and real child-process lifecycle regressions."""
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import types
import unittest
import wave
from unittest.mock import patch
import protocol
import assets
import qualification
import setup_runtime
import supervisor
from worker import validate_clean_sources,extraction_windows
import worker


def analysis():
    return dict(version=1,duration=10.,segments=[dict(start=0.,end=10.,text='Fictional fixture',speaker='A')],
        words=[],turns=[dict(start=0.,end=10.,speaker='A')],overlaps=[],embeddings=[
        dict(speaker='A',model_id='fictional-space-v1',vector=[1.,0.],clean_seconds=10.,sample_count=1)])


class ValidationTests(unittest.TestCase):
    def test_inflight_artifact_snapshot_rejects_changed_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'fictional-model';path.write_bytes(b'fictional checkpoint A')
            request=dict(model_paths={'campplus':str(path)})
            record=assets.resolve(request,'campplus')
            request['_resolved_artifacts']={'campplus':record}
            self.assertEqual(assets.resolve(request,'campplus')['sha256'],record['sha256'])
            path.write_bytes(b'fictional checkpoint B')
            with self.assertRaises(ValueError): assets.resolve(request,'campplus')

    def test_schema_accepts_fictional_fixture(self):
        protocol.validate_result(analysis())

    def test_invalid_timestamp_diagnostic_identifies_location_without_text(self):
        value = analysis()
        value['segments'][0].update(start=11., end=10.)
        with self.assertRaisesRegex(ValueError, r'Invalid result timestamp in segments\[0\]') as error:
            protocol.validate_result(value)
        self.assertIn("'start': 11.0", str(error.exception))
        self.assertIn('duration=10.0', str(error.exception))
        self.assertNotIn('Fictional fixture', str(error.exception))

    def test_rejects_nonfinite_times_vectors_and_wrong_audio(self):
        for mutate in (lambda a:a.update(duration=float('nan')),
                       lambda a:a['embeddings'][0].update(vector=[float('inf'),0.]),
                       lambda a:a['words'].append(dict(start=0.,end=20.,text='invalid')),
                       lambda a:a['segments'][0].update(speaker='missing'),
                       lambda a:a['embeddings'][0].update(vector=[2.,0.])):
            value=analysis();mutate(value)
            with self.assertRaises(ValueError): protocol.validate_result(value)
        with self.assertRaises(ValueError):
            protocol.validate_result(analysis(),dict(request_id='fictional-identifier'))

    def test_dimension_and_space_identity_are_not_interchangeable(self):
        value=analysis()
        value['embeddings'][0]['embedding_space']=dict(id='fictional-space-v1',dimension=3)
        with self.assertRaises(ValueError): protocol.validate_result(value)
        value=analysis();value['embeddings'].append(dict(value['embeddings'][0],vector=[1.,0.,0.]))
        with self.assertRaises(ValueError): protocol.validate_result(value)

    def test_unknown_version_and_experimental_or_implicit_precision_fail(self):
        for request in (dict(version=2),dict(device='auto'),dict(config=dict(speech_model='large-v3-turbo-mlx-q4'))):
            with self.assertRaises(ValueError): protocol.normalize_request(request)

    def test_auto_is_control_policy_and_never_an_inference_device(self):
        request=protocol.normalize_request(dict(operation='probe',device='auto'))
        self.assertEqual(request['speech_device'],'auto')
        with self.assertRaises(ValueError): protocol.normalize_request(dict(operation='process',device='auto'))
        with patch.object(worker.platform,'system',return_value='Linux'),patch.object(worker.platform,'machine',return_value='x86_64'):
            self.assertEqual(worker.probe_speech_devices('auto','faster_whisper',False),['cpu'])
            self.assertEqual(worker.probe_speech_devices('auto','faster_whisper',True),['cuda','cpu'])
            self.assertEqual(worker.probe_speech_devices('cpu','faster_whisper',True),['cpu'])
        with patch.object(worker.platform,'system',return_value='Darwin'),patch.object(worker.platform,'machine',return_value='arm64'):
            self.assertEqual(worker.probe_speech_devices('auto','mlx_whisper',False),['metal'])

    def test_clean_intervals_union_duplicate_turns_and_subtract_overlap(self):
        turns=[dict(start=0.,end=10.,speaker='A')]*2+[dict(start=4.,end=6.,speaker='B')]
        simultaneous=[dict(start=4.,end=6.,speakers=['A','B'])]
        self.assertEqual(protocol.clean_intervals(turns,simultaneous),[
            dict(start=0.,end=4.,speaker='A',channel=0),dict(start=6.,end=10.,speaker='A',channel=0)])

    def test_extraction_rejects_overlap_and_short_or_untracked_evidence(self):
        span=dict(start=0.,end=4.,speaker='A',channel=0)
        for request in (dict(source_intervals=[span,span]),dict(source_intervals=[dict(span,end=2.)]),
                        dict(source_intervals=[span],overlaps=[dict(start=2.,end=3.)])):
            with self.assertRaises(ValueError): validate_clean_sources(request,10.)
        value=analysis();value['embeddings'][0]['source_intervals']=[span]
        protocol.validate_result(value,dict(operation='extract_embeddings',source_intervals=[span]))
        value['embeddings'][0]['source_intervals']=[dict(span,end=5.)]
        with self.assertRaises(ValueError):
            protocol.validate_result(value,dict(operation='extract_embeddings',source_intervals=[span]))

    def test_json_does_not_accept_nan_or_path_traversal_job(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'invalid.json';path.write_text('{"number":NaN}')
            with self.assertRaises(ValueError): protocol.load_json(path)
        for value in ('../foreign','/tmp/foreign','abc','a'*81):
            with self.assertRaises(ValueError): protocol.job_id(value)

    def test_extraction_bounds_memory_and_preserves_actual_source_containment(self):
        spans=[dict(speaker='A',start=float(i*200),end=float(i*200+100),channel=0) for i in range(5)]
        windows=extraction_windows(spans)
        self.assertEqual(len(windows),3)
        self.assertTrue(all(w['end']-w['start']<=10 for w in windows))
        self.assertTrue(all(any(s['start']<=w['start']<w['end']<=s['end'] for s in spans) for w in windows))
        value=analysis();value['embeddings'][0]['source_intervals']=[dict(start=0.,end=4.,speaker='A',channel=0)]
        protocol.validate_result(value,dict(operation='embed',source_intervals=value['embeddings'][0]['source_intervals']))



class WorkerRootTests(unittest.TestCase):
    def test_old_marker_migrates_after_device_number_changes_without_touching_assets(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)/'worker';root.mkdir(mode=0o700)
            status=root.stat()
            old=dict(version=1,kind='parley-offline-worker',uid=os.getuid(),
                     directory_device=status.st_dev+1,directory_inode=status.st_ino)
            protocol.atomic_json(root/protocol.ROOT_MARKER,old)
            cache=root/'cache';cache.mkdir();asset=cache/'fictional-model';asset.write_bytes(b'keep model')
            before=asset.stat()
            self.assertEqual(protocol.private_root(root),root)
            expected=dict(version=2,kind='parley-offline-worker',uid=os.getuid(),directory_path=str(root.resolve()))
            self.assertEqual(protocol.load_json(root/protocol.ROOT_MARKER),expected)
            marker_before=(root/protocol.ROOT_MARKER).stat().st_mtime_ns
            self.assertEqual(protocol.private_root(root),root)
            self.assertEqual((root/protocol.ROOT_MARKER).stat().st_mtime_ns,marker_before)
            self.assertEqual(asset.read_bytes(),b'keep model')
            self.assertEqual(asset.stat().st_mtime_ns,before.st_mtime_ns)
            self.assertEqual(root.stat().st_mode,status.st_mode)

    def test_path_marker_survives_directory_restore_at_same_location(self):
        with tempfile.TemporaryDirectory() as directory:
            parent=Path(directory);root=protocol.private_root(parent/'worker')
            saved=protocol.load_json(root/protocol.ROOT_MARKER)
            root.rename(parent/'previous')
            root.mkdir(mode=0o700)
            protocol.atomic_json(root/protocol.ROOT_MARKER,saved)
            self.assertEqual(protocol.private_root(root),root)
            # The same marker must not authorize its new location.
            with self.assertRaises(ValueError):protocol.private_root(parent/'previous')

    def test_old_marker_mismatch_rejects_without_rewriting_it(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)/'worker';root.mkdir(mode=0o700)
            old=dict(version=1,kind='parley-offline-worker',uid=os.getuid(),
                     directory_device=root.stat().st_dev,directory_inode=root.stat().st_ino)
            for change in (dict(uid=os.getuid()+1),dict(directory_inode=root.stat().st_ino+1),
                           dict(kind='unrelated'),dict(version=99),dict(directory_device='invalid')):
                value=dict(old,**change);protocol.atomic_json(root/protocol.ROOT_MARKER,value)
                with self.assertRaises(ValueError):protocol.private_root(root)
                self.assertEqual(protocol.load_json(root/protocol.ROOT_MARKER),value)

    def test_interrupted_marker_migration_keeps_the_original_marker(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)/'worker';root.mkdir(mode=0o700)
            old=dict(version=1,kind='parley-offline-worker',uid=os.getuid(),
                     directory_device=root.stat().st_dev+1,directory_inode=root.stat().st_ino)
            protocol.atomic_json(root/protocol.ROOT_MARKER,old)
            with patch.object(Path,'replace',side_effect=OSError('fictional write failure')):
                with self.assertRaises(OSError):protocol.private_root(root)
            self.assertEqual(protocol.load_json(root/protocol.ROOT_MARKER),old)
            self.assertEqual({p.name for p in root.iterdir()},{protocol.ROOT_MARKER})

    def test_unrelated_existing_directories_are_never_changed(self):
        with tempfile.TemporaryDirectory() as directory:
            parent=Path(directory)
            for mode in (0o700,0o755,0o1777):
                root=parent/('unrelated-'+str(mode));root.mkdir();root.chmod(mode)
                sentinel=root/'keep.txt';sentinel.write_text('fictional unrelated contents')
                before=root.stat()
                for check in (protocol.private_root,supervisor.private_root,
                              lambda p:setup_runtime.install(p,'cpu'),setup_runtime.rollback):
                    with self.assertRaises(ValueError):check(root)
                    self.assertEqual(root.stat().st_mode,before.st_mode)
                    self.assertEqual(sentinel.read_text(),'fictional unrelated contents')
                    self.assertEqual({p.name for p in root.iterdir()},{'keep.txt'})

    def test_shared_and_home_paths_reject_without_filesystem_writes(self):
        # Mock all writes: this test never changes or creates a real system path.
        with patch.object(Path,'mkdir',side_effect=AssertionError('No mkdir')),patch.object(Path,'chmod',side_effect=AssertionError('No chmod')),patch.object(Path,'home',return_value=Path('/fictional-home')):
            for root in ('/','/tmp','/var/tmp','/usr/local','/fictional-home','/tmp/child/..'):
                with self.assertRaises(ValueError):protocol.private_root(root)

    def test_parallel_first_root_initialization_has_one_valid_marker(self):
        from concurrent.futures import ThreadPoolExecutor
        with tempfile.TemporaryDirectory() as directory:
            source=str(Path(__file__).parent)
            for index in range(3):
                root=Path(directory)/str(index)/'worker'
                code='import sys;sys.path.insert(0,sys.argv[1]);from protocol import private_root;private_root(sys.argv[2])'
                def create(_):
                    return subprocess.run([sys.executable,'-c',code,source,str(root)],capture_output=True).returncode
                with ThreadPoolExecutor(max_workers=8)as pool:
                    self.assertEqual(list(pool.map(create,range(8))),[0]*8)
                self.assertEqual(protocol.private_root(root),root)
                self.assertEqual({p.name for p in root.iterdir()},{protocol.ROOT_MARKER})

    def test_fresh_leaf_marker_and_symlink_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            parent=Path(directory);root=protocol.private_root(parent/'nested/worker')
            self.assertEqual(root.stat().st_mode&0o777,0o700)
            self.assertEqual(root.parent.stat().st_mode&0o777,0o700)
            self.assertTrue((root/protocol.ROOT_MARKER).is_file())
            self.assertEqual(protocol.private_root(root),root)
            linked=parent/'linked';linked.symlink_to(root)
            with self.assertRaises(ValueError):protocol.private_root(linked)
            copied=parent/'copied';shutil.copytree(root,copied)
            with self.assertRaises(ValueError):protocol.private_root(copied)

    def test_legacy_adoption_requires_contained_verified_release(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)/'legacy';root.mkdir(mode=0o700)
            release=root/'runtimes/fictional-release';release.mkdir(parents=True)
            setup_runtime.snapshot_worker(release)
            manifest=dict(**protocol.bundle_description(release/'worker'),runtime_id=release.name,runtime_verified=True,native_files={})
            protocol.atomic_json(release/'runtime.json',manifest)
            (root/'current').symlink_to(release)
            before=root.stat().st_mode
            self.assertEqual(protocol.private_root(root),root)
            self.assertEqual(root.stat().st_mode,before)
            self.assertTrue((root/protocol.ROOT_MARKER).is_file())
            (root/protocol.ROOT_MARKER).unlink();(release/'worker/worker.py').write_text('changed fictional snapshot')
            with self.assertRaises(ValueError):protocol.private_root(root)
            self.assertFalse((root/protocol.ROOT_MARKER).exists())


class InstallationTests(unittest.TestCase):
    def test_auto_install_policy_resolves_without_installing_anything(self):
        with patch.object(setup_runtime.platform,'system',return_value='Linux'),patch.object(setup_runtime,'cuda_driver_available',return_value=False):
            self.assertEqual(setup_runtime.resolve_install_device('auto'),'cpu')
            self.assertEqual(setup_runtime.resolve_install_device('vulkan'),'vulkan')
        with patch.object(setup_runtime.platform,'system',return_value='Linux'),patch.object(setup_runtime,'cuda_driver_available',return_value=True):
            self.assertEqual(setup_runtime.resolve_install_device('auto'),'cuda')
        with patch.object(setup_runtime.platform,'system',return_value='Darwin'),patch.object(setup_runtime.platform,'machine',return_value='arm64'):
            self.assertEqual(setup_runtime.resolve_install_device('auto'),'metal')

    def test_native_bundle_must_run_matching_protocol_before_activation(self):
        with tempfile.TemporaryDirectory() as directory:
            root=protocol.private_root(Path(directory)/'owned-worker');native=root/'native';native.mkdir()
            helper=native/'parley_native_worker'
            helper.write_text('#!'+sys.executable+'\nimport json\nprint(json.dumps(dict(protocol_version=1,kind="native_speaker",runtime_ready=True,runtime={"fixture":True})))\n')
            helper.chmod(0o700)
            verified=setup_runtime.verify_native(root,{})
            self.assertTrue(verified['parley_native_worker']['runnable'])
            self.assertFalse(verified['parley_native_worker']['model_verified'])
            helper.write_bytes(b'incompatible executable fixture')
            with self.assertRaises(OSError): setup_runtime.verify_native(root,{})

    def test_runtime_snapshot_preserves_real_virtualenv_python_symlink(self):
        with tempfile.TemporaryDirectory() as directory:
            root=protocol.private_root(Path(directory)/'owned-worker');release=root/'release'
            subprocess.run([sys.executable,'-m','venv','--without-pip',str(release)],check=True)
            setup_runtime.activate(root,release)
            python=supervisor.snapshot_python(root)
            self.assertEqual(python,str(release/'bin/python'))
            prefix=subprocess.check_output([python,'-c','import sys;print(sys.prefix)'],text=True).strip()
            self.assertEqual(Path(prefix),release)

    def test_atomic_activation_and_rollback_keep_previous_release(self):
        with tempfile.TemporaryDirectory() as directory:
            root=protocol.private_root(Path(directory)/'owned-worker');first=root/'first';second=root/'second'
            first.mkdir();second.mkdir()
            for release in (first,second):
                digest=setup_runtime.snapshot_worker(release)
                protocol.atomic_json(release/'runtime.json',dict(**protocol.bundle_description(release/'worker'),native_files={}))
            setup_runtime.activate(root,first);setup_runtime.activate(root,second)
            self.assertEqual((root/'current').resolve(),second)
            setup_runtime.rollback(root)
            self.assertEqual((root/'current').resolve(),first)
            self.assertEqual((root/'previous').resolve(),second)

    def test_legacy_or_changed_worker_cannot_claim_complete_rollback(self):
        with tempfile.TemporaryDirectory() as directory:
            root=protocol.private_root(Path(directory)/'owned-worker');first=root/'first';second=root/'second';first.mkdir();second.mkdir()
            setup_runtime.activate(root,first);setup_runtime.activate(root,second)
            with self.assertRaisesRegex(ValueError,'legacy'):
                setup_runtime.rollback(root)
            self.assertEqual((root/'current').resolve(),second)
            digest=setup_runtime.snapshot_worker(first)
            protocol.atomic_json(first/'runtime.json',dict(**protocol.bundle_description(first/'worker'),native_files={}))
            (first/'worker/worker.py').write_text('changed fictional worker')
            with self.assertRaisesRegex(ValueError,'changed'):
                setup_runtime.rollback(root)
            self.assertEqual((root/'current').resolve(),second)

    def test_new_eight_file_release_can_restore_known_seven_file_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            root=protocol.private_root(Path(directory)/'owned-worker');old=root/'old-seven';new=root/'new-eight';old.mkdir();new.mkdir()
            setup_runtime.snapshot_worker(old);(old/'worker/resources.py').unlink()
            (old/'worker/worker.py').write_text("print('fictional prior worker')\n")
            prior=protocol.bundle_description(old/'worker',version=1)
            # Original complete releases predated the declared-file manifest.
            protocol.atomic_json(old/'runtime.json',dict(worker_bundle_sha256=prior['worker_bundle_sha256'],native_files={}))
            setup_runtime.snapshot_worker(new)
            protocol.atomic_json(new/'runtime.json',dict(**protocol.bundle_description(new/'worker'),native_files={}))
            setup_runtime.activate(root,old);setup_runtime.activate(root,new)
            restored=setup_runtime.rollback(root)
            self.assertTrue(restored['worker_snapshot_verified']);self.assertEqual((root/'current').resolve(),old)
            worker,verified=supervisor.snapshot_program(str(old/'bin/python'))
            self.assertTrue(verified);self.assertEqual(Path(worker),old/'worker/worker.py')
            bad=dict(prior);bad['worker_bundle_files']=dict(prior['worker_bundle_files'],**{'../unexpected.py':'0'*64})
            with self.assertRaises(ValueError):protocol.bundle_sha256(old/'worker',bad)

    def test_failed_install_never_activates_incomplete_release(self):
        with tempfile.TemporaryDirectory() as directory:
            root=protocol.private_root(Path(directory)/'owned-worker');old=root/'old';old.mkdir();setup_runtime.activate(root,old)
            commands=[]
            def fail(command,environment):
                commands.append((command,environment))
                raise subprocess.CalledProcessError(1,command)
            with patch.object(setup_runtime,'uv_executable',return_value='fictional-uv'),patch.object(setup_runtime,'run',side_effect=fail):
                with self.assertRaises(subprocess.CalledProcessError): setup_runtime.install(root,'cpu')
            self.assertEqual((root/'current').resolve(),old)
            self.assertEqual(list((root/'runtimes').iterdir()),[])
            command,environment=commands[0]
            self.assertNotIn('--managed-python',command)
            self.assertEqual(environment['UV_PYTHON_PREFERENCE'],'only-managed')


class QualificationTests(unittest.TestCase):
    def test_failed_mps_execution_probe_does_not_retry_on_cpu(self):
        request=protocol.normalize_request(dict(operation='probe',device='metal',
            config=dict(speaker_pipeline='community1')))
        with (patch.dict(sys.modules,{'torch':None}),
              patch.object(worker,'versions',return_value=qualification.PACKAGES),
              patch.object(worker,'runtime_provenance',return_value={}),
              patch.object(worker,'registry',return_value={}),
              patch.object(worker,'ffmpeg_executable',side_effect=ValueError('fixture')),
              patch.object(worker,'helper_executable',side_effect=ValueError('fixture')),
              patch.object(worker,'native_speakers',side_effect=ValueError('fixture')),
              patch.object(worker,'mps_qualified',return_value=True),
              patch.object(worker,'release_models'),
              patch.object(worker,'diarize',side_effect=RuntimeError('fixture GPU failure')) as diarize):
            result=worker.probe(request)
        entry=next(e for e in result['speaker_pipelines'] if e['id']=='community1')
        self.assertEqual(entry['device'],'mps')
        self.assertFalse(entry['available'])
        self.assertFalse(entry['verified'])
        diarize.assert_called_once()
        self.assertEqual(diarize.call_args.args[0]['speaker_device'],'mps')

    def test_requested_mps_needs_qualification_and_cpu_remains_explicit(self):
        with patch.object(worker.platform,'system',return_value='Darwin'),patch.object(worker.platform,'machine',return_value='arm64'):
            with patch.object(worker,'mps_qualified',return_value=False):
                with self.assertRaises(ValueError): worker.stage_device(dict(speaker_device='mps'),'speaker')
                self.assertEqual(worker.stage_device(dict(speaker_device='cpu'),'speaker'),'cpu')
            with patch.object(worker,'mps_qualified',return_value=True):
                self.assertEqual(worker.stage_device(dict(speaker_device='mps'),'speaker'),'mps')

    def test_mps_support_is_bound_to_versions_artifact_and_platform(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);model=root/'model';model.write_bytes(b'fictional speaker model')
            request=dict(cache_dir=str(root/'cache'),model_paths={'community1':str(model)})
            artifact=assets.resolve(request,'community1')
            runtime=dict(packages=qualification.PACKAGES)
            analysis=dict(diarization_device='mps',provenance=dict(runtime=runtime,
                embedding_reference_verified=True,speaker_artifact=artifact))
            report=dict(reference_device='cpu',candidate_device='mps',fixed_reference_runtime=runtime,
                timeline=dict(reference_speakers=2,candidate_speakers=2,reference_speaker_seconds=60,
                    speaker_time_error_fraction=0.,overlap_intersection_over_union=1.),
                fixed_clean_interval_embeddings=[dict(compatible=True,cosine_similarity=1.,max_absolute_difference=0.)])
            a=root/'analysis.json';r=root/'report.json';protocol.atomic_json(a,analysis);protocol.atomic_json(r,report)
            with patch.object(qualification.platform,'system',return_value='Darwin'),patch.object(qualification.platform,'machine',return_value='arm64'),patch.object(qualification,'installed_packages',return_value=qualification.PACKAGES):
                self.assertFalse(qualification.mps_qualified(request))
                qualification.accept_mps(request,r,a)
                self.assertTrue(qualification.mps_qualified(request))
                with patch.dict(os.environ,{'PYTORCH_ENABLE_MPS_FALLBACK':'1'}):
                    self.assertFalse(qualification.mps_qualified(request))
                with patch.object(qualification,'installed_packages',return_value={}):
                    self.assertFalse(qualification.mps_qualified(request))
                for release in ('27.2.0', '28.0.0'):
                    with patch.object(qualification.platform,'release',return_value=release):
                        self.assertTrue(qualification.mps_qualified(request))
                # Retain the OS version that was actually tested, rather than
                # rewriting it to pretend a new parity comparison took place.
                saved=protocol.load_json(root/'cache/mps-qualifications-v1.json')[artifact['sha256']]
                self.assertEqual(saved['platform'],qualification.platform_identity())
                with patch.object(qualification.platform,'machine',return_value='x86_64'):
                    self.assertFalse(qualification.mps_qualified(request))
                with patch.object(qualification.platform,'system',return_value='Linux'):
                    self.assertFalse(qualification.mps_qualified(request))
                model.write_bytes(b'changed fictional speaker model')
                self.assertFalse(qualification.mps_qualified(request))


class StageContractTests(unittest.TestCase):
    def test_experimental_mps_fallback_rejected_before_standalone_model_load(self):
        def forbidden(*args,**kwargs):raise AssertionError('No model load or inference')
        torch=types.SimpleNamespace(backends=types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda:True)),ones=forbidden)
        modules={'torch':torch,'pyannote.audio':types.SimpleNamespace(Audio=forbidden),
            'pyannote.audio.pipelines.speaker_diarization':types.SimpleNamespace(SpeakerDiarization=object),
            'pyannote.audio.pipelines.speaker_verification':types.SimpleNamespace(PretrainedSpeakerEmbedding=forbidden)}
        with patch.dict(sys.modules,modules),patch.object(worker,'require_versions'),patch.object(worker.platform,'system',return_value='Darwin'),patch.object(worker.platform,'machine',return_value='arm64'),patch.dict(os.environ,{'PYTORCH_ENABLE_MPS_FALLBACK':'1'}):
            with self.assertRaisesRegex(ValueError,'fallback disabled'):
                worker.load_community_embedding(dict(speaker_device='mps',experimental_speaker_mps=True))

    def test_standalone_wespeaker_loader_never_constructs_full_pipeline(self):
        adapter=types.SimpleNamespace(sample_rate=16000)
        calls=[]
        def embedding(model,device):
            calls.append((model,device));return adapter
        def reference(context,*args,**kwargs):
            self.assertFalse(context.training)
            self.assertIs(context._embedding,adapter)
            self.assertFalse(hasattr(context,'_segmentation'))
            return 'package-reference'
        modules={
            'torch':types.SimpleNamespace(device=lambda v:v,set_num_threads=lambda n:None),
            'pyannote.audio':types.SimpleNamespace(Audio=lambda **kw:kw),
            'pyannote.audio.pipelines.speaker_diarization':types.SimpleNamespace(SpeakerDiarization=types.SimpleNamespace(get_embeddings=reference)),
            'pyannote.audio.pipelines.speaker_verification':types.SimpleNamespace(PretrainedSpeakerEmbedding=embedding)}
        record=dict(path='/fictional-model',id='community1')
        with patch.dict(sys.modules,modules),patch.object(worker,'require_versions'),patch.object(worker,'resolve',return_value=record),patch.object(worker,'load_community',side_effect=AssertionError('No full pipeline')):
            pipeline,actual,device,meter=worker.load_community_embedding(dict(speaker_device='cpu'))
        self.assertEqual(device,'cpu');self.assertEqual(actual,record)
        self.assertEqual(calls,[(dict(checkpoint='/fictional-model',subfolder='embedding'),'cpu')])
        self.assertEqual(pipeline.get_embeddings({},None),'package-reference')

    def test_camp_probe_uses_only_embedding_artifact_and_explicit_cpu(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);model=root/'fictional.onnx';model.write_bytes(b'fictional model')
            helper=root/'fake-native'
            helper.write_text('#!'+sys.executable+'''\nimport argparse,json
p=argparse.ArgumentParser();p.add_argument('--request');p.add_argument('--response');a=p.parse_args()
r=json.load(open(a.request));assert set(r['native_models'])=={'embedding'};assert r['speaker_device']=='cpu'
s=r['source_intervals'][0]
json.dump(dict(embeddings=[dict(speaker=s['speaker'],model_id='fictional-space',vector=[1.,0.],clean_seconds=3.8,source_intervals=[dict(s,start=.1,end=3.9)])],settings={}),open(a.response,'w'))
'''.replace('\n+','\n'))
            helper.chmod(0o700)
            def resolve(request,artifact):
                self.assertEqual(artifact,'campplus')
                return dict(id=artifact,path=str(model),revision='fictional',sha256='0'*64)
            request=protocol.normalize_request(dict(operation='probe',probe_scope='embedding',embedding_model='campplus',device='cuda',config=dict(speaker_pipeline='community1')))
            with patch.object(worker,'helper_executable',return_value=str(helper)),patch.object(worker,'resolve',side_effect=resolve),patch.object(worker,'release_models'),patch.object(worker,'transcribe',side_effect=AssertionError('No ASR')),patch.object(worker,'diarize',side_effect=AssertionError('No segmentation')):
                entry,metrics=worker.probe_embedding(request,True)
            self.assertTrue(entry['verified']);self.assertEqual(entry['device'],'cpu')
            self.assertIn('host_processes',metrics)

    def test_probe_advertises_additive_speech_operation_without_loading_models(self):
        request=protocol.normalize_request(dict(operation='probe',probe_scope='runtime',device='auto'))
        with patch.object(worker,'transcribe',side_effect=AssertionError('No model inference in runtime probe')),patch.object(worker,'diarize',side_effect=AssertionError('No speaker model inference')),patch.object(worker,'native_speakers',side_effect=AssertionError('No native helper')):
            result=worker.probe(request)
        self.assertEqual(result['protocol_version'],1)
        self.assertEqual(result['operations'],sorted(protocol.OPERATIONS))
        self.assertIn('speech_recognition',result['operations'])
        self.assertEqual(result['engines'],[])

    def run_fixture(self,operation,transcribe,diarize,extra=None):
        with tempfile.TemporaryDirectory() as directory:
            source=Path(directory)/'synthetic.wav'
            with wave.open(str(source),'wb')as wav:
                wav.setparams((1,2,16000,0,'NONE','not compressed'));wav.writeframes(bytes(16000*2*4))
            request=dict(version=1,operation=operation,audio_path=str(source),speech_device='cpu',speaker_device='cpu',
                config=dict(speech_engine='faster_whisper',speech_model='large-v3-turbo-int8',speaker_pipeline='community1',identification='same'))
            request.update(extra or {})
            def normalize(request,source,destination): shutil.copyfile(source,destination)
            record=dict(id='fictional-artifact',path=str(source),revision='fictional-revision',sha256='0'*64)
            with patch.object(worker,'resolve',return_value=record),patch.object(worker,'ffmpeg_executable',return_value='fictional-ffmpeg'),patch.object(worker,'convert_audio',side_effect=normalize),patch.object(worker,'transcribe',side_effect=transcribe)as recognition,patch.object(worker,'diarize',side_effect=diarize),patch.object(worker,'native_speakers',side_effect=AssertionError('Native helper must not run')),patch.object(worker,'runtime_provenance',return_value={}),patch.object(worker,'versions',return_value={}),patch.dict(os.environ,{},clear=False):
                result=worker.run(request)
                return result,recognition.call_count

    def test_asr_only_requires_no_speaker_artifacts_and_never_rebuilds_text(self):
        def recognition(request,result):
            worker.append_recognition_segment(result,0.,3.,' Fictional complete phrase',[
                dict(start=0.,end=1.,text=' Fictional'),dict(start=1.,end=1.,text=' complete'),dict(start=1.,end=3.,text=' phrase')])
        with patch.object(worker,'annotate',side_effect=AssertionError('ASR-only does not annotate')):
            result,calls=self.run_fixture('speech_recognition',recognition,AssertionError('No diarization'))
        self.assertEqual(calls,1);self.assertEqual(result['settings']['worker_operation'],'speech_recognition')
        self.assertEqual(result['segments'][0]['text'],' Fictional complete phrase')
        self.assertEqual(result['turns'],[]);self.assertEqual(result['embeddings'],[])
        self.assertEqual(result['provenance']['word_alignment']['unaligned_words'],1)

    def test_native_word_handoff_splits_speakers_and_retains_untimed_chunks(self):
        def diarization(request,result):
            result['turns']=[dict(start=0.,end=2.,speaker='A'),dict(start=1.,end=1.5,speaker='B')]
        result,calls=self.run_fixture('speaker_analysis',AssertionError('No repeat ASR'),diarization,
            dict(segments=[dict(start=0.,end=2.,text=' Hello yes again'),dict(start=2.,end=4.,text=' Keep this')],
                 words=[dict(start=0.,end=.5,text=' Hello'),dict(start=1.,end=1.5,text=' yes'),dict(start=1.6,end=2.,text=' again')]))
        self.assertEqual(calls,0)
        self.assertEqual([s['speaker'] for s in result['segments']],['A','B','A',None])
        self.assertEqual(''.join(s['text'] for s in result['segments']),' Hello yes again Keep this')
        self.assertEqual(result['segments'][1]['overlapping_speakers'],['A','B'])

    def test_speaker_only_retains_upstream_native_provenance_and_timing(self):
        def diarization(request,result): result['turns']=[dict(start=0.,end=4.,speaker='A')]
        result,calls=self.run_fixture('speaker_analysis',AssertionError('ASR must not rerun'),diarization,
            dict(segments=[dict(start=0.,end=4.,text='Fictional transcript')],upstream_provenance=dict(native_engine='fixture'),
                 upstream_recognition_device='vulkan',upstream_compute_type='backend-dependent',upstream_recognition_seconds=1.25))
        self.assertEqual(calls,0);self.assertEqual(result['recognition_device'],'vulkan')
        self.assertEqual(result['recognition_seconds'],1.25);self.assertGreaterEqual(result['elapsed_seconds'],1.25)
        self.assertEqual(result['provenance']['upstream'],dict(native_engine='fixture'))


FAKE_WORKER = '''import argparse,json,pathlib,subprocess,sys,time,os
p=argparse.ArgumentParser();p.add_argument('--request');p.add_argument('--response');a=p.parse_args()
r=json.load(open(a.request));folder=pathlib.Path(a.response).parent
if 'test_failure' in r:
 pathlib.Path(a.response+'.failure.json').write_text(json.dumps(r['test_failure']))
 print('fictional-secret https://example.invalid/private?token=fictional-secret',file=sys.stderr)
 sys.exit(1)
if r.get('test_mode')=='wait':
 child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(100)'])
 (folder/'test-child-pid').write_text(str(child.pid))
 print(json.dumps({'stage':'waiting','percent':1,'message':'Synthetic child running'}),flush=True)
 time.sleep(100)
if r.get('test_mode')=='invalid':
 pathlib.Path(a.response).write_text('{}');sys.exit(0)
result=dict(version=1,duration=10.,language=None,segments=[],words=[],turns=[],overlaps=[],embeddings=[],
 request_id=r['request_id'],audio_sha256=r.get('audio_sha256'))
pathlib.Path(a.response).write_text(json.dumps(result))
'''


class SupervisorTests(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory()
        self.root=supervisor.private_root(Path(self.temporary.name)/'runtime')
        self.program=Path(self.temporary.name)/'program';self.program.mkdir()
        for name in ('supervisor.py','protocol.py'):
            shutil.copyfile(Path(__file__).with_name(name),self.program/name)
        (self.program/'worker.py').write_text(FAKE_WORKER)
        self.payload=b'fictional audio bytes'
        self.request=dict(version=1,operation='speaker_analysis',request_id='fictional-job-123',
            audio_sha256=hashlib.sha256(self.payload).hexdigest())

    def tearDown(self):
        for folder in (self.root/'jobs').glob('*') if (self.root/'jobs').exists() else []:
            try:
                supervisor.control(self.root,dict(action='cancel',request_id=folder.name))
                self.wait(folder.name)
            except (ValueError,FileNotFoundError): pass
        self.temporary.cleanup()

    def stage(self):
        supervisor.create(self.root,self.request)
        with patch.object(sys,'stdin',types.SimpleNamespace(buffer=io.BytesIO(self.payload))):
            supervisor.receive(self.root,self.request['request_id'],self.request['audio_sha256'])

    def submit(self):
        with patch.object(supervisor,'__file__',str(self.program/'supervisor.py')):
            return supervisor.submit(self.root,self.request['request_id'],sys.executable)

    def wait(self,identity):
        end=time.monotonic()+8
        while time.monotonic()<end:
            state=supervisor.control(self.root,dict(action='status',request_id=identity))
            if state['state'] in supervisor.TERMINAL: return state
            time.sleep(.05)
        self.fail('Detached synthetic job did not terminate')

    def test_disconnect_reconnect_and_retry_return_same_terminal_result(self):
        self.stage();state=self.submit()
        self.assertEqual(state['state'],'queued')
        state=self.wait(self.request['request_id']);self.assertEqual(state['state'],'succeeded')
        self.assertNotIn('runtime_python',state)
        first=supervisor.control(self.root,dict(action='result',request_id=self.request['request_id']))
        self.assertEqual(supervisor.create(self.root,self.request)['state'],'succeeded')
        self.assertEqual(self.submit()['state'],'succeeded')
        self.assertEqual(first,supervisor.control(self.root,dict(action='result',request_id=self.request['request_id'])))
        folder=supervisor.owned_job(self.root,self.request['request_id'])
        self.assertFalse((folder/'input.audio').exists());self.assertFalse((folder/'work').exists())
        supervisor.control(self.root,dict(action='cleanup',request_id=self.request['request_id']))
        self.assertFalse(folder.exists())

    def test_terminal_result_hash_rejects_modified_result_bytes(self):
        self.stage();self.submit();self.wait(self.request['request_id'])
        folder=supervisor.owned_job(self.root,self.request['request_id'])
        (folder/'result.json').write_text('{"modified":true}')
        with self.assertRaises(ValueError):
            supervisor.control(self.root,dict(action='result',request_id=self.request['request_id']))

    def test_first_transfer_measurements_are_idempotent_and_do_not_change_result_bytes(self):
        self.stage()
        first=supervisor.control(self.root,dict(action='record_transfer',request_id=self.request['request_id'],kind='upload',seconds=.25))
        second=supervisor.control(self.root,dict(action='record_transfer',request_id=self.request['request_id'],kind='upload',seconds=.5))
        self.assertEqual(first['transport'],second['transport'])
        with self.assertRaises(ValueError):
            supervisor.control(self.root,dict(action='record_transfer',request_id=self.request['request_id'],kind='result_download',seconds=.1))
        self.submit();self.wait(self.request['request_id'])
        folder=supervisor.owned_job(self.root,self.request['request_id']);before=(folder/'result.json').read_bytes()
        for seconds in (.1,.2):
            measured=supervisor.control(self.root,dict(action='record_transfer',request_id=self.request['request_id'],kind='result_download',seconds=seconds))
            self.assertEqual(measured['transport']['result_download_wall_seconds'],.1)
        self.assertEqual((folder/'result.json').read_bytes(),before)
        self.assertEqual(measured['transport']['upload_wall_seconds'],.25)
        for seconds in (-1,float('nan'),True):
            with self.assertRaises(ValueError): supervisor.record_transfer(folder,'upload',seconds)

    def test_retry_rejects_new_bytes_or_configuration(self):
        self.stage()
        with self.assertRaises(ValueError): supervisor.create(self.root,dict(self.request,device='different'))
        with patch.object(sys,'stdin',types.SimpleNamespace(buffer=io.BytesIO(b'incorrect'))):
            with self.assertRaises(ValueError): supervisor.receive(self.root,self.request['request_id'],self.request['audio_sha256'])
        self.assertFalse((supervisor.owned_job(self.root,self.request['request_id'])/'input.partial').exists())

    def test_interrupted_upload_can_retry_same_owned_request_without_restarting_inference(self):
        supervisor.create(self.root,self.request)
        folder=supervisor.owned_job(self.root,self.request['request_id'])
        with patch.object(sys,'stdin',types.SimpleNamespace(buffer=io.BytesIO(self.payload[:5]))):
            with self.assertRaises(ValueError): supervisor.receive(self.root,self.request['request_id'],self.request['audio_sha256'])
        self.assertEqual(supervisor.control(self.root,dict(action='status',request_id=self.request['request_id']))['state'],'created')
        self.assertFalse((folder/'input.partial').exists());self.assertFalse((folder/'input.audio').exists())
        with patch.object(sys,'stdin',types.SimpleNamespace(buffer=io.BytesIO(self.payload))):
            supervisor.receive(self.root,self.request['request_id'],self.request['audio_sha256'])
        self.submit();self.assertEqual(self.wait(self.request['request_id'])['state'],'succeeded')
        self.assertEqual(self.submit()['state'],'succeeded')

    def test_cancel_after_upload_before_submit_never_starts_worker(self):
        self.stage();folder=supervisor.owned_job(self.root,self.request['request_id'])
        supervisor.control(self.root,dict(action='cancel',request_id=self.request['request_id']))
        self.assertEqual(self.submit()['state'],'cancelled')
        self.assertFalse((folder/'input.audio').exists());self.assertFalse((folder/'supervisor.log').exists())

    def test_cancel_kills_worker_and_descendants_and_removes_audio(self):
        self.request['test_mode']='wait';self.stage();self.submit()
        folder=supervisor.owned_job(self.root,self.request['request_id'])
        end=time.monotonic()+5
        while not (folder/'test-child-pid').exists() and time.monotonic()<end: time.sleep(.03)
        self.assertTrue((folder/'test-child-pid').exists())
        pid=int((folder/'test-child-pid').read_text())
        supervisor.control(self.root,dict(action='cancel',request_id=self.request['request_id']))
        self.assertEqual(self.wait(self.request['request_id'])['state'],'cancelled')
        time.sleep(.1)
        self.assertFalse((folder/'input.audio').exists());self.assertFalse((folder/'work').exists())
        if Path('/proc',str(pid),'stat').exists():
            self.assertEqual(Path('/proc',str(pid),'stat').read_text().split()[2],'Z')
        else:
            with self.assertRaises(ProcessLookupError): os.kill(pid,0)

    def test_invalid_model_output_never_becomes_a_success(self):
        self.request['test_mode']='invalid';self.stage();self.submit()
        self.assertEqual(self.wait(self.request['request_id'])['state'],'failed')
        with self.assertRaises(ValueError):
            supervisor.control(self.root,dict(action='result',request_id=self.request['request_id']))

    def test_failed_worker_promotes_only_allowlisted_public_error(self):
        self.request['test_failure']=dict(protocol_version=1,
            error_code='model_access_denied',error_artifact='community1')
        self.stage();self.submit();state=self.wait(self.request['request_id'])
        self.assertEqual(state['state'],'failed')
        self.assertEqual(state['error_code'],'model_access_denied')
        self.assertEqual(state['error_artifact'],'community1')
        self.assertNotIn('fictional-secret',json.dumps(state))
        self.assertNotIn('https://',json.dumps(state))
        folder=supervisor.owned_job(self.root,self.request['request_id'])
        self.assertIn('fictional-secret',(folder/'error.log').read_text())
        self.assertEqual((folder/'error.log').stat().st_mode & 0o077,0)
        with self.assertRaises(ValueError):
            supervisor.control(self.root,dict(action='result',request_id=self.request['request_id']))

    def test_unknown_or_malicious_worker_failure_stays_generic(self):
        valid=dict(protocol_version=1,error_code='model_access_denied',error_artifact='community1')
        for index,failure in enumerate((dict(valid,error_code='fictional-secret'),
                dict(valid,error_artifact='https://example.invalid/fictional-secret'),
                dict(valid,error='fictional-secret'),['fictional-secret'])):
            self.request.update(request_id='fictional-failure-'+str(index),test_failure=failure)
            self.stage();self.submit();state=self.wait(self.request['request_id'])
            self.assertEqual(state['state'],'failed')
            self.assertNotIn('error_code',state)
            self.assertNotIn('error_artifact',state)
            self.assertNotIn('fictional-secret',json.dumps(state))

    def test_target_lock_serializes_jobs_from_separate_callers(self):
        self.request['test_mode']='wait';self.stage();self.submit()
        first=self.request['request_id'];folder=supervisor.owned_job(self.root,first)
        end=time.monotonic()+5
        while not (folder/'test-child-pid').exists() and time.monotonic()<end: time.sleep(.03)
        self.request=dict(self.request,request_id='fictional-second-job',test_mode='complete')
        self.stage();self.submit();time.sleep(.15)
        self.assertEqual(supervisor.control(self.root,dict(action='status',request_id=self.request['request_id']))['state'],'queued')
        supervisor.control(self.root,dict(action='cancel',request_id=first))
        self.assertEqual(self.wait(first)['state'],'cancelled')
        self.assertEqual(self.wait(self.request['request_id'])['state'],'succeeded')

    def test_running_job_keeps_immutable_python_and_worker_when_current_changes(self):
        releases=[]
        for version in ('A','B'):
            release=self.root/('release-'+version)
            subprocess.run([sys.executable,'-m','venv','--without-pip',str(release)],check=True)
            setup_runtime.snapshot_worker(release)
            source=FAKE_WORKER.replace('result=dict(version=1',"time.sleep(.4)\nresult=dict(fixture_worker_version='"+version+"',version=1")
            (release/'worker/worker.py').write_text(source)
            protocol.atomic_json(release/'runtime.json',protocol.bundle_description(release/'worker'))
            releases.append(release)
        setup_runtime.activate(self.root,releases[0]);self.stage()
        with patch.object(supervisor,'__file__',str(self.program/'supervisor.py')):
            supervisor.submit(self.root,self.request['request_id'])
        setup_runtime.activate(self.root,releases[1])
        self.assertEqual(self.wait(self.request['request_id'])['state'],'succeeded')
        state=protocol.load_json(supervisor.owned_job(self.root,self.request['request_id'])/'status.json')
        self.assertEqual(Path(state['runtime_python']),releases[0]/'bin/python')
        self.assertEqual(Path(state['runtime_worker']),releases[0]/'worker/worker.py')
        self.assertTrue(state['worker_snapshot_verified'])
        result=supervisor.control(self.root,dict(action='result',request_id=self.request['request_id']))['result']
        self.assertEqual(result['fixture_worker_version'],'A')

    def test_owned_cleanup_rejects_symlinks_and_live_jobs(self):
        self.stage()
        with self.assertRaises(ValueError): supervisor.control(self.root,dict(action='cleanup',request_id=self.request['request_id']))
        foreign=Path(self.temporary.name)/'foreign';foreign.mkdir()
        (self.root/'jobs/fictional-other-job').symlink_to(foreign,target_is_directory=True)
        with self.assertRaises(ValueError): supervisor.owned_job(self.root,'fictional-other-job')


if __name__=='__main__': unittest.main()
