"""Measurement units/scope and public summary privacy; no model dependencies."""
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import resources


class MemoryTests(unittest.TestCase):
    def test_rss_platform_units_are_explicit(self):
        self.assertEqual(resources.rss_bytes(12,'Linux'),12*1024)
        self.assertEqual(resources.rss_bytes(12,'Darwin'),12)
        for value in (-1,float('nan'),float('inf')): self.assertIsNone(resources.rss_bytes(value,'Linux'))
        self.assertIsNone(resources.rss_bytes(12,'unsupported'))

    def test_available_host_memory_uses_labeled_os_estimates(self):
        linux=resources.linux_host_memory('MemTotal: 1000 kB\nMemAvailable: 250 kB\n')
        self.assertEqual(linux['available_bytes'],250*1024)
        self.assertTrue(linux['available_is_estimate'])
        mac=resources.darwin_host_memory('Mach Virtual Memory Statistics: (page size of 16384 bytes)\nPages free: 10.\nPages inactive: 20.\nPages speculative: 3.\n','1048576')
        self.assertEqual(mac['available_bytes'],33*16384)
        self.assertTrue(mac['available_is_estimate'])

    def test_child_and_worker_highwater_are_separate_not_summed(self):
        import resource
        def usage(kind): return types.SimpleNamespace(ru_maxrss=100 if kind==resource.RUSAGE_SELF else 200)
        with patch.object(resource,'getrusage',side_effect=usage),patch.object(resources.platform,'system',return_value='Linux'):
            measured=resources.process_memory_peaks()
        self.assertEqual(measured['worker_peak_rss_bytes'],100*1024)
        self.assertEqual(measured['completed_children_high_water_rss_bytes'],200*1024)
        self.assertIsNone(measured['native_helper_peak_rss_bytes'])
        self.assertFalse(measured['is_gpu_memory_measurement'])
        self.assertIn('never a sum',measured['combination'])

    def test_gpu_allocator_peak_is_not_global_free_memory_or_host_rss(self):
        reset=[]
        fake=types.SimpleNamespace(cuda=types.SimpleNamespace(reset_peak_memory_stats=lambda:reset.append(True),
            max_memory_allocated=lambda:64,max_memory_reserved=lambda:128,mem_get_info=lambda:(1024,2048)),
            mps=types.SimpleNamespace(recommended_max_memory=lambda:4096))
        with patch.dict(sys.modules,{'torch':fake}):
            peak=resources.GpuStagePeak('cuda','pytorch','speaker_analysis').finish()
            device=resources.gpu_memory_snapshot('cuda')
            unified=resources.gpu_memory_snapshot('mps')
            unavailable=resources.GpuStagePeak('cuda','ctranslate2','speech_recognition').finish()
        self.assertEqual(reset,[True]);self.assertEqual(peak['peak_allocated_bytes'],64)
        self.assertEqual(peak['peak_reserved_bytes'],128)
        self.assertEqual(device['available_bytes'],1024)
        self.assertIsNone(unified['available_bytes']);self.assertEqual(unified['recommended_working_set_bytes'],4096)
        self.assertIsNone(unavailable['peak_allocated_bytes'])
        self.assertIn('unknown is not zero',unavailable['reason'])


class SummaryPrivacyTests(unittest.TestCase):
    def test_stdout_uses_allowlist_while_explicit_private_file_retains_settings(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=root/'fictional-private-meeting.json';output=root/'private-report.json';audio=root/'input.bin'
            audio.write_bytes(b'fictional audio fixture')
            secret='FICTIONAL_PRIVATE_SENTINEL'
            data=dict(duration=2.,elapsed_seconds=1.,segments=[],words=[],turns=[],overlaps=[],
                recognition_device='cpu',diarization_device=secret,settings=dict(config=dict(speech_engine='faster_whisper',
                speech_model='large-v3-turbo-int8',target_id=secret,ssh_host=secret),python=secret,token=secret))
            source.write_text(json.dumps(data))
            run=subprocess.run([sys.executable,str(Path(__file__).with_name('compare.py')),str(source),'--audio',str(audio),'--output',str(output)],capture_output=True,text=True,check=True)
            public=json.loads(run.stdout);private=json.loads(output.read_text())
            self.assertNotIn(secret,run.stdout);self.assertNotIn(source.stem,run.stdout)
            self.assertNotIn('audio_sha256',public)
            self.assertEqual(public['results'][0]['name'],'run_1')
            self.assertEqual(public['results'][0]['settings']['speech_engine'],'faster_whisper')
            self.assertEqual(private['results'][0]['settings']['token'],secret)


if __name__=='__main__': unittest.main()
