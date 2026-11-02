"""Bounded, explicitly scoped memory measurements. RSS is never GPU memory."""
import math
import os
from pathlib import Path
import platform
import re
import subprocess
import sys


def rss_bytes(value,system):
    if not isinstance(value,(int,float)) or not math.isfinite(value) or value<0: return None
    return int(value)*(1 if system=='Darwin' else 1024) if system in ('Linux','Darwin') else None


def process_memory_peaks():
    result=dict(unit='bytes',worker_peak_rss_bytes=None,completed_children_high_water_rss_bytes=None,
        worker_scope='This worker process lifetime, including imports and all stages',
        children_scope='Platform-defined RUSAGE_CHILDREN high-water for waited-for children, including FFmpeg/native helpers',
        method='getrusage.ru_maxrss',combination='Separate high-water marks; never a sum or simultaneous process-tree peak',
        is_gpu_memory_measurement=False,native_helper_peak_rss_bytes=None,
        overlap_note='RSS may overlap unified/device-backed mappings; do not add it to GPU allocator counters',
        native_helper_scope='Not isolated per helper; see completed_children_high_water_rss_bytes')
    try:
        import resource
        result['worker_peak_rss_bytes']=rss_bytes(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,platform.system())
        result['completed_children_high_water_rss_bytes']=rss_bytes(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,platform.system())
    except (ImportError,AttributeError,OSError,ValueError):
        result['method']='unavailable'
    return result


def linux_host_memory(text):
    values={match.group(1):int(match.group(2))*1024 for match in re.finditer(r'^(MemTotal|MemAvailable):\s+(\d+)\s+kB$',text,re.M)}
    return dict(total_bytes=values.get('MemTotal'),available_bytes=values.get('MemAvailable'),
        available_is_estimate=True,method='proc_meminfo_MemAvailable',scope='Host-global kernel available-memory estimate at probe time')


def darwin_host_memory(vm_stat,total):
    size=re.search(r'page size of (\d+) bytes',vm_stat)
    pages={match.group(1):int(match.group(2)) for match in re.finditer(r'^(Pages free|Pages inactive|Pages speculative):\s+(\d+)\.',vm_stat,re.M)}
    available=sum(pages.values())*int(size.group(1)) if size and len(pages)==3 else None
    return dict(total_bytes=int(total) if str(total).strip().isdigit() else None,available_bytes=available,
        available_is_estimate=True,method='vm_stat_free_inactive_speculative',scope='Host-global reclaimable-memory estimate at probe time')


def host_memory_probe():
    try:
        if platform.system()=='Linux': return linux_host_memory(Path('/proc/meminfo').read_text())
        if platform.system()=='Darwin':
            vm=subprocess.run(['/usr/bin/vm_stat'],capture_output=True,text=True,timeout=2,check=True).stdout
            total=subprocess.run(['/usr/sbin/sysctl','-n','hw.memsize'],capture_output=True,text=True,timeout=2,check=True).stdout.strip()
            return darwin_host_memory(vm,total)
    except (OSError,ValueError,subprocess.SubprocessError): pass
    return dict(total_bytes=None,available_bytes=None,available_is_estimate=None,method='unavailable',scope='Host-global snapshot')


def gpu_memory_snapshot(device):
    result=dict(available_bytes=None,total_bytes=None,scope='Device-global snapshot at probe time',
        domain='unified' if device in ('metal','mps') else 'cuda_device' if device=='cuda' else 'not_applicable',
        method='unavailable')
    torch=sys.modules.get('torch')
    try:
        if device=='cuda' and torch is not None:
            free,total=torch.cuda.mem_get_info()
            result.update(available_bytes=int(free),total_bytes=int(total),method='torch.cuda.mem_get_info')
        elif device in ('metal','mps'):
            result['reason']='Unified-memory API does not expose global free GPU memory; host availability is reported separately'
            if torch is not None and hasattr(torch.mps,'recommended_max_memory'):
                result['recommended_working_set_bytes']=int(torch.mps.recommended_max_memory())
    except (AttributeError,RuntimeError,OSError): pass
    return result


class GpuStagePeak:
    """Allocator-specific peak; it is not whole-device VRAM or a host RSS sum."""
    def __init__(self,device,allocator,stage):
        self.device,self.allocator,self.stage=device,allocator,stage
        self.module=sys.modules.get('torch' if allocator=='pytorch' else 'mlx.core')
        self.supported=False
        try:
            if allocator=='pytorch' and device=='cuda' and self.module is not None:
                self.module.cuda.reset_peak_memory_stats();self.supported=True
            elif allocator=='mlx' and device=='metal' and self.module is not None:
                reset=getattr(self.module,'reset_peak_memory',None)
                peak=getattr(self.module,'get_peak_memory',None)
                if callable(reset) and callable(peak): reset();self.supported=True
        except (AttributeError,RuntimeError): pass

    def finish(self):
        result=dict(stage=self.stage,device=self.device,allocator=self.allocator,unit='bytes',peak_allocated_bytes=None,
            peak_reserved_bytes=None,scope='Only this allocator in this worker during model load and this stage',
            domain='unified' if self.device in ('metal','mps') else 'cuda_device' if self.device=='cuda' else 'not_applicable',
            combination='Allocated is contained in reserved; do not add to host RSS or other allocator/process peaks',method='unavailable')
        try:
            if self.supported and self.allocator=='pytorch':
                result.update(peak_allocated_bytes=int(self.module.cuda.max_memory_allocated()),
                    peak_reserved_bytes=int(self.module.cuda.max_memory_reserved()),method='torch.cuda.allocator_peak')
            elif self.supported and self.allocator=='mlx':
                result.update(peak_allocated_bytes=int(self.module.get_peak_memory()),method='mlx.allocator_peak')
        except (AttributeError,RuntimeError): pass
        if result['method']=='unavailable': result['reason']='No supported peak API for this allocator/device; unknown is not zero'
        if self.device=='cpu': result.update(method='not_applicable',reason='CPU stage has no GPU allocator peak')
        return result


def record_gpu_stage(result,meter):
    result.setdefault('provenance',{}).setdefault('resources',{}).setdefault('gpu_allocator_stages',[]).append(meter.finish())
