#!/usr/bin/env python3
"""Optional offline worker. JSON files carry data; stdout carries progress only.

No GUI, no dependency on noScribe, and no imports of ML libraries until a job
requests them. The native live engine never starts this process.
"""
import argparse
import gc
import base64
import platform
import shutil
from protocol import (normalize_request, validate_result, atomic_json, load_json, sha256,
                      clean_intervals, interval,bundle_sha256,OPERATIONS,
                      PublicWorkerError, failure_path, public_failure, VERSION)
from assets import resolve, prepare, public_record, registry
from qualification import mps_qualified
from resources import process_memory_peaks,host_memory_probe,gpu_memory_snapshot,GpuStagePeak,record_gpu_stage
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import wave


def progress(stage, percent, message):
    print(json.dumps(dict(stage=stage, percent=percent, message=message)), flush=True)


def overlaps(turns):
    """Return simultaneous *different* speakers, not adjacent speaker turns."""
    events = []
    for t in turns:
        if t['end'] > t['start']:
            events += [(t['start'], 1, t['speaker']), (t['end'], -1, t['speaker'])]
    active, result, previous = {}, [], None
    for at, delta, speaker in sorted(events):
        names = sorted(k for k, n in active.items() if n > 0)
        if previous is not None and at > previous and len(names) > 1:
            if result and result[-1]['end'] == previous and result[-1]['speakers'] == names:
                result[-1]['end'] = at
            else:
                result.append(dict(start=previous, end=at, speakers=names))
        active[speaker] = active.get(speaker, 0) + delta
        previous = at
    return result


def speaker_for(start, end, turns):
    candidates = []
    for t in turns:
        intersection = max(0., min(end, t['end']) - max(start, t['start']))
        if intersection:
            coverage = intersection / max(end - start, .001)
            # noScribe prefers a short interjection covering >=80% of a word.
            candidates.append(((coverage >= .8, -(t['end'] - t['start']) if coverage >= .8 else coverage), t['speaker']))
    return max(candidates)[1] if candidates else None


def annotate(result):
    turns = result['turns']
    result['overlaps'] = overlaps(turns)
    if result['words'] and any(s.get('_alignment_incomplete') for s in result['segments']):
        # Some aligners return zero-width words. Preserve their recognized text
        # using the actual parent segment's timing; never delete that text or
        # manufacture word boundaries to make the alignment appear complete.
        combined=[]
        for i,segment in enumerate(result['segments']):
            words=[w for w in result['words'] if w.get('_source_segment')==i]
            partial=dict(turns=turns,words=[] if segment.get('_alignment_incomplete') else words,segments=[segment])
            annotate(partial)
            combined.extend(partial['segments'])
        for word in result['words']:
            word['speaker']=speaker_for(word['start'],word['end'],turns)
            word['overlapping_speakers']=sorted({s for o in result['overlaps']
                if min(o['end'],word['end'])>max(o['start'],word['start']) for s in o['speakers']})
            word.pop('_source_segment',None)
        for segment in combined: segment.pop('_alignment_incomplete',None)
        result['segments']=combined
        return result
    if result['words']:
        segments = []
        previous_source = None
        for word in result['words']:
            source = word.get('_source_segment')
            word['speaker'] = speaker_for(word['start'], word['end'], turns)
            word['overlapping_speakers'] = sorted({s for o in result['overlaps']
                if min(o['end'], word['end']) > max(o['start'], word['start']) for s in o['speakers']})
            # Split at speaker changes, overlap boundaries, long pauses and
            # readable paragraph lengths. Timings remain actual word timings.
            if (not segments or (source is not None and source != previous_source)
                    or segments[-1]['speaker'] != word['speaker']
                    or segments[-1]['overlapping_speakers'] != word['overlapping_speakers']
                    or word['start'] - segments[-1]['end'] > 1.
                    or word['end'] - segments[-1]['start'] > 15.):
                segments.append(dict(start=word['start'], end=word['end'], text=word['text'],
                    speaker=word['speaker'], overlapping_speakers=word['overlapping_speakers']))
            else:
                segments[-1]['end'] = word['end']
                segments[-1]['text'] += word['text']
            previous_source = source
        result['segments'] = segments
    else:
        for segment in result['segments']:
            segment['speaker'] = speaker_for(segment['start'], segment['end'], turns)
            segment['overlapping_speakers'] = sorted({s for o in result['overlaps']
                if min(o['end'], segment['end']) > max(o['start'], segment['start']) for s in o['speakers']})
    for word in result.get('words',[]): word.pop('_source_segment',None)
    for segment in result.get('segments',[]): segment.pop('_alignment_incomplete',None)
    return result


def clean_seconds(speaker, turns, simultaneous):
    # Union intervals before summing so repeated same-speaker turns do not
    # inflate the evidence used for recognizing a voice across meetings.
    spans = sorted((t['start'], t['end']) for t in turns if t['speaker'] == speaker)
    merged = []
    for start, end in spans:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return sum(end - start - sum(max(0., min(end, o['end']) - max(start, o['start']))
               for o in simultaneous if speaker in o['speakers']) for start, end in merged)


def ffmpeg_executable(request):
    selected = request.get('ffmpeg')
    if selected:
        return selected
    found = shutil.which('ffmpeg')
    if found:
        return found
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        raise ValueError('FFmpeg is unavailable; install/update this isolated worker') from None


def convert_audio(request, source, destination):
    # One pipe carries FFmpeg metadata and machine-readable progress, avoiding
    # stderr backpressure. Duration is read from the source, never guessed.
    command=[ffmpeg_executable(request), '-nostdin', '-nostats', '-progress', 'pipe:1',
        '-y', '-i', str(source), '-vn', '-ac', '1', '-ar', '16000', '-c:a', 'pcm_s16le', str(destination)]
    duration=None
    last=-1
    tail=[]
    process=subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        for line in process.stdout:
            line=line.strip()
            tail=(tail+[line])[-20:]
            if line.startswith('Duration:'):
                value=line.split('Duration:',1)[1].split(',',1)[0].strip()
                try:
                    h,m,s=map(float,value.split(':'))
                    duration=h*3600+m*60+s
                except ValueError:
                    duration=None
            if line.startswith('out_time_us=') and duration and math.isfinite(duration) and duration>0:
                try: percent=min(99,max(0,int(float(line.split('=',1)[1])/1e6/duration*100)))
                except ValueError: continue
                if percent>last:
                    last=percent
                    progress('worker_decoding_progress',percent,'Converting audio to 16 kHz mono on the target')
        if process.wait():
            raise ValueError('Audio conversion failed: '+'\n'.join(tail))
        progress('worker_decoding_progress',100,'Converting audio to 16 kHz mono on the target')
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        process.stdout.close()


def read_pcm(path):
    import numpy as np
    with wave.open(str(path), 'rb') as wav:
        if (wav.getframerate(), wav.getnchannels(), wav.getsampwidth()) != (16000, 1, 2):
            raise ValueError('Expected normalized 16 kHz mono PCM16 audio')
        return np.frombuffer(wav.readframes(wav.getnframes()), dtype='<i2').astype(np.float32) / 32768.


def stage_device(request, stage):
    device = request[stage+'_device']
    allowed = ('cpu', 'cuda', 'vulkan', 'metal') if stage == 'speech' else ('cpu', 'cuda')
    if (stage=='speaker' and device=='mps' and (request.get('experimental_speaker_mps') is True or mps_qualified(request))
            and platform.system()=='Darwin' and platform.machine()=='arm64'):
        allowed=allowed+('mps',)
    if device not in allowed:
        raise ValueError('Unsupported explicit '+stage+' device; no automatic fallback is permitted')
    return device


def release_models():
    gc.collect()
    if 'torch' in sys.modules and sys.modules['torch'].cuda.is_available():
        sys.modules['torch'].cuda.empty_cache()
    if 'torch' in sys.modules and sys.modules['torch'].backends.mps.is_available():
        sys.modules['torch'].mps.empty_cache()
    if 'mlx.core' in sys.modules:
        sys.modules['mlx.core'].clear_cache()


def require_versions(packages):
    for package,expected in packages.items():
        try: installed=importlib.metadata.version(package).split('+')[0]
        except importlib.metadata.PackageNotFoundError:
            raise ValueError('Install the pinned worker package: '+package) from None
        if installed!=expected:
            raise ValueError('The selected worker package version is unqualified: '+package)


def validate_community_device(request):
    """Verify requested execution and reject implicit PyTorch CPU fallback."""
    import torch
    device=stage_device(request,'speaker')
    if device=='cuda':
        if not torch.cuda.is_available():
            raise ValueError('Selected Community-1 CUDA device is unavailable')
        torch.ones(1,device='cuda').square().cpu()
    if device=='mps':
        if not torch.backends.mps.is_available() or os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK')=='1':
            raise ValueError('MPS requires a working Apple GPU with CPU fallback disabled')
        torch.ones(1,device='mps').square().cpu()
    return device


def load_community(request,stage='speaker_analysis'):
    require_versions({'pyannote.audio':'4.0.4','torch':'2.8.0','torchaudio':'2.8.0'})
    import torch
    from pyannote.audio import Pipeline
    device=validate_community_device(request)
    torch.set_num_threads(min(8, os.cpu_count() or 4))
    meter=GpuStagePeak(device,'pytorch',stage)
    record = resolve(request, 'community1')
    pipeline = Pipeline.from_pretrained(record['path'])
    pipeline.embedding_exclude_overlap = True
    pipeline.to(torch.device(device))
    return pipeline, record, device,meter



def load_community_embedding(request):
    """Load only the package embedding adapter, retaining its exact reference.

    SpeakerDiarization.get_embeddings needs the adapter, Audio cropper and batch
    size in evaluation mode; it does not need segmentation or PLDA. Reuse that
    package method directly rather than copying preprocessing or its math.
    """
    require_versions({'pyannote.audio':'4.0.4','torch':'2.8.0','torchaudio':'2.8.0'})
    import torch
    from types import SimpleNamespace,MethodType
    from pyannote.audio import Audio
    from pyannote.audio.pipelines.speaker_diarization import SpeakerDiarization
    from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
    device=validate_community_device(request)
    torch.set_num_threads(min(8,os.cpu_count()or 4))
    meter=GpuStagePeak(device,'pytorch','embedding_extraction')
    record=resolve(request,'community1')
    embedding=PretrainedSpeakerEmbedding(dict(checkpoint=record['path'],subfolder='embedding'),device=torch.device(device))
    pipeline=SimpleNamespace(training=False,_embedding=embedding,
        _audio=Audio(sample_rate=embedding.sample_rate,mono='downmix'),embedding_batch_size=1)
    pipeline.get_embeddings=MethodType(SpeakerDiarization.get_embeddings,pipeline)
    return pipeline,record,device,meter


def community_space(pipeline, record):
    # Hash the actual bundled embedding weights, rather than trusting dimension
    # or a broad WeSpeaker label. Keep local paths out of result provenance.
    root = Path(record['path']) / 'embedding'
    weights = sorted(p for p in root.rglob('*') if p.is_file() and p.suffix in ('.bin', '.safetensors', '.ckpt', '.onnx'))
    if not weights:
        raise ValueError('Cannot identify the bundled Community-1 embedding checkpoint')
    digest = hashlib.sha256()
    for path in weights:
        digest.update(bytes.fromhex(sha256(path)))
    revision = digest.hexdigest()
    return dict(id='wespeaker:'+revision+':pcm16k-mono:l2:clean-three10s-mean-v1',
        model='wespeaker', revision=revision,
        preprocessing='pcm16k-mono:package-reference-mask-v1', normalization='l2',
        aggregation='longest-three-clean-10s-duration-mean-v1', extraction_version='pyannote-4.0.4-clean-v1',
        dimension=int(pipeline._embedding.dimension))


def validate_clean_sources(request, duration):
    spans = request.get('source_intervals', [])
    if not spans:
        raise ValueError('Embedding extraction requires explicit clean source intervals')
    for span in spans:
        if (not interval(span, duration) or span.get('channel', 0) != 0
                or not isinstance(span.get('speaker'), str) or not span['speaker']
                or span['end']-span['start'] < 3.):
            raise ValueError('Extraction intervals need a speaker, mono channel, and at least 3 clean seconds')
    for i, a in enumerate(spans):
        for b in spans[i+1:]:
            if max(a['start'], b['start']) < min(a['end'], b['end']):
                raise ValueError('Embedding source intervals must be disjoint and free of overlap')
    for span in spans:
        if any(max(span['start'], o['start']) < min(span['end'], o['end']) for o in request.get('overlaps', [])):
            raise ValueError('Embedding source interval intersects known simultaneous speech')
    return spans


def extraction_windows(spans):
    """Bound embedding memory without inventing or extending clean evidence."""
    selected=[]
    for speaker in sorted({s['speaker'] for s in spans}):
        candidates=sorted((s for s in spans if s['speaker']==speaker),
                          key=lambda s:(-(s['end']-s['start']),s['start']))[:3]
        for span in candidates:
            start=span['start']+max(0.,(span['end']-span['start']-10.)/2)
            selected.append(dict(span,start=start,end=min(span['end'],start+10.)))
    return sorted(selected,key=lambda s:(s['start'],s['speaker']))


def community_extract(request, result, pipeline=None, record=None):
    """Use the package's own embedding adapter and verify reference equivalence.

    One exact clean interval is one chunk. Full-one masks use precisely the
    package get_embeddings preprocessing; the first interval is checked against
    that reference path on every extraction. Pooled diarization centroids use a
    different policy and are deliberately never reused as these embeddings.
    """
    import numpy as np
    import torch
    from pyannote.core import SlidingWindow, SlidingWindowFeature
    own = pipeline is None
    if own:
        pipeline, record, _,meter = load_community_embedding(request)
    spans = extraction_windows(validate_clean_sources(request, result['duration']))
    space = community_space(pipeline, record)
    audio = read_pcm(request['audio_path'])
    grouped = {}
    progress('embedding_progress', 0, 'Extracting clean voice samples for speaker identification')
    for i, span in enumerate(spans):
        samples = audio[round(span['start']*16000):round(span['end']*16000)]
        waveform = torch.from_numpy(samples.copy())[None, None, :]
        # 100 ms masks suffice because every frame is clean, including its edges.
        masks = torch.ones((1, max(2, round(len(samples)/1600))), dtype=torch.float32)
        vector = np.asarray(pipeline._embedding(waveform, masks=masks))[0]
        if i == 0:
            duration = len(samples)/16000.
            feature = SlidingWindowFeature(np.ones((1, masks.shape[1], 1), dtype=np.float32),
                SlidingWindow(start=0., duration=duration, step=duration))
            reference = pipeline.get_embeddings({'waveform': waveform[0], 'sample_rate':16000},
                feature, exclude_overlap=True)[0, 0]
            if not np.allclose(vector, reference, rtol=2e-4, atol=2e-5, equal_nan=False):
                raise ValueError('Standalone WeSpeaker extraction failed package-reference validation')
        if vector.shape != (space['dimension'],) or not np.isfinite(vector).all() or np.linalg.norm(vector) < 1e-6:
            raise ValueError('WeSpeaker returned an invalid embedding')
        vector = vector / np.linalg.norm(vector)
        grouped.setdefault(span['speaker'], []).append((span, vector))
        progress('embedding_progress', int(100*(i+1)/len(spans)), 'Extracting clean voice samples for speaker identification')
    for speaker, samples in grouped.items():
        total = sum(s['end']-s['start'] for s, _ in samples)
        vector = sum(v*(s['end']-s['start']) for s,v in samples)/total
        vector /= np.linalg.norm(vector)
        result['embeddings'].append(dict(speaker=speaker, model_id=space['id'], embedding_space=space,
            vector=[float(x) for x in vector], clean_seconds=total, sample_count=len(samples),
            source_intervals=[s for s,_ in samples]))
    result.setdefault('provenance', {})['embedding_reference_verified'] = True
    if own:
        record_gpu_stage(result,meter)
        del pipeline
        release_models()


def community_progress(stage, total=None, completed=None):
    labels = {'segmentation': 'Detecting speech and simultaneous speakers',
              'embeddings': 'Extracting speaker embeddings from audio windows',
              'clustering': 'Grouping windows by speaker',
              'discrete_diarization': 'Building the speaker timeline'}
    message = 'Community-1: ' + labels.get(stage, stage.replace('_', ' '))
    if total and completed is not None:
        progress('diarizing_progress', min(100, max(0, int(100*completed/total))), message)
    else:
        # pyannote's uncounted hook is a stage-completion notification.
        # In particular, clustering starts after this embeddings notification;
        # continuing to display "extracting embeddings" here is misleading.
        following = {
            'segmentation': 'Counting simultaneous speakers',
            'speaker_counting': 'Preparing speaker embedding batches',
            'embeddings': 'Grouping speech windows by speaker',
            'discrete_diarization': 'Finalizing speaker turns and overlaps',
        }
        progress('diarizing', 0, 'Community-1: ' + following.get(stage, labels.get(stage, stage.replace('_', ' '))))


def clip_speaker_turns(turns, duration):
    """Keep only the part of each speaker turn supported by source audio."""
    clipped = []
    for turn in turns:
        if not all(math.isfinite(turn[key]) for key in ('start', 'end')):
            raise ValueError('Non-finite speaker turn timestamp')
        start = max(0., turn['start'])
        end = min(duration, turn['end'])
        # Segmentation windows can extend into padding past the recording.
        # Check after clipping: an originally valid turn can now be empty.
        if end > start:
            clipped.append(dict(turn, start=start, end=end))
    return clipped


def diarize(request, result):
    import torch
    progress('diarizing', 10, 'Loading Community-1 speaker analysis')
    pipeline, record, device,meter = load_community(request)
    waveform = torch.from_numpy(read_pcm(request['audio_path']))[None, :]
    kwargs = {'num_speakers':request['num_speakers']} if request.get('num_speakers') else {}
    def hook(stage, artifact, file=None, total=None, completed=None):
        community_progress(stage, total, completed)
    output = pipeline({'waveform':waveform, 'sample_rate':16000}, hook=hook, **kwargs)
    labels = list(output.speaker_diarization.labels())
    names = {label:'Speaker '+str(i+1) for i,label in enumerate(labels)}
    # Keep the regular overlap-capable timeline. The exclusive timeline is not
    # a substitute for the simultaneous-speech record.
    result['turns'] = clip_speaker_turns(
        (dict(start=float(turn.start), end=float(turn.end), speaker=names[speaker])
         for turn,_,speaker in output.speaker_diarization.itertracks(yield_label=True)),
        result['duration'])
    result['overlaps'] = overlaps(result['turns'])
    spans = clean_intervals(result['turns'], result['overlaps'])
    if spans:
        community_extract(dict(request, source_intervals=spans), result, pipeline, record)
    result['diarization_device'] = device
    result.setdefault('provenance', {})['speaker_artifact'] = public_record(record)
    record_gpu_stage(result,meter)
    del pipeline, waveform, output
    release_models()


def append_recognition_segment(result,start,end,text,raw_words):
    end=min(result['duration'],end)
    if end<=start:
        if text.strip(): raise ValueError('Recognized text has no valid source segment timing')
        return
    index=len(result['segments'])
    segment=dict(start=start,end=end,text=text,speaker=None,overlapping_speakers=[])
    result['segments'].append(segment)
    alignment=result.setdefault('provenance',{}).setdefault('word_alignment',
        dict(timed_words=0,unaligned_words=0,segment_fallback_count=0))
    for word in raw_words:
        word=dict(word,end=min(result['duration'],word['end']))
        if word['end']<=word['start']:
            segment['_alignment_incomplete']=True
            alignment['unaligned_words']+=1
        else:
            result['words'].append(dict(word,_source_segment=index))
            alignment['timed_words']+=1
    if segment.get('_alignment_incomplete'): alignment['segment_fallback_count']+=1


def transcribe(request, result):
    require_versions({'faster-whisper':'1.2.1','ctranslate2':'4.7.1'})
    from faster_whisper import WhisperModel
    import ctranslate2
    progress('transcribing', 45, 'Loading faster-whisper')
    device = stage_device(request, 'speech')
    if device not in ('cpu', 'cuda') or (device == 'cuda' and not ctranslate2.get_cuda_device_count()):
        raise ValueError('Selected faster-whisper device is unavailable')
    compact = request['config']['speech_model'] == 'large-v3-turbo-int8'
    compute = request.get('compute_type') or (('int8_float16' if device == 'cuda' else 'int8_float32') if compact else ('float16' if device == 'cuda' else 'float32'))
    if compute not in ctranslate2.get_supported_compute_types(device):
        raise ValueError('Requested execution precision is unsupported on this device')
    record = resolve(request, request['config']['speech_model'])
    meter=GpuStagePeak(device,'ctranslate2','speech_recognition')
    model = WhisperModel(record['path'], device=device, compute_type=compute, cpu_threads=min(8,os.cpu_count() or 4))
    actual = model.model.compute_type
    if actual != compute:
        raise ValueError('CTranslate2 changed the requested execution precision')
    language = request.get('language')
    translate = language == 'auto-translate'
    language = None if language in (None,'','auto','auto-translate') else language
    beam = request.get('beam_size',5)
    if not isinstance(beam,int) or not 1 <= beam <= 10:
        raise ValueError('beam_size must be between 1 and 10')
    segments,info = model.transcribe(read_pcm(request['audio_path']), language=language,
        task='translate' if translate else 'transcribe', beam_size=beam, word_timestamps=True,
        vad_filter=True, vad_parameters=dict(threshold=.5,min_silence_duration_ms=500,speech_pad_ms=50))
    result['language'] = info.language
    for segment in segments:
        if segment.end <= segment.start:
            continue
        append_recognition_segment(result,float(segment.start),float(segment.end),segment.text,[
            dict(start=float(w.start),end=float(w.end),text=w.word,probability=float(w.probability)) for w in segment.words or []])
        progress('transcribing',45+min(45,int(45*segment.end/max(1.,result['duration']))),'Transcribing audio')
    result.update(recognition_device=device,compute_type=actual,beam_size=beam)
    result.setdefault('provenance',{}).update(speech_artifact=public_record(record),decoder=dict(beam_size=beam,word_timestamps=True,vad_filter=True))
    record_gpu_stage(result,meter)
    del model
    release_models()


def mlx_transcribe(request, result):
    progress('transcribing', 0, 'Loading MLX Whisper and transcribing audio')
    if platform.system() != 'Darwin' or platform.machine() != 'arm64' or stage_device(request,'speech') != 'metal':
        raise ValueError('MLX Whisper requires the explicitly selected Apple GPU')
    require_versions({'mlx-whisper':'0.4.3','mlx':'0.29.3'})
    import mlx_whisper
    import mlx.core as mx
    if not mx.metal.is_available():
        raise ValueError('MLX could not initialize the Apple GPU')
    if request.get('beam_size',1) != 1:
        raise ValueError('The qualified MLX adapter uses greedy decoding (beam size 1)')
    record = resolve(request,request['config']['speech_model'])
    meter=GpuStagePeak('metal','mlx','speech_recognition')
    language = request.get('language')
    transcript = mlx_whisper.transcribe(read_pcm(request['audio_path']),path_or_hf_repo=record['path'],
        language=None if language in (None,'','auto','auto-translate') else language,
        task='translate' if language=='auto-translate' else 'transcribe',word_timestamps=True,temperature=0.,
        condition_on_previous_text=False)
    result['language'] = transcript.get('language')
    for segment in transcript['segments']:
        if segment['end'] <= segment['start']:
            continue
        append_recognition_segment(result,float(segment['start']),float(segment['end']),segment['text'],[
            dict(start=float(w['start']),end=float(w['end']),text=w['word'],probability=w.get('probability')) for w in segment.get('words',[])])
    result.update(recognition_device='metal',compute_type='float16',beam_size=1)
    result.setdefault('provenance',{}).update(speech_artifact=public_record(record),decoder=dict(beam_size=1,temperature=0.,
        word_timestamps=True,condition_on_previous_text=False))
    record_gpu_stage(result,meter)
    # mlx-whisper retains a process-global model holder; clear it explicitly
    # before a subsequent stage, as well as releasing the Metal cache.
    try:
        from mlx_whisper.transcribe import ModelHolder
        ModelHolder.model = None
        ModelHolder.model_path = None
    except ImportError:
        raise ValueError('Pinned MLX runtime model-release contract is unavailable') from None
    release_models()


def helper_executable(request, name):
    explicit = request.get('helpers',{}).get(name)
    root = Path(os.environ.get('PARLEY_WORKER_ROOT',Path(__file__).parent))
    candidates = [Path(explicit)] if explicit else [Path(sys.prefix)/'native'/name,
        root/'native'/name, root/'current/native'/name,Path(__file__).parent/'native'/name]
    for candidate in candidates:
        if candidate.is_file() and os.access(candidate,os.X_OK):
            return str(candidate.resolve())
    raise ValueError('Native helper requires explicit install/build: '+name)


def native_speakers(request,result,embedding_only=False):
    if request['speaker_device']!='cpu':
        raise ValueError('The qualified Sherpa package requires explicitly resolved CPU execution')
    records = {k:resolve(request,v) for k,v in [('segmentation','sherpa-segmentation'),('embedding','campplus')] if not embedding_only or k=='embedding'}
    models = {k:v['path'] for k,v in records.items()}
    helper=helper_executable(request,'parley_native_worker')
    helper_request = dict(request,version=1,native_models=models,
        operation='extract_embeddings' if embedding_only else 'speaker_analysis',
        duration=result['duration'],segments=result['segments'],turns=result['turns'],overlaps=result['overlaps'])
    with tempfile.TemporaryDirectory(prefix='native-speaker-',dir=request.get('work_dir')) as folder:
        source,response = Path(folder)/'request.json',Path(folder)/'response.json'
        atomic_json(source,helper_request)
        subprocess.run([helper,'--request',str(source),'--response',str(response)],check=True,stdout=sys.stderr)
        output=load_json(response)
    for key in ('turns','overlaps','embeddings'):
        if key=='embeddings':
            result[key].extend(output.get(key,[]))
        elif not embedding_only:
            result[key]=output.get(key,[])
    result['diarization_device']=request['speaker_device']
    result.setdefault('provenance',{})['native_speaker']=output.get('settings',{})
    result['provenance']['speaker_artifacts']={k:public_record(v) for k,v in records.items()}
    result['provenance']['speaker_helper_sha256']=sha256(helper)
    record_gpu_stage(result,GpuStagePeak('cpu','native_sherpa','embedding_extraction' if embedding_only else 'speaker_analysis'))


def whisper_cpp_transcribe(request,result):
    progress('transcribing', 0, 'Loading whisper.cpp and transcribing audio on the target')
    # Use Parley's shared whisper-core decoder through its existing stdio
    # protocol. No second implementation of Whisper parameters lives here.
    record=resolve(request,request['config']['speech_model'])
    helper=helper_executable(request,'whisper-helper-'+stage_device(request,'speech'))
    if request.get('language')=='auto-translate':
        raise ValueError('The native helper does not support the selected translation operation')
    process=subprocess.Popen([helper],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=sys.stderr,text=True)
    def rpc(message):
        process.stdin.write(json.dumps(message)+'\n');process.stdin.flush()
        line=process.stdout.readline()
        if not line:
            raise ValueError('Native speech helper closed its protocol stream')
        answer=json.loads(line)
        if answer.get('error') or answer.get('type')=='error':
            raise ValueError('Native speech helper failed its requested operation')
        return answer
    try:
        probe=rpc(dict(type='probe',model_path=record['path']))
        if (probe.get('backend')!=request['speech_device'] or probe.get('decode_ok') is not True
                or probe.get('device_verified') is not True or not probe.get('initialized_device')):
            raise ValueError('Native speech helper did not verify the requested backend')
        if rpc(dict(type='load_model',path=record['path'])).get('type')!='loaded':
            raise ValueError('Native speech helper could not load the selected model')
        samples=read_pcm(request['audio_path'])
        # Whole-recording transport avoids invented chunk joins or duplicate
        # decoding logic. Limit explicitly until streaming file IPC is added.
        if len(samples)>16000*7200:
            raise ValueError('Native worker recordings are currently limited to two hours')
        language=request.get('language')
        answer=rpc(dict(type='transcribe',samples_b64=base64.b64encode(samples.astype('<f4').tobytes()).decode(),
            language=None if language in (None,'','auto') else language,context_prompt=None,max_threads=min(8,os.cpu_count() or 4),greedy=False))
        for item in answer.get('segments',[]):
            start,end=item['start_ms']/1000.,min(result['duration'],item['end_ms']/1000.)
            if end>start:
                result['segments'].append(dict(start=start,end=end,text=item['text'],speaker=None,overlapping_speakers=[]))
        rpc(dict(type='unload'));rpc(dict(type='shutdown'))
        process.wait(timeout=10)
        if process.returncode:
            raise ValueError('Native speech helper exited unsuccessfully')
        result.update(recognition_device=probe['backend'],compute_type='backend-dependent')
        result.setdefault('provenance',{}).update(speech_artifact=public_record(record),
            native_probe=dict(backend=probe['backend'],decode_ok=True,device_verified=True,
                initialized_device=probe['initialized_device']),speech_helper_sha256=sha256(helper),
            decoder=dict(shared='whisper-core',greedy=False))
        record_gpu_stage(result,GpuStagePeak(probe['backend'],'whisper_cpp','speech_recognition'))
    finally:
        if process.poll() is None:
            process.kill();process.wait()


def versions():
    result={}
    for name in ('faster-whisper','ctranslate2','pyannote.audio','torch','torchaudio','mlx-whisper','mlx','numpy','imageio-ffmpeg'):
        try: result[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: result[name]=None
    return result


def runtime_provenance():
    manifest=Path(sys.prefix)/'runtime.json'
    installed=load_json(manifest) if manifest.is_file() else {}
    own_snapshot=Path(__file__).parent.resolve()==(Path(sys.prefix)/'worker').resolve()
    description=installed if own_snapshot and installed.get('worker_bundle_sha256') else None
    result=dict(protocol_version=1,worker_sha256=sha256(__file__),bundle_sha256=bundle_sha256(Path(__file__).parent,description),
        python_version=platform.python_version(),platform=dict(system=platform.system(),architecture=platform.machine()),
        packages=versions())
    lock=Path(sys.prefix)/'installed-requirements.txt'
    if lock.is_file(): result['dependency_lock_sha256']=sha256(lock)
    if manifest.is_file():
        result['runtime_id']=installed.get('runtime_id')
        result['installed_device']=installed.get('device')
        result['dependency_lock_sha256']=installed.get('requirements_sha256')
        result['worker_snapshot_verified']=installed.get('worker_bundle_sha256')==result['bundle_sha256']
        if not installed.get('worker_bundle_sha256'):
            result['worker_snapshot_note']='Legacy runtime has no saved worker-code rollback proof'
    return result


def probe_speech_devices(policy,engine,cuda_verified):
    if policy!='auto': return [policy]
    apple=platform.system()=='Darwin' and platform.machine()=='arm64'
    if engine=='mlx_whisper': return ['metal'] if apple else ['cpu']
    if engine=='faster_whisper': return ['cuda','cpu'] if cuda_verified else ['cpu']
    return ['metal','cpu'] if apple else (['cuda','vulkan','cpu'] if cuda_verified else ['vulkan','cpu'])


def probe_embedding(request,cuda_verified):
    selected=request.get('embedding_model')
    if selected not in ('campplus','wespeaker'): raise ValueError('Choose a supported identification model')
    policy=request.get('device',request.get('speech_device','cpu'))
    qualified=selected=='wespeaker' and policy in ('metal','auto') and mps_qualified(request)
    device='cpu' if selected=='campplus' else 'mps' if qualified else 'cuda' if policy=='cuda' or(policy=='auto'and cuda_verified)else'cpu'
    packages=versions()
    if selected=='campplus':
        try: helper_executable(request,'parley_native_worker');installed=True
        except ValueError: installed=False
    else: installed=packages.get('pyannote.audio')=='4.0.4' and all(str(packages.get(p,'')).split('+')[0]=='2.8.0' for p in ('torch','torchaudio'))
    entry=dict(id=selected,installed=installed,available=False,verified=False,device=device,numerical_probe_only=True)
    if not installed:
        entry['issue']='Install the selected embedding adapter in target setup'
        return entry,{}
    try:
        with tempfile.TemporaryDirectory(prefix='parley-embedding-probe-',dir=request.get('work_dir')) as folder:
            audio=Path(folder)/'probe.wav'
            # Four seconds accommodates native trimming while retaining >=3 s.
            # Deterministic synthetic samples are never saved as profile evidence.
            with wave.open(str(audio),'wb')as wav:
                wav.setparams((1,2,16000,0,'NONE','not compressed'))
                wav.writeframes(b''.join((((i*7919)%2001)-1000).to_bytes(2,'little',signed=True)for i in range(16000*4)))
            test=dict(request,operation='extract_embeddings',embedding_model=selected,audio_path=str(audio),speaker_device=device,
                source_intervals=[dict(speaker='Probe',start=0.,end=4.,channel=0)])
            output=empty_result(test,4.)
            if selected=='campplus': native_speakers(test,output,True)
            else: community_extract(test,output)
            validate_result(output,test)
            if not output['embeddings']: raise ValueError('The extractor returned no synthetic evidence')
            entry.update(available=True,verified=True,embedding_space=output['embeddings'][0].get('embedding_space'))
            return entry,dict(host_processes=process_memory_peaks(),**output.get('provenance',{}).get('resources',{}))
    except Exception as error:
        entry['issue']=type(error).__name__+': prepare the selected embedding artifact or verify its requested device'
        return entry,{}
    finally: release_models()


def probe(request):
    packages=versions()
    device=request['speech_device']
    result=dict(version=1,protocol_version=1,packages=packages,platform=dict(system=platform.system(),architecture=platform.machine()),
        engines=[],speaker_pipelines=[],artifacts=[],resolved=dict(speech_device=device,speaker_device=request['speaker_device']),
        devices=[],issues=[],sequential_loading=True)
    result['runtime']=runtime_provenance()
    result['operations']=sorted(OPERATIONS)
    result['probe_scopes']=['runtime','embedding','all']
    result['host_memory']=host_memory_probe()
    result['memory_available_bytes']=result['host_memory']['available_bytes']
    if request.get('probe_scope')=='runtime':
        return result
    try:
        ffmpeg=ffmpeg_executable(request)
        subprocess.run([ffmpeg,'-version'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,check=True)
        result['ffmpeg']=True
    except (ValueError,OSError,subprocess.CalledProcessError): result['ffmpeg']=False
    try:
        import torch
        for d in ('cpu','cuda'):
            if d=='cuda' and not torch.cuda.is_available(): continue
            try:
                torch.ones((2,2),device=d).matmul(torch.ones((2,2),device=d)).cpu()
                info=dict(id=d,verified=True)
                info['gpu_memory']=gpu_memory_snapshot(d)
                if d=='cuda':
                    free,total=torch.cuda.mem_get_info();info.update(free_memory_bytes=free,total_memory_bytes=total)
                result['devices'].append(info)
            except Exception: result['devices'].append(dict(id=d,verified=False))
    except ImportError: pass
    cuda_verified=any(d['id']=='cuda' and d['verified'] for d in result['devices'])
    if request.get('probe_scope')=='embedding':
        entry,measurements=probe_embedding(request,cuda_verified)
        result['embedding_models']=[entry]
        result['resources']=measurements
        result['resolved']['embedding_device']=entry['device']
        return result
    for artifact,record in registry(request).items():
        try: resolve(request,artifact);result['artifacts'].append(dict(id=artifact,ready=True,revision=record['revision'],sha256=record['sha256']))
        except ValueError: result['artifacts'].append(dict(id=artifact,ready=False))
    # Readiness means real execution using installed model weights, never a
    # successful Python import or a machine-name heuristic.
    with tempfile.TemporaryDirectory(prefix='parley-probe-',dir=request.get('work_dir')) as folder:
        audio=Path(folder)/'probe.wav'
        with wave.open(str(audio),'wb') as wav:
            wav.setparams((1,2,16000,0,'NONE','not compressed'));wav.writeframes(bytes(16000*2*3))
        for engine,model in [('whisper_cpp','large-v3-turbo-q5_0'),('faster_whisper','large-v3-turbo-fp16'),('mlx_whisper','large-v3-turbo-mlx-fp16')]:
            candidates=[a['id'] for a in result['artifacts'] if a['ready'] and a['id'] in __import__('assets').SPEECH and (
                ('mlx' in a['id']) if engine=='mlx_whisper' else ('q5' in a['id'] or a['id']=='large-v3-turbo') if engine=='whisper_cpp' else a['id'] in ('large-v3-turbo-fp16','large-v3-turbo-int8'))]
            preferred=request['config'].get('speech_model')
            candidates.sort(key=lambda candidate:candidate!=preferred)
            stage_devices=probe_speech_devices(device,engine,cuda_verified)
            if engine=='whisper_cpp':
                installed=False
                for backend in stage_devices:
                    try: helper_executable(request,'whisper-helper-'+backend);installed=True
                    except ValueError: pass
            elif engine=='faster_whisper':
                installed=packages.get('faster-whisper')=='1.2.1' and packages.get('ctranslate2')=='4.7.1'
            else:
                installed=(platform.system()=='Darwin' and platform.machine()=='arm64'
                    and packages.get('mlx-whisper')=='0.4.3' and packages.get('mlx')=='0.29.3')
            entry=dict(id=engine,installed=installed,available=False,verified=False,device=stage_devices[0])
            if candidates:
                for backend in stage_devices:
                    try:
                        test=dict(request,audio_path=str(audio),speech_device=backend,config=dict(request['config'],speech_engine=engine,speech_model=candidates[0]),beam_size=1 if engine=='mlx_whisper' else 5)
                        output=empty_result(test,3.)
                        {'whisper_cpp':whisper_cpp_transcribe,'faster_whisper':transcribe,'mlx_whisper':mlx_transcribe}[engine](test,output)
                        entry.update(available=True,verified=True,device=backend,compute_type=output.get('compute_type'),model=candidates[0])
                        if backend=='metal' and not any(d['id']=='metal' for d in result['devices']):
                            result['devices'].append(dict(id='metal',verified=True,gpu_memory=gpu_memory_snapshot('metal')))
                        entry.pop('issue',None)
                        if engine==request['config']['speech_engine']:
                            result['resolved'].update(speech_device=backend,compute_type=output.get('compute_type'))
                        break
                    except Exception as error: entry['issue']=type(error).__name__+': selected engine/model/device did not pass execution probe'
                    finally: release_models()
            else: entry['issue']='Prepare a compatible model before verification'
            result['engines'].append(entry)
            release_models()
        for speaker in ('sherpa','community1'):
            speaker_device='cpu' if speaker=='sherpa' else ('cuda' if device=='cuda' or(device=='auto' and cuda_verified) else 'cpu')
            qualified=speaker=='community1' and device in ('metal','auto') and mps_qualified(request)
            if qualified: speaker_device='mps'
            if speaker=='sherpa':
                try: helper_executable(request,'parley_native_worker');installed=True
                except ValueError: installed=False
            else:
                installed=packages.get('pyannote.audio')=='4.0.4' and all(str(packages.get(p,'')).split('+')[0]=='2.8.0' for p in ('torch','torchaudio'))
            entry=dict(id=speaker,installed=installed,available=False,verified=False,device=speaker_device)
            if speaker=='community1' and device in ('metal','auto') and platform.system()=='Darwin':
                entry['mps_qualified']=qualified
                entry['note']='MPS matches an accepted version/artifact parity qualification' if qualified else 'CPU selected explicitly; MPS requires a matching inspected parity qualification'
            try:
                test=dict(request,audio_path=str(audio),speaker_device=speaker_device,config=dict(request['config'],speaker_pipeline=speaker))
                output=empty_result(test,3.)
                (diarize if speaker=='community1' else native_speakers)(test,output)
                entry.update(available=True,verified=True)
                if speaker_device=='mps' and not any(d['id']=='mps' for d in result['devices']):
                    result['devices'].append(dict(id='mps',verified=True,gpu_memory=gpu_memory_snapshot('mps')))
                if speaker==request['config']['speaker_pipeline']:
                    result['resolved']['speaker_device']=speaker_device
            except Exception as error: entry['issue']=type(error).__name__+': speaker package did not pass execution probe'
            result['speaker_pipelines'].append(entry)
            release_models()
    return result


def empty_result(request,duration):
    return dict(version=1,duration=duration,language=request.get('language'),segments=request.get('segments',[]),
        words=request.get('words',[]),turns=[],overlaps=[],embeddings=[],request_id=request.get('request_id'),audio_sha256=request.get('audio_sha256'))


def snapshot_artifacts(request,legacy,original):
    op=request['operation'];config=request['config'];ids=[]
    if op in ('process','speech_recognition') and (not legacy or original.get('recognizer') in ('faster','mlx')):
        compatible={'whisper_cpp':('large-v3-turbo','large-v3-turbo-q5_0'),
            'faster_whisper':('large-v3-turbo-fp16','large-v3-turbo-int8'),
            'mlx_whisper':('large-v3-turbo-mlx-fp16',)}
        if config['speech_model'] not in compatible[config['speech_engine']]:
            raise ValueError('Selected model format does not match the requested speech engine')
        ids.append(config['speech_model'])
    if op=='extract_embeddings':
        selected=request.get('embedding_model') or config.get('identification','same')
        if selected=='same': selected='campplus' if config['speaker_pipeline']=='sherpa' else 'wespeaker'
        if selected not in ('campplus','wespeaker'):
            raise ValueError('Choose an embedding model for extraction')
        ids.append('campplus' if selected=='campplus' else 'community1')
    elif op!='speech_recognition' and (not legacy or original.get('diarization')=='community'):
        ids+=['community1'] if config['speaker_pipeline']=='community1' else ['sherpa-segmentation','campplus']
    request['_resolved_artifacts']={artifact:resolve(request,artifact) for artifact in ids}


def run(original):
    started=time.monotonic()
    legacy='config' not in original
    request=normalize_request(original)
    op=request['operation']
    if op=='prepare': return prepare(request,progress)
    if op=='probe': return probe(request)
    source=Path(request['audio_path'])
    progress('worker_verifying', 0, 'Verifying the recording on the processing target')
    source_digest=sha256(source)
    if request.get('audio_sha256') and source_digest!=request['audio_sha256']:
        raise ValueError('Input audio checksum mismatch')
    request.setdefault('audio_sha256',source_digest)
    # Inference must consume already prepared artifacts. Credentials and model
    # license acceptance are handled by explicit preparation outside manifests.
    os.environ['HF_HUB_OFFLINE']='1'
    snapshot_artifacts(request,legacy,original)
    with tempfile.TemporaryDirectory(prefix='parley-audio-',dir=request.get('work_dir')) as folder:
        progress('worker_decoding', 0, 'Extracting audio and converting to 16 kHz mono on the target')
        audio=Path(folder)/'audio.wav'
        convert_audio(request, source, audio)
        with wave.open(str(audio),'rb') as wav: duration=wav.getnframes()/wav.getframerate()
        if duration<=0:
            raise ValueError('The source recording contains no decodable audio')
        request=dict(request,audio_path=str(audio))
        result=empty_result(request,duration)
        if request.get('words'):
            for i, segment in enumerate(result['segments']):
                words=[w for w in result['words'] if w['start'] >= segment['start']-1e-6 and w['end'] <= segment['end']+1e-6]
                for word in words: word['_source_segment']=i
                if ''.join(w['text'] for w in words).split() != segment['text'].split():
                    segment['_alignment_incomplete']=True
        result['provenance']=dict(audio_preprocessing=dict(version=1,sample_rate=16000,
            sample_format='pcm_s16le',channel_policy='mono_downmix',evidence_channel=0))
        if request.get('upstream_provenance') is not None:
            if not isinstance(request['upstream_provenance'],dict): raise ValueError('Invalid upstream provenance')
            result['provenance']['upstream']=request['upstream_provenance']
        for source,destination in [('upstream_recognition_device','recognition_device'),('upstream_compute_type','compute_type'),('upstream_recognition_seconds','recognition_seconds')]:
            if request.get(source) is not None: result[destination]=request[source]
        if op=='extract_embeddings':
            result['turns']=request.get('turns',[])
            selected=request.get('embedding_model') or request['config'].get('identification','same')
            if selected=='same': selected='campplus' if request['config']['speaker_pipeline']=='sherpa' else 'wespeaker'
            validate_clean_sources(request,duration)
            if selected=='wespeaker': community_extract(request,result)
            elif selected=='campplus': native_speakers(request,result,True)
            else: raise ValueError('Choose an embedding model for extraction')
        else:
            if op!='speech_recognition' and (not legacy or original.get('diarization')=='community'):
                phase=time.monotonic()
                (diarize if request['config']['speaker_pipeline']=='community1' else native_speakers)(request,result)
                result['diarization_seconds']=time.monotonic()-phase
            if op in ('process','speech_recognition') and (not legacy or original.get('recognizer') in ('faster','mlx')):
                result['segments']=[]
                phase=time.monotonic()
                {'faster_whisper':transcribe,'mlx_whisper':mlx_transcribe,'whisper_cpp':whisper_cpp_transcribe}[request['config']['speech_engine']](request,result)
                result['recognition_seconds']=time.monotonic()-phase
            if op!='speech_recognition' and (result['turns'] or result['segments'] or result['words']):
                progress('aligning', 0, 'Aligning transcript words with speakers and overlapping speech')
                annotate(result)
            if op=='speech_recognition':
                for segment in result['segments']: segment.pop('_alignment_incomplete',None)
                for word in result['words']: word.pop('_source_segment',None)
        worker_elapsed=time.monotonic()-started
        upstream_seconds=request.get('upstream_recognition_seconds',0.)
        if not isinstance(upstream_seconds,(int,float)) or not math.isfinite(upstream_seconds) or upstream_seconds<0:
            raise ValueError('Invalid upstream stage timing')
        result['elapsed_seconds']=worker_elapsed+upstream_seconds
        result['settings']=dict(config=request['config'],speech_device=request['speech_device'],speaker_device=request['speaker_device'],
            versions=versions(),sequential_loading=True,worker_operation=op)
        result['provenance']['runtime']=runtime_provenance()
        result['provenance']['sequential_loading']=True
        result['provenance']['timings_seconds']={k:result[k] for k in
            ('elapsed_seconds','diarization_seconds','recognition_seconds') if k in result}
        result['provenance']['timings_seconds']['worker_elapsed_seconds']=worker_elapsed
        result['provenance'].setdefault('resources',{})['host_processes']=process_memory_peaks()
        if request.get('experimental_speaker_mps'):
            result['provenance']['experimental_speaker_device']='mps-unqualified'
        return validate_result(result,request)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--request',type=Path)
    parser.add_argument('--response',type=Path)
    parser.add_argument('--probe',action='store_true')
    args=parser.parse_args()
    if args.probe:
        print(json.dumps(probe(normalize_request(dict(operation='probe'))),allow_nan=False))
        return
    if not args.request or not args.response: parser.error('--request and --response are required')
    failure_path(args.response).unlink(missing_ok=True)
    try:
        result=run(load_json(args.request))
        atomic_json(args.response,result)
    except Exception as error:
        # Exception messages/chains may contain credentials or authenticated
        # download URLs. Only this closed contract is exposed through status;
        # stderr remains in the supervisor's permission-protected private log.
        fields=public_failure(dict(protocol_version=VERSION, **error.public_error)) if isinstance(error,PublicWorkerError) else {}
        atomic_json(failure_path(args.response),dict(protocol_version=VERSION,**fields))
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__=='__main__': main()
