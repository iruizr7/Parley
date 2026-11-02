#!/usr/bin/env python3
"""Summarize benchmark results without printing private transcript text."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import unicodedata


def words(text):
    return re.findall(r'\w+', unicodedata.normalize('NFC', text).casefold())


def word_error_rate(reference, hypothesis):
    previous = list(range(len(hypothesis) + 1))
    for i, expected in enumerate(reference, 1):
        current = [i]
        for j, actual in enumerate(hypothesis, 1):
            current.append(min(previous[j] + 1, current[j-1] + 1, previous[j-1] + (expected != actual)))
        previous = current
    return previous[-1] / max(1, len(reference))


def public_settings(value):
    """Only curated identifiers, never arbitrary paths/targets/custom strings."""
    value=value if isinstance(value,dict) else {}
    config=value.get('config') if isinstance(value.get('config'),dict) else value
    choices={'speech_engine':{'whisper_cpp','faster_whisper','mlx_whisper'},
        'speech_model':{'large-v3-turbo','large-v3-turbo-q5_0','large-v3-turbo-fp16','large-v3-turbo-int8','large-v3-turbo-mlx-fp16'},
        'speaker_pipeline':{'sherpa','community1'},'identification':{'same','off','campplus','wespeaker'},
        'recognizer':{'native','faster','mlx'},'diarization':{'native','community'},'model':{'fast','precise'}}
    return {key:config[key] for key,allowed in choices.items() if isinstance(config.get(key),str) and config[key] in allowed}


def public_report(report):
    result=dict(results=[],accuracy_note=report['accuracy_note'])
    metrics=('audio_seconds','processing_seconds','words','segments','speakers','overlap_intervals','overlap_seconds','word_error_rate',
        'worker_peak_rss_bytes','completed_children_high_water_rss_bytes','upload_wall_seconds','result_download_wall_seconds')
    devices={'cpu','cuda','vulkan','metal','mps'}
    for i,private in enumerate(report['results'],1):
        row=dict(name='run_'+str(i),settings=public_settings(private.get('settings')))
        for key in metrics:
            value=private.get(key)
            if isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value) and value>=0: row[key]=value
        for key in ('recognition_device','diarization_device'):
            row[key]=private[key] if isinstance(private.get(key),str) and private[key] in devices else None
        result['results'].append(row)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('results', nargs='+', type=Path)
    parser.add_argument('--audio', type=Path)
    parser.add_argument('--reference', type=Path, help='Manually corrected plain text for the exact excerpt')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = dict(results=[])
    if args.audio:
        report['audio_sha256'] = hashlib.sha256(args.audio.read_bytes()).hexdigest()
    reference = words(args.reference.read_text(encoding='utf-8')) if args.reference else None
    for path in args.results:
        data = json.loads(path.read_text(encoding='utf-8'))
        text = ' '.join(s['text'] for s in data['segments'])
        result = dict(name=path.stem, audio_seconds=data['duration'],
            processing_seconds=data.get('elapsed_seconds'),
            recognition_device=data.get('recognition_device'),diarization_device=data.get('diarization_device'),
            words=len(words(text)),segments=len(data['segments']),
            speakers=len({t['speaker'] for t in data['turns']}),
            overlap_intervals=len(data['overlaps']),overlap_seconds=sum(o['end']-o['start'] for o in data['overlaps']),
            settings=data.get('settings'))
        provenance=data.get('provenance',{})
        host=provenance.get('resources',{}).get('host_processes',{})
        transport=provenance.get('transport',{})
        for key in ('worker_peak_rss_bytes','completed_children_high_water_rss_bytes'): result[key]=host.get(key)
        for key in ('upload_wall_seconds','result_download_wall_seconds'): result[key]=transport.get(key)
        if reference is not None:
            result['word_error_rate'] = word_error_rate(reference, words(text))
        report['results'].append(result)
    report['accuracy_note'] = 'No accuracy ranking without a manually corrected reference.' if reference is None else 'Word error rate ignores case/punctuation, retains Spanish accents. Speaker accuracy needs timed speaker reference.'
    args.output.write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(public_report(report),indent=2,ensure_ascii=False,allow_nan=False))


if __name__ == '__main__':
    main()
