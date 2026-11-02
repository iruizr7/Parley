#!/usr/bin/env python3
"""Private MLX comparison using the same production worker and provenance."""
import argparse
import json
from pathlib import Path
from protocol import atomic_json
from worker import run


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--audio',type=Path,required=True)
    parser.add_argument('--assets',required=True)
    parser.add_argument('--response',type=Path,required=True)
    parser.add_argument('--language',default='es')
    parser.add_argument('--speech-model-path',type=Path)
    parser.add_argument('--cache-dir',type=Path)
    parser.add_argument('--prepare',action='store_true',help='Explicitly register/download selected artifacts before comparison')
    args=parser.parse_args()
    request=dict(version=1,operation='process',audio_path=str(args.audio),assets_dir=args.assets,
        language=args.language,speech_device='metal',speaker_device='cpu',beam_size=1,
        config=dict(speech_engine='mlx_whisper',speech_model='large-v3-turbo-mlx-fp16',
            speaker_pipeline='community1',identification='same'))
    if args.speech_model_path:
        request['model_paths']={'large-v3-turbo-mlx-fp16':str(args.speech_model_path)}
    if args.cache_dir: request['cache_dir']=str(args.cache_dir)
    if args.prepare: run(dict(request,operation='prepare'))
    result=run(request)
    atomic_json(args.response,result)
    print(json.dumps({k:result[k] for k in ('elapsed_seconds','diarization_seconds','recognition_seconds')}))


if __name__=='__main__': main()
