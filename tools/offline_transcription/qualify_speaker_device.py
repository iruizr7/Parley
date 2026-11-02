#!/usr/bin/env python3
"""Private CPU/MPS parity harness. It never enables a product capability.

Use the exact same audio, assets, expected speaker count and pinned runtime.
CPU parity is a numerical/behavioral check, not a manually reviewed reference.
"""
import argparse
import math
from pathlib import Path
from protocol import atomic_json,load_json,sha256
from worker import run


def turn_comparison(reference,candidate):
    """Permutation-invariant speaker-time disagreement, retaining overlap."""
    from scipy.optimize import linear_sum_assignment
    labels=sorted({t['speaker'] for t in reference['turns']})
    proposed=sorted({t['speaker'] for t in candidate['turns']})
    edges=sorted({0.,reference['duration'],candidate['duration']}|
        {float(t[k]) for value in (reference,candidate) for t in value['turns'] for k in ('start','end')})
    intervals=[];weights=[[0. for _ in proposed] for _ in labels]
    for start,end in zip(edges,edges[1:]):
        active=[{t['speaker'] for t in value['turns'] if t['start']<end and t['end']>start}
                for value in (reference,candidate)]
        intervals.append((end-start,*active))
        for i,label in enumerate(labels):
            for j,other in enumerate(proposed):
                if label in active[0] and other in active[1]: weights[i][j]+=end-start
    mapping={}
    if labels and proposed:
        rows,columns=linear_sum_assignment([[-w for w in row] for row in weights])
        mapping={proposed[j]:labels[i] for i,j in zip(rows,columns)}
    total=missed=false=confused=overlap_disagreement=overlap_union=overlap_intersection=0.
    for duration,expected,actual in intervals:
        actual={mapping.get(label,'unmatched:'+label) for label in actual}
        total+=duration*len(expected)
        missed+=duration*max(0,len(expected)-len(actual))
        false+=duration*max(0,len(actual)-len(expected))
        confused+=duration*(min(len(expected),len(actual))-len(expected&actual))
        a,b=len(expected)>1,len(actual)>1
        overlap_disagreement+=duration*(a!=b)
        overlap_union+=duration*(a or b);overlap_intersection+=duration*(a and b)
    return dict(reference_speakers=len(labels),candidate_speakers=len(proposed),
        reference_speaker_seconds=total,missed_speaker_seconds=missed,false_speaker_seconds=false,
        confused_speaker_seconds=confused,speaker_time_error_fraction=(missed+false+confused)/max(total,1e-9),
        overlap_disagreement_seconds=overlap_disagreement,
        overlap_intersection_over_union=overlap_intersection/overlap_union if overlap_union else 1.,
        reference_turns=len(reference['turns']),candidate_turns=len(candidate['turns']))


def embedding_comparison(reference,candidate):
    comparisons=[]
    for expected in reference.get('embeddings',[]):
        actual=next((e for e in candidate.get('embeddings',[]) if e['speaker']==expected['speaker']),None)
        if actual is None:
            comparisons.append(dict(speaker=expected['speaker'],compatible=False,reason='Missing candidate evidence'))
            continue
        same=(expected.get('embedding_space',{}).get('id')==actual.get('embedding_space',{}).get('id')
              and expected.get('source_intervals')==actual.get('source_intervals')
              and len(expected['vector'])==len(actual['vector']))
        entry=dict(speaker=expected['speaker'],compatible=same)
        if same:
            a,b=expected['vector'],actual['vector']
            entry['cosine_similarity']=sum(x*y for x,y in zip(a,b))/math.sqrt(sum(x*x for x in a)*sum(x*x for x in b))
            entry['max_absolute_difference']=max(abs(x-y) for x,y in zip(a,b))
        comparisons.append(entry)
    return comparisons


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--audio',type=Path,required=True)
    parser.add_argument('--assets',type=Path,required=True)
    parser.add_argument('--device',choices=('cpu','cuda','mps'),required=True)
    parser.add_argument('--response',type=Path,required=True)
    parser.add_argument('--baseline',type=Path,help='CPU result for the same exact audio and worker configuration')
    parser.add_argument('--comparison',type=Path)
    parser.add_argument('--fixed-extraction-response',type=Path)
    parser.add_argument('--fixed-reference-response',type=Path)
    parser.add_argument('--cache-dir',type=Path)
    parser.add_argument('--num-speakers',type=int,default=0)
    args=parser.parse_args()
    request=dict(version=1,operation='speaker_analysis',audio_path=str(args.audio),audio_sha256=sha256(args.audio),
        assets_dir=str(args.assets),speech_device='cpu',speaker_device=args.device,
        experimental_speaker_mps=args.device=='mps',num_speakers=args.num_speakers,
        config=dict(speech_engine='whisper_cpp',speech_model='large-v3-turbo-q5_0',
            speaker_pipeline='community1',identification='same'))
    if args.cache_dir: request['cache_dir']=str(args.cache_dir)
    result=run(request);atomic_json(args.response,result)
    if args.baseline:
        baseline=load_json(args.baseline)
        # Accept direct Analysis or the generic validation service envelope.
        baseline=baseline.get('result',baseline)
        baseline=baseline.get('analysis',baseline)
        digest=baseline.get('audio_sha256') or baseline.get('settings',{}).get('audio_sha256')
        if digest!=request['audio_sha256']: raise ValueError('Baseline does not match the exact input audio checksum')
        report=dict(version=1,reference_device=baseline.get('diarization_device'),candidate_device=args.device,
            audio_sha256=digest,timeline=turn_comparison(baseline,result),
            reference_seconds=baseline.get('diarization_seconds'),candidate_seconds=result.get('diarization_seconds'),
            reference_runtime=baseline.get('provenance',{}).get('runtime'),candidate_runtime=result['provenance']['runtime'],
            product_enabled=False,qualification_note='CPU agreement is not ground truth; review speaker and overlap quality before enabling MPS.')
        spans=[s for e in baseline.get('embeddings',[]) for s in e.get('source_intervals',[])]
        if spans:
            # Re-extract the CPU reference with this same pinned adapter. This
            # also makes comparisons explicit when an older saved run predates
            # the bounded clean-window policy; old vectors are never relabeled.
            reference_extracted=run(dict(request,operation='extract_embeddings',speaker_device='cpu',
                experimental_speaker_mps=False,embedding_model='wespeaker',source_intervals=spans))
            exact=[s for e in reference_extracted['embeddings'] for s in e['source_intervals']]
            extracted=run(dict(request,operation='extract_embeddings',embedding_model='wespeaker',source_intervals=exact))
            report['fixed_clean_interval_embeddings']=embedding_comparison(reference_extracted,extracted)
            report['fixed_reference_runtime']=reference_extracted['provenance']['runtime']
            if args.fixed_reference_response: atomic_json(args.fixed_reference_response,reference_extracted)
            if args.fixed_extraction_response: atomic_json(args.fixed_extraction_response,extracted)
        if not args.comparison: parser.error('--comparison is required with --baseline')
        atomic_json(args.comparison,report)


if __name__=='__main__': main()
