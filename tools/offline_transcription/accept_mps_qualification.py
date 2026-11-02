#!/usr/bin/env python3
"""Accept an inspected private CPU/MPS comparison on its matching runtime."""
import argparse
import json
from pathlib import Path
from qualification import accept_mps


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--assets',type=Path,required=True)
    parser.add_argument('--cache-dir',type=Path,required=True)
    parser.add_argument('--comparison',type=Path,required=True)
    parser.add_argument('--analysis',type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(accept_mps(dict(assets_dir=str(args.assets),cache_dir=str(args.cache_dir)),args.comparison,args.analysis)))


if __name__=='__main__': main()
