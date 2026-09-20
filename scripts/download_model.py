#!/usr/bin/env python3
"""Download a pinned checkpoint into the local model store."""
import argparse
from pathlib import Path
from huggingface_hub import snapshot_download

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--revision', default='1c5edc17a7acd8701df6fc341c0d179f1c62c982')
p.add_argument('--variant', choices=['english', 'multilingual', 'typed-decisions', 'all'], default='english')
a = p.parse_args()
variants = ['english', 'multilingual', 'typed-decisions'] if a.variant == 'all' else [a.variant]
root = Path(__file__).resolve().parents[1] / 'models' / 'laya'
for variant in variants:
    prefix = '' if variant == 'english' else variant + '/'
    snapshot_download('convaiinnovations/laya', revision=a.revision, local_dir=root,
                      allow_patterns=[prefix + x for x in ['model.safetensors', 'rl_agent_config.json', 'encoder/*', 'tokenizer/*']])
    (root / prefix / 'REVISION').write_text(a.revision + '\n')
    print(root / prefix)
