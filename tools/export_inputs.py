#!/usr/bin/env python3
"""Export the native C++ mt19937 inputs used by both communication implementations."""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--generator', type=Path, required=True, help='Built export_indices executable')
p.add_argument('--output', type=Path, required=True)
p.add_argument('--tokens', default='1,8,64,512,4096,32768,131072,262144')
p.add_argument('--ranks', default='0,1,2,3,4,5,6,7')
p.add_argument('--distributions', default='stratified,uniform')
p.add_argument('--rounds', type=int, default=20)
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=True)
for rank in map(int, a.ranks.split(',')):
    for t in map(int, a.tokens.split(',')):
        for distribution in a.distributions.split(','):
            subprocess.run([str(a.generator.resolve()), '--rank', str(rank), '--tokens', str(t),
                            '--rounds', str(a.rounds), '--warmups', '1', '--distribution', distribution,
                            '--dump-indices', str(a.output.resolve())], check=True)
files = {f.name: dict(bytes=f.stat().st_size, sha256=hashlib.sha256(f.read_bytes()).hexdigest())
         for f in a.output.glob('*-e*.bin')}
(a.output / 'manifest.json').write_text(json.dumps(files, indent=2))
print(f'Exported {len(files)} files; input generation is outside fetch timing.')
