#!/usr/bin/env python3
"""Create isolated runtime variants without changing an installed DeepEP package."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--package', type=Path, required=True, help='Built deep_ep package directory containing _C*.so')
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
assert len(list(a.package.glob('_C*.so'))) == 1, 'Build the source extension first'
assert a.output.resolve() != a.package.resolve()
manifest = {}
for name, enabled, lanes in [('baseline', 0, 16), ('hybrid8', 1, 8), ('hybrid16', 1, 16), ('hybrid32', 1, 32)]:
    target = a.output / name / 'deep_ep'
    assert not target.exists(), f'Refusing to overwrite {target}'
    shutil.copytree(a.package, target, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    cfg = target / 'include/deep_ep/impls/engram/local_cpu_config.cuh'
    # Rewrite only numeric defaults. Both fetch and wait include this same file.
    import re
    text = re.sub(r'(#define EP_ENGRAM_LOCAL_CPU_READ )\d+', rf'\g<1>{enabled}', cfg.read_text())
    text = re.sub(r'(#define EP_ENGRAM_LOCAL_COPY_LANES )\d+', rf'\g<1>{lanes}', text)
    cfg.write_text(text)
    files = [*target.glob('_C*.so'), *cfg.parent.glob('*.cuh')]
    manifest[name] = dict(local_cpu_read=enabled, local_copy_lanes=lanes,
                          hashes={f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in files})
(a.output / 'manifest.json').write_text(json.dumps(manifest, indent=2))
print(json.dumps(manifest, indent=2))
