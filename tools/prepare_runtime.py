#!/usr/bin/env python3
"""Copy a built Engram runtime; C2C is selected by ENGRAM_READ_C2C per Buffer."""
import argparse,hashlib,json,shutil
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--package',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
assert len(list(a.package.glob('_C*.so')))==1
assert not a.output.exists(),a.output
shutil.copytree(a.package,a.output/'deep_ep',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
files=[*a.output.rglob('*.so'),*a.output.rglob('*.cuh')]
(a.output/'manifest.json').write_text(json.dumps({str(f.relative_to(a.output)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files},indent=2))
