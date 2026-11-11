#!/usr/bin/env python3
"""Summarize eight concurrent ranks without trimming slow rounds."""
import argparse
import csv
import json
import statistics
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('inputs', nargs='+', type=Path, help='Per-configuration JSON files containing config and samples')
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
rows = []
for path in a.inputs:
    data = json.loads(path.read_text())
    assert not data.get('failure'), path
    by = {}
    for s in data['samples']:
        assert s['data_ok'] and s['scale_ok'], path
        if not s['warmup']:
            by.setdefault(s['round'], []).append(s)
    assert by and all(len(v) == 8 and {s['rank'] for s in v} == set(range(8)) for v in by.values()), path
    times = [max(s['e2e_ms'] for s in samples) for _, samples in sorted(by.items())]
    byte_count = next(iter(by.values()))[0]['bytes']
    rows.append(dict(name=data['config']['name'], tokens=data['config']['tokens'], rounds=len(times),
                     median_ms=statistics.median(times), mean_ms=statistics.mean(times), max_ms=max(times),
                     gbps=byte_count/statistics.median(times)/1e6,
                     total_over_wall_gbps=len(times)*byte_count/sum(times)/1e6))
a.output.parent.mkdir(parents=True, exist_ok=True)
with a.output.open('w') as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
print(a.output)
