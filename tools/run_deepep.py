#!/usr/bin/env python3
"""Run actual DeepEP Engram API/hooks with a fixed local/remote JIT configuration."""
import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path
from mpi_common import add_common, command, environment, ensure_idle, ssh

p = argparse.ArgumentParser(description=__doc__)
add_common(p)
p.add_argument('--runtime', type=Path, required=True, help='Parent of the built deep_ep package')
p.add_argument('--variant', default='hybrid16')
p.add_argument('--local-cpu-read', type=int, choices=[0, 1], required=True)
p.add_argument('--local-copy-lanes', type=int, default=16)
p.add_argument('--rd-atomic', type=int, choices=[8, 16], required=True)
p.add_argument('--graph', action='store_true')
p.add_argument('--master-port', type=int, default=23890)
p.add_argument('--jit-cache', type=Path, required=True, help='Separate cache per variant')
p.add_argument('--benchmark', type=Path, default=Path(__file__).resolve().parents[1]/'benchmarks/compare_deepep.py')
a = p.parse_args()
ensure_idle(a)
a.output.mkdir(parents=True, exist_ok=True)
assert not list(a.output.glob('deepep-r*.json')), 'Use a fresh output directory to avoid stale results'
values = dict(PYTHONPATH=a.runtime, WORLD_SIZE=2, MASTER_ADDR=a.hosts.split(',')[0], MASTER_PORT=a.master_port,
              NCCL_CROSS_NIC=0, NCCL_GIN_TYPE=3, NCCL_GIN_GDAKI_MAX_QP_RD_ATOMIC=a.rd_atomic,
              NCCL_GIN_GDAKI_MAX_DEST_RD_ATOMIC=16, NCCL_DEBUG='WARN', OMP_NUM_THREADS=1,
              MKL_NUM_THREADS=1, EP_JIT_CACHE_DIR=a.jit_cache)
argv = ['python', str(a.benchmark), '--tokens', str(a.tokens), '--rounds', str(a.rounds), '--warmups', '1',
        '--indices-dir', str(a.indices), '--output-dir', str(a.output), '--distribution', a.distribution,
        '--vary-indices', '--config-read-requests', '16', '--qp-depth', '0',
        '--local-cpu-read', str(a.local_cpu_read), '--local-copy-lanes', str(a.local_copy_lanes)]
if a.graph:
    argv += ['--cuda-graph']
shell = environment(a, values) + 'export RANK=$OMPI_COMM_WORLD_RANK; exec ' + shlex.join(argv)
with (a.output/'mpi.log').open('w') as log:
    proc = subprocess.run(command(a, 2, 1) + ['bash', '-c', shell], stdout=log, stderr=subprocess.STDOUT)
for rank in range(4, 8):
    path = a.output/f'deepep-r{rank}.json'
    cp = subprocess.run(ssh(a) + ['cat '+shlex.quote(str(path))], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if cp.returncode == 0:
        path.write_bytes(cp.stdout)
samples, metadata = [], []
for path in sorted(a.output.glob('deepep-r*.json')):
    data = json.loads(path.read_text()); samples += data['samples']; metadata.append(data['metadata'])
failure = f'MPI exit {proc.returncode}' if proc.returncode else None
if not failure and (len(samples) != 8*(a.rounds+1) or not all(s['data_ok'] and s['scale_ok'] for s in samples)):
    failure = 'sample count or validation'
config = dict(name=a.name, tokens=a.tokens, rounds=a.rounds, distribution=a.distribution,
              variant=a.variant, cuda_graph=a.graph, rd_atomic=a.rd_atomic)
(a.output/'results.json').write_text(json.dumps(dict(config=config, failure=failure, samples=samples, metadata=metadata), indent=2))
print(json.dumps(dict(name=a.name, failure=failure, rank_samples=len(samples))))
if failure:
    print((a.output/'mpi.log').read_text()[-10000:], file=sys.stderr)
    sys.exit(1)
