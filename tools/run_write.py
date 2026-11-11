#!/usr/bin/env python3
"""Run the native SGE WRITE prototype with GPU requests, local CPU loads, DONE and scatter."""
import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path
from mpi_common import add_common, command, environment, ensure_idle, ssh

p = argparse.ArgumentParser(description=__doc__)
add_common(p)
p.add_argument('--binary', type=Path, required=True, help='engram_sge_write; sibling gpu_request.ptx must exist on both nodes')
p.add_argument('--data-port', type=int, default=26700)
a = p.parse_args()
ensure_idle(a)
a.output.mkdir(parents=True, exist_ok=True)
assert not list(a.output.glob('rank-*.jsonl')), 'Use a fresh output directory'
values = dict(NCCL_CROSS_NIC=0, NCCL_GIN_TYPE=3, NCCL_GIN_GDAKI_MAX_QP_RD_ATOMIC=8,
              NCCL_GIN_GDAKI_MAX_DEST_RD_ATOMIC=16, NCCL_DEBUG='WARN', OMP_NUM_THREADS=1, ENGRAM_GPU_DEBUG='')
argv = [str(a.binary), '--tokens', str(a.tokens), '--rounds', str(a.rounds), '--warmups', '1',
        '--ips', a.hosts, '--indices-dir', str(a.indices), '--distribution', a.distribution,
        '--methods', 'write', '--layout', 'merged', '--merge-rows', '30', '--pool', '1024', '--qps', '4',
        '--engine', 'ex', '--compact-qp', '1', '--protocol', 'frame', '--compact-request', '1',
        '--done-gpu', '1', '--gpu-request', '1', '--vary-indices', '1', '--profile-routes', '0',
        '--gpu-graph', '1', '--adaptive-grid', '1', '--compact-done', '1', '--hybrid-local', '1',
        '--owner-poll-us', '0', '--owner-windows', '1', '--port', str(a.data_port)]
# The worker output directory must exist before per-rank shell redirection.
subprocess.run(ssh(a) + ['mkdir -p '+shlex.quote(str(a.output))], check=True)
prefix = shlex.quote(str(a.output/'rank-'))
shell = (environment(a, values) + 'exec ' + shlex.join(argv) + ' > '+prefix+
         '${OMPI_COMM_WORLD_RANK}.jsonl 2> '+prefix+'${OMPI_COMM_WORLD_RANK}.stderr')
with (a.output/'mpi.log').open('w') as log:
    proc = subprocess.run(command(a, 8, 4) + ['bash', '-c', shell], stdout=log, stderr=subprocess.STDOUT)
for rank in range(4, 8):
    for ext in ['jsonl', 'stderr']:
        path = a.output/f'rank-{rank}.{ext}'
        cp = subprocess.run(ssh(a) + ['cat '+shlex.quote(str(path))], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if cp.returncode == 0:
            path.write_bytes(cp.stdout)
samples = []
for path in sorted(a.output.glob('rank-*.jsonl')):
    samples += [json.loads(line) for line in path.read_text().splitlines() if line.startswith('{')]
failure = f'MPI exit {proc.returncode}' if proc.returncode else None
if not failure and (len(samples) != 8*(a.rounds+1) or not all(s['data_ok'] and s['scale_ok'] for s in samples)):
    failure = 'sample count or validation'
config = dict(name=a.name, tokens=a.tokens, rounds=a.rounds, distribution=a.distribution,
              variant='write', gpu_graph=1, qps=4, pool=1024, merge_rows=30)
(a.output/'results.json').write_text(json.dumps(dict(config=config, failure=failure, samples=samples), indent=2))
print(json.dumps(dict(name=a.name, failure=failure, rank_samples=len(samples))))
if failure:
    print((a.output/'mpi.log').read_text()[-10000:], file=sys.stderr)
    sys.exit(1)
