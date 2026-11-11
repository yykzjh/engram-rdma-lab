#!/usr/bin/env python3
"""Execute a fixed configuration list sequentially; never overlap GPU benchmarks."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--configs', type=Path, required=True)
p.add_argument('--hosts', required=True)
p.add_argument('--ssh-port', type=int, default=22)
p.add_argument('--mpi-prefix', type=Path)
p.add_argument('--mpirun', default='mpirun')
p.add_argument('--env-script', type=Path)
p.add_argument('--runtime-root', type=Path, required=True, help='One shared prepare_runtime.py output; ENGRAM_READ_C2C selects the path')
p.add_argument('--write-binary', type=Path)
p.add_argument('--indices', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
root = Path(__file__).resolve().parent
for cfg in json.loads(a.configs.read_text()):
    write = cfg['variant'] == 'write'
    script = root / ('run_write.py' if write else 'run_deepep.py')
    argv = [sys.executable, str(script), '--hosts', a.hosts, '--ssh-port', str(a.ssh_port), '--mpirun', a.mpirun,
            '--output', str((a.output/cfg['name']).resolve()), '--name', cfg['name'], '--tokens', str(cfg['tokens']),
            '--rounds', str(cfg.get('rounds', 20)), '--distribution', cfg['distribution'],
            '--indices', str(a.indices.resolve()), '--c2c', str(cfg.get('c2c',0))]
    if a.mpi_prefix: argv += ['--mpi-prefix', str(a.mpi_prefix)]
    if a.env_script: argv += ['--env-script', str(a.env_script)]
    if write:
        assert a.write_binary, '--write-binary is required for WRITE configurations'
        argv += ['--binary', str(a.write_binary.resolve())]
    else:
        assert cfg['variant']=='read', 'Only read and write are supported'
        argv += ['--runtime', str(a.runtime_root.resolve()), '--variant', 'read',
                 '--rd-atomic', str(cfg.get('rd_atomic',8 if cfg['tokens']==1 or cfg['tokens']>=131072 else 16)),
                 '--jit-cache', str((a.output/'jit').resolve())]
        if cfg.get('graph',True): argv += ['--graph']
    if write and cfg.get('async_check'): argv += ['--async-check']
    subprocess.run(argv, check=True)
