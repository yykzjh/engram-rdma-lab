"""MPI command and environment helpers for a two-node/eight-rank experiment."""
import os
import shlex
import subprocess
from pathlib import Path


def add_common(parser):
    parser.add_argument('--hosts', required=True, help='Two SSH-reachable hosts, comma separated')
    parser.add_argument('--ssh-port', type=int, default=22)
    parser.add_argument('--mpi-prefix', type=Path)
    parser.add_argument('--mpirun', default='mpirun')
    parser.add_argument('--env-script', type=Path, help='Optional existing shell environment script on both nodes')
    parser.add_argument('--output', type=Path, required=True, help='Same absolute path on both nodes')
    parser.add_argument('--name', required=True)
    parser.add_argument('--tokens', type=int, required=True)
    parser.add_argument('--rounds', type=int, default=20)
    parser.add_argument('--distribution', choices=['stratified', 'uniform'], default='stratified')
    parser.add_argument('--indices', type=Path, required=True)


def ssh(args):
    hosts = args.hosts.split(',')
    assert len(hosts) == 2 and all(hosts), 'Exactly two hosts are required'
    return ['ssh', '-p', str(args.ssh_port), '-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=yes', hosts[1]]


def command(args, processes, per_node):
    hosts = args.hosts.split(',')
    result = [args.mpirun]
    if os.geteuid() == 0:
        result += ['--allow-run-as-root']
    if args.mpi_prefix:
        result += ['--prefix', str(args.mpi_prefix)]
    result += ['-np', str(processes), '--host', ','.join(f'{h}:{per_node}' for h in hosts),
               '--map-by', f'ppr:{per_node}:node', '--bind-to', 'none', '--mca', 'pml', 'ob1',
               '--mca', 'btl', 'self,vader,tcp', '--mca', 'osc', 'pt2pt', '--mca', 'plm_rsh_args',
               f'-p {args.ssh_port} -o BatchMode=yes -o StrictHostKeyChecking=yes']
    return result


def environment(args, values):
    script = 'set -euo pipefail; '
    if args.env_script:
        script += 'source ' + shlex.quote(str(args.env_script)) + '; '
    script += 'export ' + ' '.join(f'{k}={shlex.quote(str(v))}' for k, v in values.items()) + '; '
    return script


def ensure_idle(args):
    query = ['nvidia-smi', '--query-compute-apps=pid,process_name,used_gpu_memory', '--format=csv,noheader']
    for cmd in [query, ssh(args) + [shlex.join(query)]]:
        assert not subprocess.check_output(cmd, text=True).strip(), 'GPU busy; no processes were stopped'
