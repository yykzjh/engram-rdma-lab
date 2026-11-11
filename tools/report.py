#!/usr/bin/env python3
"""Create tables and scientific plots from the complete, untrimmed comparison."""
import argparse
import csv
import json
import statistics
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--results', type=Path, required=True, help='Directory containing sanitized per-configuration JSON files')
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
rows = []
for path in sorted(a.results.glob('*.json')):
    data = json.loads(path.read_text())
    if not {'config', 'samples'} <= data.keys():
        continue
    cfg = data['config']
    assert not data.get('failure'), path
    by = {}
    for s in data['samples']:
        assert s['data_ok'] and s['scale_ok'], path
        if not s['warmup']:
            by.setdefault(s['round'], []).append(s)
    assert len(by) == cfg['rounds']
    assert all(len(v) == 8 and {s['rank'] for s in v} == set(range(8)) for v in by.values()), path
    times = [max(s['e2e_ms'] for s in samples) for _, samples in sorted(by.items())]
    byte_count = next(iter(by.values()))[0]['bytes']
    median = statistics.median(times)
    graph = cfg.get('cuda_graph', cfg.get('gpu_graph', 1))
    mode = 'write' if cfg['variant'] == 'write' else ('read' if cfg['variant'] == 'baseline' else 'hybrid-read')
    if not graph: mode += '-api'
    rows.append(dict(name=cfg['name'], tokens=cfg['tokens'], distribution=cfg['distribution'], mode=mode,
                     rounds=len(times), median_ms=median, mean_ms=statistics.mean(times), max_ms=max(times),
                     min_ms=min(times), gbps=byte_count/median/1e6,
                     total_over_wall_gbps=len(times)*byte_count/sum(times)/1e6,
                     per_round_max_ms=times))
a.output.mkdir(parents=True, exist_ok=True)
(a.output/'summary.json').write_text(json.dumps(rows, indent=2)+'\n')
fields = [k for k in rows[0] if k != 'per_round_max_ms']
with (a.output/'summary.csv').open('w') as f:
    w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore', lineterminator='\n'); w.writeheader(); w.writerows(rows)
lookup = {(r['distribution'], r['tokens'], r['mode']): r for r in rows}
text = ['# 完整 fetch 对照结果', '',
        '每个点为 20 轮、每轮 8 ranks 最大完整 fetch wall time 的中位数。',
        '延迟以 ms、带宽以十进制 GB/s 表示；带宽只计每 rank embedding 字节，scales 处理包含在时间内。',
        '所有轮次均保留，均值、最大值见 `summary.csv`，逐轮值见 `summary.json` 与 `raw/`。', '']
for distribution in ['stratified', 'uniform']:
    text += [f'## {distribution}', '',
             '| Tokens | READ ms / GB/s | 混合 READ ms / GB/s | WRITE ms / GB/s | WRITE / 混合 READ 带宽 | 混合 READ API ms |',
             '|---:|---:|---:|---:|---:|---:|']
    for t in sorted({r['tokens'] for r in rows if r['distribution'] == distribution}):
        rr, hh, ww, api = [lookup[distribution,t,mode] for mode in ['read','hybrid-read','write','hybrid-read-api']]
        def pair(x): return f"{x['median_ms']:.4f} / {x['gbps']:.3f}"
        text.append(f"| {t} | {pair(rr)} | {pair(hh)} | {pair(ww)} | {ww['gbps']/hh['gbps']:.2f}x | {api['median_ms']:.4f} |")
    text += ['']
text += ['## 口径', '',
         'READ 是同一精简 DeepEP Engram 中关闭本地 CPU 直读的对照，包含已验证的 CUSTOM_STRIDE representative-rank 修正。混合 READ 同机直接加载 CPU 词表、跨机 GIN GET。WRITE 同机直接加载 CPU 词表，跨机由 CPU owner 构造 30-SGE verbs WRITE，并包含 GPU 分组、请求、DONE 和重排。', '',
         '每轮索引不同且两条路径使用同一 C++ mt19937 规则。词表、输出容量和配置相同。READ 使用 G=4 custom-stride rail 和 owner CPU 注册别名；WRITE 为各远端 owner/requester 对建立 RC 数据通道，NCCL_CROSS_NIC 只约束其 GIN 控制面。WRITE 每 rank 使用 4 个 CPU owner 服务线程及额外 staging / positions；READ 无每次查询的 CPU owner 服务。两者的 QP 组织、CPU 服务与排队机制不同，属于应用设计对照。', '',
         '主对照均为 CUDA Graph；API 列是普通 Python fetch/hook 的完整 fetch 时延；Graph/API 在不同进程中顺序运行，大规模测点不可直接相减作为 Python 开销。初始化、上游输入生成/上传和完整正确性校验不计入 fetch。', '',
         '![Bandwidth and latency](comparison.png)', '', '## 131072 tokens 的运行波动', '',
         '混合 READ 在约 105 ms 的快轮与更慢轮之间呈现明显波动。中位数会受慢轮比例影响；同时查看均值和逐轮数据。Graph/API 为顺序独立运行，不把二者中位数差额当作 Python 时间。', '',
         '| 分布 | 模式 | 中位数 ms | 均值 ms | 最小 ms | 最大 ms |',
         '|---|---|---:|---:|---:|---:|']
for distribution in ['stratified', 'uniform']:
    for mode in ['hybrid-read', 'hybrid-read-api', 'write']:
        x=lookup[distribution,131072,mode]
        text.append(f"| {distribution} | {mode} | {x['median_ms']:.3f} | {x['mean_ms']:.3f} | {x['min_ms']:.3f} | {x['max_ms']:.3f} |")
text += ['', '![Per-round variation](variability.png)', '']
(a.output/'README.md').write_text('\n'.join(text))
# Plotting is an optional reporting dependency, not a benchmark dependency.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.5), constrained_layout=True)
styles = [('read','GIN GET', '#64748b'), ('hybrid-read','CPU local + GIN GET', '#2563eb'),
          ('write','CPU local + SGE WRITE', '#dc2626'), ('hybrid-read-api','Hybrid GET, Python API', '#0d9488')]
for col, distribution in enumerate(['stratified', 'uniform']):
    for mode, label, color in styles:
        line = sorted((r for r in rows if r['distribution']==distribution and r['mode']==mode), key=lambda r:r['tokens'])
        x = [r['tokens'] for r in line]
        axes[0,col].plot(x,[r['gbps'] for r in line],marker='o',color=color,label=label,linewidth=1.8,markersize=4)
        axes[1,col].plot(x,[r['median_ms'] for r in line],marker='o',color=color,label=label,linewidth=1.8,markersize=4)
    for row in range(2):
        ax=axes[row,col]; ax.set_xscale('log',base=2); ax.set_xlabel('Tokens per requester')
        ax.grid(True,alpha=.2); ax.set_title(distribution); ax.spines[['top','right']].set_visible(False)
    axes[0,col].set_ylabel('Embedding GB/s per rank'); axes[0,col].set_ylim(bottom=0)
    axes[1,col].set_ylabel('Full-fetch latency (ms)'); axes[1,col].set_yscale('log')
axes[0,0].legend(fontsize=8,loc='upper left')
fig.suptitle('8 concurrent requesters / 8 CPU owners; FP8 256 B rows; 2 layers x 24 rows/token\nMedian of 20 per-round maximum-rank wall times; no slow-round trimming', fontsize=12)
for ext in ['png','svg']:fig.savefig(a.output/f'comparison.{ext}',dpi=180)
plt.close(fig)
print(f'Summarized {len(rows)} configurations into {a.output}')

fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.8), constrained_layout=True)
for ax, distribution in zip(axes, ['stratified', 'uniform']):
    for mode,label,color in [styles[1],styles[3],styles[2]]:
        v=lookup[distribution,131072,mode]['per_round_max_ms']
        ax.plot(range(1,len(v)+1),v,marker='o',markersize=3,label=label,color=color,alpha=.85)
    ax.set_title(distribution);ax.set_xlabel('Measured round (separate sequential runs)');ax.set_ylabel('Max-rank full fetch (ms)')
    ax.grid(True,alpha=.2);ax.set_ylim(bottom=0);ax.spines[['top','right']].set_visible(False)
axes[0].legend(fontsize=8)
fig.suptitle('131072 tokens: retain every round, including slow READ rounds',fontsize=11)
for ext in ['png','svg']:fig.savefig(a.output/f'variability.{ext}',dpi=180)
plt.close(fig)

# Matplotlib SVG paths contain optional trailing spaces; normalize text exports.
for path in a.output.glob('*.svg'):
    path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')
