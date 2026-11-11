#!/usr/bin/env python3
"""Summarize the retained READ/WRITE designs and their opt-in C2C paths."""
import argparse,csv,json,statistics
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--results',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
rows=[]
for path in sorted(a.results.glob('*.json')):
 d=json.loads(path.read_text())
 if 'samples' not in d or 'config' not in d:continue
 c=d['config']
 if c.get('async_check') or c.get('target_umr'):continue
 assert not d.get('failure'),path
 by={}
 for s in d['samples']:
  assert s['data_ok'] and s['scale_ok'],path
  if not s['warmup']:by.setdefault(s['round'],[]).append(s)
 assert len(by)==c['rounds'] and all(len(v)==8 and {s['rank'] for s in v}==set(range(8)) for v in by.values()),path
 times=[max(s['e2e_ms'] for s in v) for _,v in sorted(by.items())];median=statistics.median(times);b=c['tokens']*12288
 rows.append(dict(name=c['name'],tokens=c['tokens'],distribution=c['distribution'],design=c['variant'],c2c=c.get('c2c',0),rounds=len(times),median_ms=median,mean_ms=statistics.mean(times),min_ms=min(times),max_ms=max(times),gbps=b/median/1e6,total_over_wall_gbps=b/statistics.mean(times)/1e6,per_round_max_ms=times))
a.output.mkdir(parents=True,exist_ok=True)
(a.output/'summary.json').write_text(json.dumps(rows,indent=2)+'\n')
fields=[k for k in rows[0] if k!='per_round_max_ms']
with (a.output/'summary.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore',lineterminator='\n');w.writeheader();w.writerows(rows)
lookup={(r['distribution'],r['tokens'],r['design'],r['c2c']):r for r in rows}
text=['# 两套设计与 C2C 开关对照','','每组20轮；每轮取8 ranks完整fetch wall time最大值，再取中位数。带宽仅计每rank逻辑embedding字节，包含本地查询；scales成本计入时间。不裁剪慢轮。','']
for dist in ['stratified','uniform']:
 text += [f'## {dist}','','| Tokens | READ ms / GB/s | READ+C2C ms / GB/s | WRITE ms / GB/s | WRITE+C2C ms / GB/s |','|---:|---:|---:|---:|---:|']
 for t in sorted({r['tokens'] for r in rows if r['distribution']==dist}):
  values=[lookup[dist,t,v,c] for v,c in [('read',0),('read',1),('write',0),('write',1)]]
  text.append('| '+str(t)+' | '+' | '.join(f"{v['median_ms']:.4f} / {v['gbps']:.3f}" for v in values)+' |')
 text+=['']
text+=['READ在部分大规模点存在运行波动，请同时查看均值、最大值和逐轮数据。两套设计的CPU服务线程、QP连接组织和staging资源不同。WRITE的forward控制由GPU发起/等待，owner CPU仍构造和提交SGE WRITE。','','![Latency and bandwidth](comparison.png)','']
(a.output/'README.md').write_text('\n'.join(text))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axes=plt.subplots(2,2,figsize=(11.5,7.5),constrained_layout=True)
styles=[('read',0,'READ','#64748b'),('read',1,'READ + C2C','#2563eb'),('write',0,'SGE WRITE','#ea580c'),('write',1,'SGE WRITE + C2C','#dc2626')]
for col,dist in enumerate(['stratified','uniform']):
 for design,c,label,color in styles:
  v=sorted([r for r in rows if r['distribution']==dist and r['design']==design and r['c2c']==c],key=lambda r:r['tokens']);x=[r['tokens'] for r in v]
  axes[0,col].plot(x,[r['gbps'] for r in v],marker='o',label=label,color=color)
  axes[1,col].plot(x,[r['median_ms'] for r in v],marker='o',label=label,color=color)
 for row in range(2):
  ax=axes[row,col];ax.set_xscale('log',base=2);ax.set_xlabel('Tokens per requester');ax.set_title(dist);ax.grid(True,alpha=.2);ax.spines[['top','right']].set_visible(False)
 axes[0,col].set_ylabel('Embedding GB/s per rank');axes[0,col].set_ylim(bottom=0)
 axes[1,col].set_ylabel('Full-fetch latency (ms)');axes[1,col].set_yscale('log')
axes[0,0].legend(fontsize=8)
fig.suptitle('8 concurrent requesters / 8 CPU owners; FP8 256 B rows; 2 layers x 24 rows/token\n20 rounds; max-rank full-fetch wall time; all slow rounds retained',fontsize=11)
for ext in ['png','svg']:fig.savefig(a.output/f'comparison.{ext}',dpi=180)
plt.close(fig)
for path in a.output.glob('*.svg'):path.write_text('\n'.join(l.rstrip() for l in path.read_text().splitlines())+'\n')
print('Summarized',len(rows),'configurations')
