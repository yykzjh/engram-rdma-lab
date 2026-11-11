# 最早的 UMR READ / SGE WRITE 原语验证

这是早期 `24 行 × 256B` 单 owner/requester 微基准的原始源码归档，用于解释与验证间接 MKey/KLM、UMR 配置、READ 与本端 SGE WRITE。它不表示后续的 8-rank Engram 实际流量场景；当前性能结论使用根目录的 DeepEP / SGE WRITE 对照。

```bash
cmake -S experiments/umr_read_minimal -B build/umr_read_minimal
cmake --build build/umr_read_minimal -j4
# owner 节点先启动，requester 节点再启动，选择空闲 GPU 和 HCA。
build/umr_read_minimal/umr_read_minimal --owner --device mlx5_bond_0 --port 39143
build/umr_read_minimal/umr_read_minimal --host OWNER_HOST --device mlx5_bond_0 --gpu 0 --port 39143
```

默认 4096 组、24 行/组、256B/行、pool=1024、4 QPs。两端参数必须一致。程序依次执行 READ+UMR 与 WRITE+SGE，按窗口发送请求/响应/ACK，每轮校验完整输出和 guards；打印 `layout_ms`、`umr_ms`、`umr_post_ms`、`data_ms`、控制面与端到端耗时。这些阶段与当前完整 GPU fetch 的计时边界不同。

源码从早期测试成果保留，不包含节点地址或平台部署脚本。
