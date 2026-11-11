# 历史 UMR / READ / WRITE 实验

这些代码保留方案演进过程，与根目录当前的精简 SGE WRITE 实现分开。当前主对照的固定配置与结果以根目录 README / configs / results 为准。

| 目录 / 入口 | 用途 |
|---|---|
| `umr_read_minimal/benchmark.cpp` | 最早的两端原语微基准：owner CPU 离散行经 KLM/UMR 变成连续区间，requester READ 到 GPU；对照 owner 本端 SGE WRITE。每组 24×256B，不代表完整 Engram 流量。 |
| `rdma_gather/benchmark.cpp` + `benchmark_impl.hpp` | 八 rank CPU 控制路径，真正按各 rank 分片词表处理原始 token 索引，比较 READ+UMR / WRITE+SGE 以及多个输出布局。 |
| `rdma_gather/planner*.hpp` | CPU 索引解码、分组、跨 token 合并、并行规划与分阶段 profiling。 |
| `rdma_gather/target_umr.hpp` | requester 目标映射实验：每 owner 根 MKey、子 MKey/KLM、动态目标布局与策略检查。 |
| `rdma_gather/gpu_request.cpp/.cu` + `gpu_graph.hpp` | GPU 生成并发送 owner 请求、GPU DONE 等待/重排；含独立 GIN GET、GPU 本地 CPU 直读等演进选项。 |

详细命令与每个历史参数见各子目录说明。历史 UMR 布局策略可能需要相应的硬件能力；保留不等于它优于当前 WRITE，也不代表所有组合均在本次重测。迁移清单与源码哈希位于 `provenance/`。平台部署脚本、节点地址、运行镜像和业务信息不放在公开仓库。
