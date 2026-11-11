# Engram RDMA Lab

Engram 通信实验的独立归档：DeepEP Engram 的 GIN GET／同机 CPU 直读实现、GPU 发请求的 SGE WRITE，以及之前的 UMR / READ 实验。默认分支为 `main`。这不是完整 DeepEP，未包含 EP dispatch/combine、PP 或 Bucket 实现。

## 代码目录

| 路径 | 内容 |
|---|---|
| `deepep/` | 从固定 DeepEP 提交抽取的 Engram、必要的 Context/VMM/GIN/JIT 依赖与 Python API、原 Engram 测试 |
| `sge_write/` | 精简的 GPU 请求 + CPU owner 构造 SGEs + verbs WRITE + GPU DONE/重排实现 |
| `experiments/umr_read_minimal/` | 最早的 UMR READ / SGE WRITE 原语验证，24×256B 微基准，仅作历史归档 |
| `experiments/rdma_gather/` | 完整迁入的早期原型：普通 READ、READ + UMR、WRITE + SGE、目标 UMR、CPU/GPU 控制路径等 |
| `benchmarks/` | 调用真实 `deep_ep.EngramBuffer.fetch` 的数据、FP8 scales 与完整 fetch 校验 |
| `tools/` | 输入导出、隔离 runtime、两机 MPI 启动、固定配置矩阵、结果汇总 |
| [`results/2026-11-11/`](results/2026-11-11/) | 64 组 × 20 轮的完整结果、原始样本、带宽/时延曲线及 READ 波动说明 |
| `provenance/` | 上游提交、迁移清单与来源哈希 |

原 DeepEP 工作区中的实验目录已迁出；原工作区只保留用户的三处学习注释文件，不应用本仓库的功能修改。

## 两种当前实现

**READ：** 原 DeepEP Engram 的 GPU 内核负责查询索引。混合路径对同机 owner 使用 NCCL window 的本地 CPU 内存映射，通过 GPU load 直接写入最终输出；跨机保持 GIN GET。局部复制由多个 issue warp 分摊，默认 16 lanes/行。fetch 和 wait 共用配置，wait 不等待未发出的本地 GIN 请求。同机词表仍在 CPU，未复制成 GPU 模型。

`deepep/deep_ep/include/deep_ep/impls/engram/local_cpu_config.cuh` 中 `EP_ENGRAM_LOCAL_CPU_READ=0` 可作为保留 GIN GET 路径的对照；默认是 1。`EP_ENGRAM_LOCAL_COPY_DISTRIBUTED=1` 为默认的分摊实现。`tools/prepare_runtime.py` 自动生成独立版本，不修改已有安装。

**WRITE：** requester GPU 每次按当前原始索引计算 owner、计数、前缀和打包，跨机用 GIN PUT 将行索引送到 owner CPU 的 inbox。owner CPU 动态构建最大 30 个 256B SGE 的 WRITE；可跨 token 合并到一个 WR，写入 requester GPU 的 owner-major staging。数据 QP 的 DONE 到达后，GPU 重排到最终输出。同机部分直接读取 CPU 词表。固定目标缓冲区的注册信息在初始化交换，当前请求的行索引、行数和位置在每次 fetch 中生成和发送。数据面 WRITE 使用原生 verbs，GIN PUT 只承担请求控制面。WRITE 每 rank 使用 4 个远端 requester 的 CPU owner 服务线程，并需要额外的 staging / positions 工作区；READ 的每次查询不需要 CPU owner 服务线程。

一个逻辑请求发送全部索引；owner 内部以 1024 WR 窗口提交/排空 CQ，不逐窗口做网络 ACK。每个活跃数据 QP 发送最终 DONE，GPU 等待这一组 DONE。CPU 统计信息在 GPU 完成后回收，既不驱动本次完成，也不预先准备下一次索引的描述符。

**UMR / READ 历史原型：** 保留于 `experiments/rdma_gather`，包括 owner CPU 离散行到连续虚拟区间的间接 MKey，以及 requester 目标 UMR 等实验。这里是实验归档；本次主对照是上面的真实 DeepEP READ 与精简 WRITE。历史路径的参数与当前主对照不同，不能直接混用结果。

## 构建

需要 Linux、支持本路径的 CUDA GPU、GPUDirect RDMA、mlx5 verbs/DevX、CUDA Driver/NVRTC、NCCL GIN GDAKI、PyTorch、MPI、CMake，以及支持 C++20 `std::format` 的编译器。本次在两台 Grace/L20A、4 GPU/机、CX7 环境验证；不是任意 CUDA/NCCL 版本都可直接构建。

本次版本：CUDA Toolkit 13.2、PyTorch `2.12.0a0+5aff3928d8.nv26.05`、NCCL `2.32.3+cuda13.4`、GCC 13、OpenMPI。代码要求 NCCL 头文件与实际运行库匹配。

```bash
export NCCL_ROOT=/path/to/nccl
export EP_NCCL_ROOT_DIR="$NCCL_ROOT"
export CXX=g++-13
export MAX_JOBS=4
(cd deepep && python setup.py build_ext --inplace)
cmake -S sge_write -B build/sge_write -DNCCL_ROOT="$NCCL_ROOT" \
  -DMPI_CXX_COMPILER=/path/to/mpi/bin/mpic++
cmake --build build/sge_write -j4
python tools/prepare_runtime.py --package deepep/deep_ep --output /absolute/path/runtime
```

历史 UMR/READ 原型可独立编译：

```bash
cmake -S experiments/rdma_gather -B build/experiments
cmake --build build/experiments -j4
# 若还需要历史 GPU requester：配置时加入
# -DENGRAM_BUILD_GPU_REQUESTER=ON -DNCCL_ROOT="$NCCL_ROOT" \
# -DMPI_CXX_COMPILER=/path/to/mpi/bin/mpic++
```

每个节点都需相同的源码、已编译扩展、PTX、runtime 与绝对路径。MPI/SSH 使用已有凭据；脚本不修改系统网络、NCCL 安装或 GPU 上的其他程序。CPU affinity 与 `mlx5_bond_0..3` 来自本次机器拓扑，换机需要核对 `benchmarks/compare_deepep.py` 和 `sge_write/rdma_common.hpp`。

原 Engram 测试保存在 `deepep/tests/engram/test_engram.py`，包含原学习注释与所需通用测试工具；它的 profiler 计时不同于本次主对照。

## 复现主对照

实验固定 8 ranks、每机 4 ranks，所有 requester 同时向全部 8 个 owner 查询。两层各 rank 存储 524288 / 524309 行，每 token 每层查询 24 行，FP8 每行 256B，另有两个 FP32 scale。各测点的 Buffer `max_tokens=T`，两种实现采用相同容量；词表大小保持固定。`stratified` 复现 DeepEP 原测试的分段索引分布；`uniform` 覆盖随机 owner、空组及非连续槽位。

输入是上层提供的原始 GPU 索引。原生实现用 C++ `std::mt19937`，种子为 `12345 + rank*1009 + epoch*97`；以下导出器使用同一函数产生 DeepEP 所读文件。每轮更换索引，不保存当前索引的 owner 分组、SGE、重排表或远端地址计算结果。

WRITE harness 直接调用上述 C++ 生成函数产生上层输入，DeepEP 读取同一函数导出的文件。因此这里的 `--indices` 应使用本仓库导出器生成的固定场景文件；将文件替换成任意索引并不会同步改变 WRITE harness 的输入。实际通信路径每轮都重新读取 GPU 原始索引并完成动态路由。

```bash
python tools/export_inputs.py --generator build/sge_write/export_indices \
  --output /absolute/path/inputs --rounds 20
# 将 inputs/ 与 manifest.json 同步到另一节点的相同路径。
python tools/run_matrix.py --configs configs/comparison.json \
  --hosts NODE_A,NODE_B --ssh-port 22 --mpi-prefix /path/to/mpi \
  --mpirun /path/to/mpi/bin/mpirun --env-script /path/to/runtime-env.sh \
  --runtime-root /absolute/path/runtime \
  --write-binary /absolute/path/build/sge_write/engram_sge_write \
  --indices /absolute/path/inputs --output /absolute/path/new-results
python tools/summarize.py /absolute/path/new-results/*/results.json \
  --output /absolute/path/new-results/summary.csv
```

脚本固定 `NCCL_CROSS_NIC=0`，要求 NCCL **实际检测** `ginMinStride=4`，没有增加用户可指定的 min_stride。READ 的 QP 数与 depth 采用相同的 DeepEP 推荐规则；RD atomic 在相同 token 数的 READ 对照间一致。WRITE 的活跃数据 QP 为每个远端 owner 4 个，SGE 合并 30 行、窗口 1024 WR。READ 使用 G=4 custom-stride rail 连接，通过代表 rank 访问该组 owner 的 CPU 注册映射。WRITE 数据面在每个跨机 owner/requester 对之间建立独立 RC 通道；`NCCL_CROSS_NIC` 只控制它的 GIN 请求路径，不重写原生 verbs 数据 QP。两者的连接组织、传输调度与 CPU 资源消耗不同，不将差异全部归因于 READ/WRITE opcode。

## 计时和边界

主指标是每轮 **8 ranks 完整 fetch wall time 的最大值**，再取 20 轮中位数；同时保留均值、最大值与全部原始样本，不裁剪慢轮。带宽为每 rank 逻辑 embedding 字节数除以上述延迟，以十进制 GB/s 表示。它不是网卡单端口线速或 8 ranks 总带宽。scales 的处理在计时内，带宽分子只计 embedding。

DeepEP Graph 包含实际 fetch 和两个 layer hook；普通 Python API 路径额外包含 Python 发起与 hook 开销。WRITE 包含 GPU 分组、请求、owner 描述符/WQE、WRITE、最终 DONE、GPU 重排及 scales。输入生成/上传、模型写入、MR/QP、容量工作区、JIT、Graph capture 和校验在计时外，当前索引相关的实际通信准备工作在计时内。

大 token 的 READ 有明显轮间波动。请同时查看均值、最大值及原始数据，勿把少量 smoke 轮次当作稳定结论。此前 14.4 GB/s 的单 owner 流量场景也不能与这里 8 个 requester/8 个 owner 的全负载直接对比。

## 来源

DeepEP 代码与原学习注释来源见 `provenance/source-origin.json`。保留 DeepSeek MIT 许可证。DeepJIT 仅保留 Engram 构建所需 header 依赖，固定提交与来源见 `THIRD_PARTY_NOTICES.md`。历史实验迁移清单见 `provenance/migration.json`。
