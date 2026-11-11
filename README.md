# Engram RDMA Lab

本仓库只保留两套通信设计：**原 DeepEP Engram READ** 与 **SGE WRITE**。两者均可通过环境变量启用节点内 GPU 直接读取 CPU 词表的 C2C 路径。跨机分别使用 NCCL GIN GET 与原生 verbs SGE WRITE。

| 设计 | 环境变量 | `0`（默认） | `1` |
|---|---|---|---|
| DeepEP Engram READ | `ENGRAM_READ_C2C` | 保持 GIN GET 查询路径 | 同节点 CPU 词表由 GPU 直接 load，跨节点保持 GIN GET |
| SGE WRITE | `ENGRAM_WRITE_C2C` | 所有 owner 均通过 SGE WRITE 返回数据 | 同节点由 GPU 直接 load，跨节点由 CPU owner SGE WRITE |

开关在 Buffer / 通信对象初始化前设置，生命周期内固定。READ 的选择写入 JIT 源码，因此 fetch 和 wait 始终一致，两个值共用同一个 runtime / JIT cache，无需修改头文件或准备多份安装。C2C 开关只改变词表数据路径；FP8 scales 继续使用相同的节点内 GPU 分片布局。

旧独立 GET、CPU 索引规划器、READ+UMR、逐 token WRITE、早期原语等原型代码已从当前主分支移除。历史版本与测试结果仍可从 Git 提交 `91a6eba` 追溯。原 DeepEP 工作区只保留用户的三个学习注释文件，所有功能开发位于本仓库。

## 目录

- `deepep/`：Engram API / kernels，以及所需的 Context、VMM、GIN、JIT 通用依赖；不包含完整 EP/PP/Bucket 实现。
- `sge_write/`：GPU 发请求、CPU owner 构造 SGE WRITE、GPU 等待 DONE 和重排。
- `benchmarks/`、`sge_write/async_validation.hpp`：完整正确性与异步 producer → fetch → consumer 验证。
- `tools/`：两机启动、共用输入生成、runtime 打包和结果统计。
- `results/`：原始样本、完整 fetch 时延、带宽与验证报告。
- `provenance/`：上游版本、构建、迁移与源码哈希记录。

## WRITE 的 GPU 控制路径

每次 fetch：

1. requester GPU 从当前原始索引计算 owner、计数和前缀，打包当前行索引与输出位置。
2. GPU 用 GIN PUT 将行索引和请求元数据写入远端 owner 的 CPU 注册邮箱，并发布 ready 信号。节点内控制邮箱通过映射指针访问。
3. owner CPU 持久服务线程读取邮箱，为当前请求构造最多 30 个 SGE 的 WRITE。可合并同一 requester/owner 的多个 token，但不缓存索引相关的 SGEs。
4. owner 完成数据后，通过每个活跃数据 QP 的 RDMA WRITE 发布 requester GPU DONE。
5. requester GPU 等待全部 DONE，将 owner-major staging 重排到最终输出。C2C 打开时，本地行直接写最终位置。

初始化使用 MPI/TCP 交换地址、rkey 和 QP 信息；**forward 不使用 TCP 请求/DONE/ACK，也不下载原始 GPU 索引到 requester CPU**。owner CPU 仍负责生成、提交数据 WQE；索引仍会由 GPU DMA 到 owner CPU 邮箱。这里的异步 GPU 控制不等于取消 owner CPU 数据服务。

每个 owner 的内部发送窗口用于限制 WR/CQ 容量，没有逐窗口网络 ACK。容量缓冲区、QP/MR、模型、Graph 在初始化建立；当前索引的路由、行列表、SGEs 和重排位置在每次 fetch 中重新产生。

## 异步 forward 接口

`GpuFetchGraph::enqueue(CUstream inference_stream)` 只将 fetch Graph 排到调用方的 CUDA stream，不执行主机等待、索引 D2H、TCP 通信或用户级资源分配。后续依赖输出的 GPU kernel 排在同一 stream 即可；跨 stream 使用 CUDA event 建立依赖。

```cpp
// Buffer、词表、通信和 Graph 已在初始化完成。
fetch.enqueue(inference_stream);
// 将模型中消费 embedding 的 kernel 排到 inference_stream。
// 如需在另一 stream 消费，先 record event，再由该 stream wait event。
```

同一 Buffer 的输入、输出和工作区按 stream 顺序复用；可连续异步排队多个 forward。跨 stream 并发需要独立 Buffer 或外部依赖管理。测试 harness 的计时/校验等待位于 enqueue 接口之外。异步测试在 GPU 生成不同索引，并在 GPU 逐帧校验 embedding 和 scales，最后只做一次主机同步。

## 构建与运行

需要 Linux、CUDA Toolkit / Driver / NVRTC、GPUDirect RDMA、mlx5 verbs/DevX、匹配的 NCCL host/device GDAKI、PyTorch、MPI、CMake 和支持 C++20 `std::format` 的编译器。本次环境为两机 L20A / Grace，每机 4 GPU，CX7、MTU 4096、GID 3；CPU affinity 和 HCA 名称需按机器调整。

```bash
export NCCL_ROOT=/path/to/nccl
export EP_NCCL_ROOT_DIR="$NCCL_ROOT"
export CXX=g++-13 MAX_JOBS=4
(cd deepep && python setup.py build_ext --inplace)
cmake -S sge_write -B build/sge_write -DNCCL_ROOT="$NCCL_ROOT" \
  -DMPI_CXX_COMPILER=/path/to/mpi/bin/mpic++
cmake --build build/sge_write -j4
python tools/prepare_runtime.py --package deepep/deep_ep --output /absolute/path/runtime
python tools/export_inputs.py --generator build/sge_write/export_indices \
  --output /absolute/path/inputs --rounds 20
```

两机需具有相同的源码、二进制、PTX、runtime 与绝对路径；输入文件也同步到两机。脚本使用已有 SSH/MPI 凭据并检查 GPU 空闲，不停止其他作业或修改系统/NIC 配置。

```bash
python tools/run_matrix.py --configs configs/comparison.json \
  --hosts NODE_A,NODE_B --ssh-port 22 --mpi-prefix /path/to/mpi \
  --mpirun /path/to/mpi/bin/mpirun --env-script /path/to/runtime-env.sh \
  --runtime-root /absolute/path/runtime \
  --write-binary /absolute/path/build/sge_write/engram_sge_write \
  --indices /absolute/path/inputs --output /absolute/path/new-results
```

独立运行可用 `ENGRAM_READ_C2C=1` / `ENGRAM_WRITE_C2C=1`，或启动工具的 `--c2c 1`。READ 默认原路径。`tools/run_write.py --async-check --rounds 8 ...` 执行连续异步 forward 验证。

WRITE harness 直接调用 C++ `mt19937` 生成上层输入，DeepEP 读取同一函数导出的文件。种子 `12345 + rank*1009 + epoch*97`。`--indices` 用于这个共用生成器的固定场景，不接受任意文件替换后仍假定 WRITE 输入相同。异步验证另用 GPU producer 动态生成输入并由 GPU consumer 检查。

## 测量口径

8 ranks 同时作为 requester，查询全部 8 个 CPU owner 的分片词表。两层每 rank 分别存储 524288 / 524309 行，每 token 每层查询 24 行，FP8 256B/行，另有两个 FP32 scale。`stratified` 为原 DeepEP 分段索引分布；`uniform` 允许任意 owner、空组和不连续输出槽位。

主指标为每轮 8 ranks 完整 fetch wall time 的最大值，再取 20 轮中位数；保留全部慢轮、均值、最大值。带宽分子为每 rank 逻辑 embedding 字节，使用十进制 GB/s，包含本地查询，不能当作网卡线速。scales 成本包含在时间中。各测点两套方案的 `max_tokens=T`，词表大小固定。

所有路由、请求、owner SGE/WQE、传输、DONE、重排和 scales 都在 fetch 内。模型写入、容量分配、QP/MR、JIT、Graph 初始化、上游输入生成/上传及结果校验不计入 fetch。

固定 `NCCL_CROSS_NIC=0`、GDAKI，NCCL 实际检测 G=4，无用户 min_stride 覆盖。READ 使用 custom-stride rail 和 owner CPU 注册别名；WRITE 为 owner/requester 对建立 RC 数据通道，CROSS_NIC 仅约束它的 GIN 控制面。WRITE 每 rank 使用 4（C2C 开）或 8（C2C 关）个 CPU owner 服务线程和额外 staging。该对照包含整个设计与资源差异。

## 本轮验证结果

完整数据见[两套设计与 C2C 开关对照](results/2026-11-11-two-designs/)、[UMR 验证](results/2026-11-11-two-designs/UMR.md)、[异步 forward 验证](results/2026-11-11-two-designs/ASYNC.md)。

131072 tokens、随机 owner 归属、20轮 max-rank 完整 fetch 中位数：

| 设计 | C2C关 | C2C开 |
|---|---|---|
| READ | 201.91 ms / 7.98 GB/s | 105.91 ms / 15.21 GB/s |
| SGE WRITE | 44.17 ms / 36.46 GB/s | 25.56 ms / 63.01 GB/s |

UMR测试确实省掉每rank 1.5 GiB staging并取消GPU重排，但131072 tokens的分段/随机场景分别变为30.76/369.99ms，原路径约25.57/25.56ms。全部所测规模均无优势，当前主分支不保留UMR实现。随机布局约使用54k MKeys/rank，动态映射关键耗时约76.15ms；完整fetch包括这部分成本。

异步测试每rank连续排队8个forward，6组共384个GPU消费样本全部正确。enqueue中位数2.45–2.90µs，所有rank返回时末轮GPU仍在执行，实际应用Socket收发增量0，无逐forward主机同步、无requester原始索引D2H。owner CPU数据服务仍然存在。

## 来源与许可

上游 DeepEP / DeepJIT 提交见 `provenance/source-origin.json`；保留 DeepSeek MIT 许可证与 `THIRD_PARTY_NOTICES.md`。当前原 DeepEP 学习注释保留情况见 `provenance/original-worktree-verification.json`。
