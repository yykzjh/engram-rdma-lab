# Engram RDMA Lab

两套独立实现，均提供 DeepEP 风格的 EngramBuffer 调用：

| 方案 | Python 包 | 唯一测试入口 | EGM 开关（默认0） |
|---|---|---|---|
| READ / NCCL GIN GET | `deep_ep` | `benchmarks/test_read.py` | `ENGRAM_EGM_ENABLE` |
| GPU 请求 / CPU SGE WRITE | `engram_sge_write` | `benchmarks/test_write.py` | `ENGRAM_EGM_ENABLE` |

两个包不相互导入、链接或共享通信对象。WRITE 自己管理 CPU 词表、GIN 请求邮箱、verbs QP、每层 DONE 和 GPU 重排。

## CPU 词表路由

`ENGRAM_EGM_ENABLE=0/1` 同时控制机内和同超节点跨机的 EGM，默认 `0`。开启表示逐 GPU rank 对优先尝试 EGM；探测与协商在初始化时完成，之后在 Buffer 生命周期内固定路由。

| owner 位置 | EGM=0 | EGM=1 |
|---|---|---|
| 同一物理主机 | RDMA | owner CPU 内存 → owner C2C → GPU NVLink → requester GPU |
| 不同主机、同一 scaleup fabric 域 | RDMA | owner CPU 内存 → owner C2C → 超节点 GPU NVLink → requester GPU |
| 不同 scaleup 域/机柜 | RDMA | RDMA |

READ 的 RDMA 数据路径使用 NCCL GIN GET；WRITE 使用 GPU 请求邮箱与 owner CPU SGE WRITE。所有配置都只分配一份 CUDA VMM CPU 词表，EGM 映射与 RDMA 注册复用相同物理内存，权重仅在 `write()` 更新。

初始化通过主机 `boot_id` 区分物理节点，通过当前 CUDA GPU UUID 查询 NVML fabric cluster UUID/clique。跨机 EGM 要求双方处于同一 cluster/clique、完成 fabric 注册并可导出 HOST_NUMA FABRIC handle。未知域、不同域或缺少 fabric/IMEX 能力时走 RDMA，`get_route_info()` 报告逐 owner 的实际路由及能力信息。每条候选链路独立尝试句柄导入、映射或访问权限探测；任一方向失败，仅该 GPU rank 对统一回退 RDMA，其他链路继续使用 EGM。`get_route_info()["owner_fallback_reasons"]` 按 owner 给出映射失败方向和原因。开关不修改 Fabric Manager、IMEX 通道或驱动配置。

EGM 路由初始化分为三个阶段：探测每个 requester→owner 映射；全体交换结果，逐 rank 对取双向成功结果的交集；释放最终降级链路的临时别名，上传最终 GPU 路由并为 RDMA 边创建通道。成功映射直接复用，不重复导入。READ 中一个 GIN 分组可同时包含 EGM 与 RDMA owner，只有整个分组都直读时才跳过 GIN 完成描述符；WRITE 在最终路由确定后才创建匹配的数据 QP 和 owner 服务。

回退仅处理 EGM 数据路径的初始化失败。基础词表分配、GIN/邮箱/scales 必需映射、GPU 工作区或 RDMA 初始化失败仍按初始化错误处理；运行中的链路故障不进行热切换。路由协商不会新增 forward 中的主机等待或进程组通信。

WRITE 的直读路径用普通kernel融合GPU计数、16B词表读取和FP8 scales收集。逐owner路由表决定哪些行直接写最终输出、哪些行打包到RDMA请求；直接读取的owner不创建数据QP/CPU服务线程，不发送邮箱、不等待DONE。后续的打包/发送kernel仅对RDMA行发布请求，逐层hook等待相应QP的DONE并重排。route配置在Buffer生命周期内固定，Graph重放仍可动态改变查询owner和空层。

READ保持自己的GIN GET和逐层wait：直读owner由SM gather完成，其他owner继续GIN；仅全直读的GIN分组跳过完成描述符，部分直读分组仍正常flush/wait。GIN rail、请求邮箱及FP8 scales继续使用节点内LSA域；EGM跨节点别名独立导入CUDA FABRIC handle，不把超节点当成能共享POSIX fd的单主机。

`ENGRAM_WRITE_LOCAL_SF=1` 在每个 GPU 保留完整 FP8 scales 副本，在 `write()` 时更新；开启 EGM 时默认启用。关闭 EGM 时默认使用节点内分片 scales（`0`）。对照词表链路性能时可固定为 `1`，保持 scales 路径一致。

```bash
export ENGRAM_EGM_ENABLE=0  # 统一走RDMA
export ENGRAM_EGM_ENABLE=1  # 机内及同fabric域跨机优先走EGM
```

## 构建

需要 Linux、CUDA Driver/Toolkit、支持 GDAKI 的匹配 NCCL host/device 库、GPUDirect RDMA、mlx5 verbs/DevX、PyTorch、NumPy、GCC 13（C++20）。WRITE 不再依赖 MPI 或 CMake。两机源码、包和 PTX 路径一致。

```bash
export NCCL_ROOT=/path/to/nccl EP_NCCL_ROOT_DIR=/path/to/nccl
export CXX=g++-13 MAX_JOBS=2
(cd deepep && python setup.py build_ext --inplace)
(cd sge_write && python setup.py build_ext --inplace)
export PYTHONPATH="$PWD/sge_write:$PWD/deepep"
export LD_LIBRARY_PATH="$NCCL_ROOT/lib:${LD_LIBRARY_PATH:-}"
export NCCL_IB_HCA=mlx5_bond_0,mlx5_bond_1,mlx5_bond_2,mlx5_bond_3
export NCCL_SOCKET_IFNAME=eth0 GLOO_SOCKET_IFNAME=eth0 NCCL_IB_GID_INDEX=3
export NCCL_CROSS_NIC=0 NCCL_GIN_TYPE=3 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
```

WRITE 使用检测到的 GIN stride，要求它等于每节点参与的 GPU 数；进程组 ranks 按节点连续排列。可通过 `ENGRAM_WRITE_HCA` 指定本进程 HCA，默认 `mlx5_bond_<CUDA device>`。GPU DONE 要求原生 GPUDirect OWNER ordering。

## WRITE 接口

```python
import torch
import engram_sge_write as backend

entries, hidden, max_tokens, ept = [524288, 524309], 256, 4096, 24
dtype, packs = torch.float8_e4m3fn, 2
sizes = backend.EngramBuffer.get_storage_size_hint(
    group, entries, hidden, max_tokens, ept, dtype, packs)
config = backend.EngramBuffer.get_theoretical_config(group, len(entries), max_tokens, ept)
buffer = backend.EngramBuffer(
    group, *sizes, True, *config, explicitly_destroy=True)
buffer.set_config(entries, hidden, max_tokens, ept, dtype, packs)
buffer.write(storages, sfs)  # 各层本rank权重，以及各层全局scales

hooks = buffer.fetch(indices)  # CUDA int32 [layers,tokens,ept]；只提交发起路径
unrelated_compute()
data, scales = hooks[0]()      # 此处才提交第0层等待与重排
consume(data, scales)
data1, scales1 = hooks[1]()    # 也支持先调用第1层hook
consume(data1, scales1)
buffer.destroy()              # 所有ranks共同调用
```

BF16 使用 `dtype=torch.bfloat16, num_sf_packs=0`，hook 直接返回数据 Tensor；FP8 hook 返回 `(data, scales)`。数据 shape 为 `[tokens, hidden*ept]`。scales 默认按列存储、token stride=1、列 stride 对齐4；`use_tma_aligned_col_major_sf=False` 返回行主序。负索引作为padding，数据和scales填0。

`fetch()` 与 hook 均使用调用时的 CUDA stream，并支持外部 CUDA Graph capture/replay。跨 stream hook 通过事件依赖发起路径；普通执行复用工作区前等待上一轮各层重排。首次 capture 前须同步该 Buffer 的已有工作；Graph 内须连接所有 hook stream，并顺序 replay。调用方仍须保证旧输出的消费完成后才复用 Buffer。一轮所有 hook 均须调用，旧 hook 在新 fetch 后失效。初始化、写表、销毁允许主机同步；forward 无索引 D2H、主机等待或进程组通信。

`write()` 接收每层连续 `[本rank行数, hidden]` 权重；FP8 的全局 scales 为 `[world*本rank行数, num_sf_packs]`，在节点内分片、跨节点冗余。可写入真实 CPU/CUDA Tensor，不再在通信实现中构造测试权重。

兼容范围：CPU词表、BF16/FP8、可配置层数/行宽/词表/token容量、逐层hook、QP子集、scales两种布局。当前每 Buffer 初始化配置一次，支持最多64 ranks/32层/16数据QP；GPU词表和非hybrid拓扑会明确拒绝。`restrict_rd_atomic` 保留签名，WRITE 不使用 READ credit。容量和QP数须使用 WRITE 自己的 size/config helper，不沿用 READ 的资源估值。`get_comm_stream()` 返回当前 stream；辅助 `barrier()` 是同步主机/进程组 barrier，不能放进 Graph，不等同于 DeepEP 的 GPU barrier。核心 fetch/hook 调用方式对齐，内部 context 和通信资源不互换。

## 测试

两机各运行一次；`WORLD_SIZE` 是节点数，`RANK` 是节点编号0/1，各入口再启动4个本地GPU进程。两机先设置相同环境：

```bash
export MASTER_ADDR=NODE_A MASTER_PORT=23890 WORLD_SIZE=2
export ENGRAM_EGM_ENABLE=1
# 节点A用RANK=0，节点B用RANK=1；两机同时执行所选入口。
RANK=0 python benchmarks/test_read.py --tokens 4096 --rounds 20 \
  --distribution uniform --cuda-graph --output-dir /absolute/path/read-results
RANK=0 python benchmarks/test_write.py --tokens 4096 --rounds 20 \
  --cuda-graph --output-dir /absolute/path/write-results
```

依次运行两套测试，先确认GPU空闲。WRITE 同入口可选 `--async-check --rounds 8` 连续Graph forward；再加 `--async-eager` 则连续调用普通fetch/hook。`--hook-order reverse`、`--cross-stream`、`--row-major-sf`、`--padding` 分别检查逆序hook、跨stream、scales行布局与padding；`--dtype bf16 --hidden 128 --entries 4096,4103,4111 --ept 7 --capacity 8192` 检查不同shape与容量。普通测试去掉 `--cuda-graph` 使用Python API。两种异步验证循环内都只在GPU校验，最后一次同步后回收结果。
开启 EGM 后，两入口均可加 `--egm-fail-pair 0:4`，仅破坏 0→4 的 EGM 映射元数据，验证 0↔4 协商回退而其他链路继续走 EGM；改为 `0:2` 可测机内链路，选项可重复。故障注入只在测试入口包装原生探测方法，生产实现没有额外故障开关。

使用 `--async-check --locality-cycle --rounds 24` 可在同一个 Buffer/Graph 内由 GPU 循环切换纯机内、跨机、混合、空层和全 padding，检查空请求跳过后的序号与输出；可同时加跨 stream 和逆序 hook 选项。

WRITE 常规测试使用GPU动态索引。做性能对照时，为WRITE加 `--matched-read --distribution uniform --cuda-graph`，READ也使用 `--distribution uniform --cuda-graph`：两入口使用完全相同的MT19937动态索引、权重、scales及CPU绑定，每轮输入生成和上传均在计时外；Graph中只保留fetch和hook。完整fetch计时包含发起、等待、重排与scales，初始化/上游输入/校验在计时外；逐轮结果按rank保存，比较时取同轮最慢rank并保留慢轮。有效embedding带宽按每rank返回字节数除以该时延计算，不代表网卡线速。

采集WRITE时间线使用同一入口加 `--profile --rounds 3`，默认只对全局rank 0采集CPU/CUDA事件，输出 `write-sge-r0.trace.json`，可用Perfetto打开。普通模式标注发请求、独立计算和每层DONE等待/重排；加 `--cuda-graph` 可采集Graph replay。初始化、预热与最后一轮数据校验在采集区间外，区间内连续提交forward、最后统一同步。Torch trace展示CPU提交和GPU kernel；后台verbs服务与网卡活动未单独埋点。

流量拓扑默认 `--traffic all-to-all`，8个requester分别从全部8个owner词表均匀采样。单点对照使用 `--traffic single --requester-rank 0 --owner-rank 4`，WRITE同时加 `--matched-read`：仅rank 0执行fetch/hook，全部索引限定为owner 4的行；其他rank只参与初始化、CPU控制组会合并保留owner服务，不发送占位fetch。单点时延只统计活跃requester；空闲rank记录零字节、空时延及零forward提交数。0和4跨机：EGM关闭或不在同fabric域时走RDMA；同域且EGM开启时走EGM。WRITE只为有RDMA数据的owner/层发送请求并等待DONE。
将owner改为 `--owner-rank 2` 可测0→2机内查询；使用 `--traffic single-requester --requester-rank 0` 可测仅rank 0均匀查询全部8个owner。四种场景的tokens均指每活跃requester，返回字节数为 `2*tokens*24*256`；全对全一轮总输出为单requester场景的8倍。

## 代码结构与执行流程

两套后端按相同职责分层，但各自实现，不相互导入或链接。

| 职责 | READ | WRITE |
|---|---|---|
| Python API、配置与逐层 hook | `deepep/deep_ep/buffers/engram.py` | `sge_write/engram_sge_write/buffer.py` |
| 域识别、配置解析、逐 owner 路由 | `deepep/deep_ep/utils/engram_routing.py` | `sge_write/engram_sge_write/routing.py` |
| CPU 词表分配与 EGM 映射 | `deepep/csrc/buffers/engram_storage.hpp` | `sge_write/csrc/storage.hpp` |
| 原生运行时与资源生命周期 | `deepep/csrc/buffers/engram.hpp`、`engram_routes.hpp` | `sge_write/csrc/runtime.cpp` |
| GPU 数据面 | `deepep/deep_ep/include/deep_ep/impls/engram/` | `sge_write/csrc/gpu_request.cu` |
| owner CPU 服务与数据 QP | NCCL GIN | `sge_write/csrc/owner_service.hpp`、`rdma.hpp` |

WRITE 初始化依次建立词表/邮箱、交换域与句柄、逐链路探测映射、交换双向结果并提交最终路由、连接 RDMA QP 和 GIN，再启动 owner 服务线程。`protocol.hpp` 定义 GPU/CPU 共用的 64B 邮箱头及行号布局。

- `fetch()`：GPU 根据逐 owner 路由直接收集 EGM 行，把 RDMA 行号打包并发布到 owner 邮箱，随后异步返回 hooks。
- owner 服务：按 requester/层轮询邮箱，根据行号构造 SGE，经数据 QP 写回 requester；每个用过的 QP 在本层数据之后发布自己的 DONE。
- 逐层 hook：向当前 stream 提交 DONE 等待与重排，随后返回输出视图；独立计算可插在 fetch 与 hook 之间。
- CUDA Graph：同 stream 的 kernel 顺序建立 issue 内部依赖，事件连接跨 stream hook；重放时 GPU 动态生成请求与 epoch，CPU 不缓存查询相关的 SGE。
- 销毁：等待在途 GPU 工作，停止并回收 owner 线程，解除通信注册和 importer 映射，最后释放 owner 分配。输出 Tensor 持有 WRITE Runtime，保证视图引用的内存仍然有效。

源码中保留邮箱发布、SGE 构造、每 QP DONE 和 Graph 节点依赖的中文注释。两个测试入口在 `benchmarks/`，构建产物和测试结果写入独立输出目录。

历史完整Graph WRITE版本及测试报告可从Git提交 `54c16b7` / `13a5e97` 追溯。来源与许可见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) 和各 LICENSE。
