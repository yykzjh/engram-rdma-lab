# Engram RDMA Lab

只保留两套 CPU 词表查询方案：

| 方案 | 测试入口 | 节点内 C2C 开关（默认 0） |
|---|---|---|
| DeepEP Engram READ：GPU 发起 GIN GET | `benchmarks/test_read.py` | `ENGRAM_READ_C2C=1` |
| SGE WRITE：GPU 发请求，CPU owner 提交 verbs WRITE | `build/sge_write/engram_sge_write` | `ENGRAM_WRITE_C2C=1` |

开关在初始化前设置。C2C 开启后，同机 GPU 直接读 CPU 词表；跨机仍走各自的 GET / WRITE。两套测试固定为两机、每机 4 GPU；2 层分片表（524288 / 524309 行/rank），每 token 每层 24 行，FP8 256B/行及两个 FP32 scales。

## 构建

需要 Linux、CUDA Driver/Toolkit/NVRTC、GPUDirect RDMA、mlx5 verbs/DevX、支持 GDAKI 的匹配 NCCL host/device 头文件与库、PyTorch、NumPy、Open MPI、CMake；DeepEP 使用支持 C++20 的 GCC 13。节点需支持 GPU 直接访问 CPU 内存，WRITE 的 GPU DONE 要求原生 GPUDirect OWNER ordering。

在两机相同绝对路径构建或同步源码与构建产物：

```bash
export NCCL_ROOT=/path/to/nccl
export EP_NCCL_ROOT_DIR="$NCCL_ROOT"
export CXX=g++-13 MAX_JOBS=4
(cd deepep && python setup.py build_ext --inplace)
cmake -S sge_write -B build/sge_write -DNCCL_ROOT="$NCCL_ROOT" \
  -DMPI_CXX_COMPILER=/path/to/mpi/bin/mpic++
cmake --build build/sge_write -j4
```

使用搬迁后的 HPC-X / Open MPI 时，在构建和运行前设置 `OPAL_PREFIX=/path/to/mpi`，使编译器 wrapper 能找到当前安装的配置文件。

两机分别加载已有的 CUDA/NCCL/MPI 环境，并设置（HCA、网口、GID、CPU affinity 按机器调整）：

```bash
export PYTHONPATH="$PWD/deepep"
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_HCA=mlx5_bond_0,mlx5_bond_1,mlx5_bond_2,mlx5_bond_3
export NCCL_IB_GID_INDEX=3 NCCL_CROSS_NIC=0 NCCL_GIN_TYPE=3
export NCCL_GIN_GDAKI_MAX_QP_RD_ATOMIC=8
export NCCL_GIN_GDAKI_MAX_DEST_RD_ATOMIC=16
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
```

使用 NCCL 实际检测的 G=4，无显式 min_stride 覆盖。128K 示例用 READ atomic=8；测试其他规模时，历史配置为 T=1 或 T≥131072 用8，其余用16。测试前自行确认 GPU 空闲，不与业务负载并行。

## READ 测试

两台机器同时运行，`MASTER_ADDR` 为节点 A 的可达地址；这里 `WORLD_SIZE=2` 表示节点数，程序在每节点启动4个进程。输出目录分别保存本机4个 ranks 的 JSON。

```bash
# 两机均设置
export MASTER_ADDR=NODE_A MASTER_PORT=23890 WORLD_SIZE=2
export ENGRAM_READ_C2C=1

# 节点 A
RANK=0 python benchmarks/test_read.py --tokens 131072 --rounds 20 \
  --distribution uniform --cuda-graph --output-dir /absolute/path/read-results

# 节点 B
RANK=1 python benchmarks/test_read.py --tokens 131072 --rounds 20 \
  --distribution uniform --cuda-graph --output-dir /absolute/path/read-results
```

去掉 `--cuda-graph` 可测普通 fetch/hook 路径。输入直接生成，每轮变化，无外部索引文件或 runtime 打包步骤。

## SGE WRITE 测试

在节点 A 执行；需要已有 SSH/MPI 凭据，两个节点的工作目录、二进制、旁边的 `gpu_request.ptx` 及运行库路径一致。MPI 的 `-x` 将当前环境传给远端；远端需能定位相同版本的 CUDA/NCCL/MPI 动态库。

```bash
export ENGRAM_WRITE_C2C=1 ENGRAM_WRITE_ASYNC_CHECK=0
mpirun -np 8 --host NODE_A:4,NODE_B:4 --map-by ppr:4:node --bind-to none \
  --wdir "$PWD" \
  -x PATH -x LD_LIBRARY_PATH \
  -x NCCL_SOCKET_IFNAME -x NCCL_IB_HCA -x NCCL_IB_GID_INDEX \
  -x NCCL_CROSS_NIC -x NCCL_GIN_TYPE \
  -x NCCL_GIN_GDAKI_MAX_QP_RD_ATOMIC -x NCCL_GIN_GDAKI_MAX_DEST_RD_ATOMIC \
  -x OMP_NUM_THREADS -x ENGRAM_WRITE_C2C -x ENGRAM_WRITE_ASYNC_CHECK \
  ./build/sge_write/engram_sge_write --ips NODE_A_IP,NODE_B_IP \
  --tokens 131072 --rounds 20 --distribution uniform
```

`--ips` 使用两机可达的 IPv4 地址。root 环境按 Open MPI 要求增加 `--allow-run-as-root`。以非默认 SSH 端口启动时增加对应 MPI SSH 参数。

同一入口的异步验证：将 `ENGRAM_WRITE_ASYNC_CHECK=1`、`--rounds 8`，其余命令相同。GPU producer → fetch → consumer 连续排队，最后只同步一次；检查每帧 embedding/scales、epoch 和 forward 期间 TCP 收发计数。默认模式逐轮同步计时和全量校验。

## 学习代码

建议按以下顺序阅读，关键处已补中文注释：

- `sge_write/gpu_request.cpp`：初始化、CPU 邮箱布局、owner 持久服务与 SGE 构造。
- `sge_write/gpu_request.cu`：GPU route/count/prefix/pack、发布 ready、等待每 QP DONE、C2C 与重排。
- `sge_write/rdma.hpp`：多 SGE WRITE 提交、各 QP 的 data → DONE 顺序。
- `sge_write/gpu_graph.hpp`：Graph 依赖、异步 `enqueue(stream)` 与同步测试包装。
- `sge_write/async_validation.hpp`：同 stream 上游/下游如何衔接。
- `deepep/deep_ep/buffers/engram.py`、`deepep/deep_ep/include/deep_ep/impls/engram/`：READ API 与 kernels。

WRITE 的 `enqueue` 只提交 Graph；同 Buffer 按同 stream 顺序复用，跨 stream 使用 event 或独立 Buffer。owner CPU 仍负责每轮 SGE/WQE，索引仍由 GPU DMA 写入 owner CPU 邮箱；初始化 TCP 不进入 forward。WRITE 当前提供独立 C++ 接口，尚未接入 Python EngramBuffer API。

## 测量口径

两套入口每轮使用相同的 MT19937 原始 uint32 序列（种子 `12345+rank*1009+epoch*97`），支持 `stratified` / `uniform`；数据与 scales 全量校验。默认预热1轮、测20轮。输入生成/上传、初始化和校验位于计时外；当前索引处理、请求、传输、DONE、重排与 scales 位于完整 fetch 内。

取每轮8 ranks最大 wall time，再取各轮中位数，并保留慢轮。带宽分子只计每 rank embedding 逻辑字节，含本地查询，scales成本计入时间；不是单 NIC 线速。WRITE 额外使用 owner CPU 服务线程与 GPU staging。

历史实验、性能原始样本和报告已从当前工作树清理，可在 Git 历史提交 `13a5e97` 查看。来源与许可见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) 和各 LICENSE。
