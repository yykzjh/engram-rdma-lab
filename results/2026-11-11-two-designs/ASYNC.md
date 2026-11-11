# GPU 发起与异步 forward 验证

每rank连续排队8次：GPU producer生成本轮新索引 → fetch.enqueue(调用方stream) → GPU consumer逐行检查embedding和scales。循环中不做主机同步、原始索引D2H或TCP请求/完成通信；最后只同步一次并回收小型校验结果。应用Socket类的实际send/recv计数在整段forward期间增量为0。

| Tokens | C2C | enqueue中位数 µs | enqueue最大 µs | GPU fetch中位数 ms | 返回时末轮未完成 ranks | TCP消息 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0 | 2.448 | 8.928 | 0.057 | 8/8 | 0 |
| 1 | 1 | 2.896 | 10.560 | 0.055 | 8/8 | 0 |
| 4096 | 0 | 2.624 | 11.936 | 1.580 | 8/8 | 0 |
| 4096 | 1 | 2.624 | 8.672 | 0.873 | 8/8 | 0 |
| 131072 | 0 | 2.560 | 9.088 | 43.010 | 8/8 | 0 |
| 131072 | 1 | 2.880 | 9.888 | 24.264 | 8/8 | 0 |

所有GPU consumer输出与epoch正确。此表GPU event时延为诊断指标，与主表max-rank wall统计不同，不混合计算带宽。

异步机制：GPU打包索引 → GIN PUT写owner CPU注册邮箱并发布ready → owner CPU构造SGE/WQE → RDMA WRITE数据与每QP GPU DONE → GPU等待并重排。节点内数据C2C可独立开关。初始化仍使用MPI/TCP。

消除的是requester主机的逐次同步、原始索引D2H与TCP控制协议。索引仍会以GPU发起DMA送到owner CPU邮箱，owner CPU仍运行持久服务线程；不是完全零CPU的数据面。`GpuFetchGraph::enqueue()`本身无主机等待；同Buffer按同一stream顺序复用，后续GPU计算在该stream直接消费，跨stream用event同步。

源码：sge_write/gpu_graph.hpp与sge_write/async_validation.hpp。测试中按需回拷校验结果位于forward排队之后，不是通信推进条件。
