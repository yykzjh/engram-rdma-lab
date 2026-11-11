# UMR 目标映射验证

每个点20轮，取每轮8 ranks最大完整fetch时延的中位数。C2C打开；两侧使用同一候选二进制，只改变目标UMR开关。所有输出、scales与guards正确。UMR模式的GPU staging分配确认为0，并跳过GPU重排；映射全部由当前GPU路由位置动态产生，未假定owner槽位固定。

| 分布 | Tokens | staging+重排 ms | UMR直写 ms | UMR/原时延 | CPU映射关键耗时 ms | 使用MKeys/rank |
|---|---:|---:|---:|---:|---:|---:|
| stratified | 512 | 0.1844 | 0.4771 | 2.59x | 0.312 | 10 |
| stratified | 4096 | 0.9673 | 1.2377 | 1.28x | 0.382 | 14 |
| stratified | 131072 | 25.5734 | 30.7638 | 1.20x | 3.096 | 149 |
| uniform | 512 | 0.1844 | 1.4579 | 7.91x | 1.181 | 254 |
| uniform | 4096 | 0.9628 | 5.8569 | 6.08x | 3.889 | 1971 |
| uniform | 131072 | 25.5629 | 369.9878 | 14.47x | 76.147 | 54008 |

结论：本轮所有测点均变慢。131072 tokens省去每rank 1.5 GiB GPU staging，但随机分布的CPU映射关键耗时约76ms，已超过原完整fetch约25.6ms。按要求不在主分支保留该通路，也不提供第三套通信实现。

实现：初始化在requester侧为每个owner创建根MKey并交换rkey；每次GPU route/pack输出实际位置到CPU映射区，发布epoch；requester的CPU映射服务按当前位置合并实际连续段/重复stride并建立分层KLM，完成配置后GPU发送行索引请求；owner按连续逻辑地址SGE WRITE，NIC映射到最终GPU位置。owner归属每轮可变；MKey仅在上一轮GPU完成之后重映射。

本轮使用当前mlx5dv接口的inline配置，使用分层MKey覆盖大列表，并复用通用MKey资源池。当前索引的KLM、布局和映射每轮重建，构建/配置/READY等待均在fetch计时内。配置接口的inline要求见[rdma-core说明](https://github.com/linux-rdma/rdma-core/blob/master/providers/mlx5/man/mlx5dv_wr_mkey_configure.3.md)。这不是所有UMR实现的理论上限；本结论针对本次动态Engram布局和所测路径。

阶段数据见umr-summary.json及raw/umr。CPU build/config时间取各rank关键值，存在并行关系，不能将各列直接相加推算传输时间。
