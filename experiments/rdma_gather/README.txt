Historical experiment archive

Migrated from DeepEP/tests/engram/rdma_gather. This document describes several
earlier variants; see the root README for the current minimal implementations
and current fixed-configuration comparison. Machine-specific operational reports
are retained separately in local evidence and are not part of this public archive.

Eight-rank Engram communication prototype and full-fetch comparison

Build on the CUDA/RDMA ARM64 test image:
  cmake -S experiments/rdma_gather -B /tmp/engram-rdma-build
  cmake --build /tmp/engram-rdma-build -j4
Requires libibverbs, libmlx5/DevX, CUDA driver/NVRTC, GPUDirect RDMA, four GPUs and
four mlx5_bond HCAs per node. This is a standalone experiment, not a modification
to DeepEP/NCCL transport. Data WRITE uses verbs; compare_deepep.py uses GIN GET,
and the optional GPU requester uses GIN PUT for control messages.

Ranks and model:
Two nodes with ranks 0..3 on A and 4..7 on B. Each rank owns a CPU vocabulary
shard with two layers: 524288 and 524309 entries, 256B/entry (FP8 hidden=256).
Each token queries 24 entries/layer, potentially from all eight owner shards.
Every rank also queries concurrently into its own GPU. FP8 scales are two FP32
packs/row, sharded across the four local GPUs exactly as in DeepEP (two global
owners/shard). CUDA IPC provides local peer access to those scale shards.

Use --ips NODE_A,NODE_B. Start eight owner processes first, then eight requesters,
with the same parameters and output-layout flags on both ends. Example rank 0:
  engram_rdma_bench --owner --rank 0 --ips NODE_A,NODE_B --tokens 131072 --pool 1024 --qps 4 --rounds 10 --methods write --layout direct --port 26500 --indices-dir DIR
  engram_rdma_bench         --rank 0 --ips NODE_A,NODE_B --tokens 131072 --pool 1024 --qps 4 --rounds 10 --methods write --layout direct --port 26500 --indices-dir DIR
Repeat for ranks 1..7 on the appropriate nodes. --pool is the control/data window
length in WRs; WRITE-only mode with target_umr=none allocates no UMR MKeys. Eight peer channels share
one physical owner CPU table, rather than copying the table for every peer.
Default port is 39300; --port 26500 is preferred for these coordinated runs.
Each pair uses base+owner*8+requester; a global start barrier uses base+200.
Control messages use plain TCP on the private test network, with a homogeneous
little-endian, same-binary wire format. Do not expose this test service publicly.

Affinity matches the L20A topology: owner thread bases 0,16,72,88 and requester
bases 32,48,104,120, eight cores/rank. Adapt pin_core() for another CPU topology.
All intra-node and cross-node embedding payload uses verbs in this prototype;
DeepEP retains its normal G=4 hybrid/intra-node access path for the comparison.

Three WRITE variants (--layout):
direct: each (layer,token,owner) contiguous run is one WRITE with local SGEs,
writing straight into the final [layer,token,slot] destination. No GPU staging,
permutation upload, or embedding reorder kernel. For the original Engram index
rule, owners are monotonically ordered within each token, so each owner has one
run. For arbitrary interleaved owner slots it splits the run into several WRs
instead of overwriting another owner's slots. No WR crosses a token boundary.
staged: each nonempty (layer,token,owner) group is one WRITE into compact
owner-major staging, followed by GPU scatter into [layer,token,slot]. It does NOT
combine multiple tokens into a data WR. Thus direct versus staged isolates the
receive-layout/reorder choice under the original index rule.
merged: same requester/owner's rows across multiple tokens are combined into up
to --merge-rows local SGEs per WRITE (default/max 30, from advertised max_sge).
Rows target compact owner-major staging, followed by the same GPU reorder as
staged. Combining across tokens and descriptors is rebuilt inside every fetch.
One WR can be larger than the path MTU; e.g. 30*256B=7680B is two data packets
with a 4096B MTU, not a single 7680B wire packet. Do not confuse SGE/WR/packet.

Index semantics:
--distribution stratified reproduces tests/engram/test_engram.py:
  interval = floor(8 * entries_per_rank / 24)
  global_index[token, slot] = slot * interval + randint(0, interval)
Owner=floor(global_index/entries_per_rank). Since each slot's interval follows
the previous one, global indices and owner IDs are monotonic within a token.
There are typically three rows/owner, with occasional two/four-row groups.
--distribution uniform instead samples the whole global vocabulary, exercising
empty groups, variable lengths and interleaved owner slots.
--indices-dir stores/loads UNPROCESSED uint32 [2,tokens,24] global indices as
indices-rRANK-tTOKENS-DISTRIBUTION.bin. Different ranks have independent seeds.
The same files are read by both backends; no owner grouping is stored in files.

Actual fetch boundary:
Inputs are ordinary original GPU index tensors. Input generation/upload is
outside timing for BOTH backends, just as when previous application layers
provide their GPU results. Each native timed fetch:
  launch current-input FP8 scale gather from the four local GPU shards;
  download original GPU indices into pinned host workspace;
  recompute owner/local-row decoding, counts, group offsets and descriptors;
  build/upload the fresh output permutation if staging is needed;
  send this fetch's row indices and destination offsets to each owner;
  owner builds local SGEs and performs RDMA WRITE, drains all its data QPs;
  owner sends DONE; requester waits for all owners, then restores output order
  when needed and waits for all GPU outputs/scales to be ready.
The persistent destination base address/rkey is exchanged at Buffer setup;
current-input destination offsets and row requests are sent on every fetch.
The bounded protocol sends an ACK per window for flow control. Requests, DONE,
ACKs, native WR construction/allocation/posting and CQ waits are all timed.
GPUDirect write-ordering support is queried; a flush is issued when required.

Initialization is limited to model tables/scale shards, registered output memory,
QP/CQ/PD and Buffer-owned capacities derived from max_tokens/layers/world/EPT.
No current-input owner IDs, groups, SGEs, merged row lists, permutations or remote
addresses are prepared or cached by index identity. Direct omits unnecessary
staging/permutation memory. Grouping uses flat arrays and two parallel passes,
not millions of per-token vectors. Persistent workers reuse the same eight cores
already assigned to the requester. expected-reference buffers are validation
harness storage and never participate in actual fetch planning or transmission.

Timing fields:
e2e_ms / including_plan_ms: the SAME full GPU-input-to-GPU-output wall time.
index_download_ms: original GPU indices -> pinned CPU memory.
plan_ms: current-input decode/count, prefix, fill and optional token coalescing.
map_upload_ms: current-input GPU permutation upload (zero for direct).
fetch_ms: TCP request/notification + owner descriptors + RDMA WRs/CQ/ACK.
reorder_ms: embedding scatter plus final GPU completion; direct has only sync.
Eight peer paths overlap. response_wait includes owner work, so do not add it
again to owner times. data_ms includes data-WR construction/post/poll, not pure DMA.
Initialization/JIT, input generation, global start barrier, validation-only clears
and validation itself are excluded. These are not input-dependent preprocessing.
Every warmup and timed sample validates ALL embedding bytes, ALL FP8 scales and
output guards. The main bandwidth numerator is embedding bytes (1.5GiB/rank for
128K tokens); an additional 48MiB/rank scale output is disclosed separately.
These are useful-output bandwidths, not single-NIC wire bandwidths.

DeepEP comparison:
compare_deepep.py reads the identical eight raw files and constructs the same
vocabulary bytes/scales. Its timer encloses buffer.fetch(original_gpu_indices),
all layer hooks and torch.cuda.synchronize(). It verifies complete embedding
bytes and scale outputs. DeepEP uses its valid recommended GIN-context count,
32768 queue depth and default detected G=4; READ atomic 8 is the primary baseline,
16 is a sensitivity test. A GIN context count is not the same parameter as the
native number of verbs QPs/owner. Parameters and CPU resources are disclosed.
Native uses eight CPU owner service threads/rank as well as requester workers;
DeepEP's GPU-initiated GET does not require that CPU owner service.

Example in the existing verified DeepEP environment (usual two-node env vars):
  python compare_deepep.py --tokens 131072 --rounds 10 --indices-dir DIR --output-dir OUT
No package upgrades or shared-runtime modifications are required.
--planner-profile runs CPU-only profiling and exact legacy equivalence checks.
It also changes indices between rounds. The diagnostic per-token clock/counter
breakdown has overhead; uninstrumented legacy_full timing is reported separately.

The implementation is a prototype for the tested positive-index, two-layer,
24-EPT configuration. It does not implement the complete public DeepEP Python
Buffer API or every padding/inactive-rank case. Current tests measure full fetch,
not layer-by-layer overlap. RDMA resources and CUDA allocations are freed on
normal exit; test only with available GPUs. Existing business jobs are not killed.

WRITE tuning controls (2026-11-10):
--engine baseline retains classic per-window user WR allocation/construction.
--engine reuse keeps capacity and invariant SGE length/lkey fields, rebuilding
all input-dependent addresses, offsets, counts and WR links inside each fetch.
--engine ex uses ibv_wr_start/rdma_write/set_sge_list/complete to construct the
provider WQEs directly. Its verbs_post_ms includes both direct WQE construction
and submission; compare build+post totals with classic verbs, not wr_build alone.
--pipeline 0 retains per-window DONE/ACK stop-and-wait. Values 1..16 are WRITE-only:
1 removes the redundant requester ACK after owner CQ completion; larger values
allow that many request windows in flight on each existing peer TCP channel.
The owner still drains every data window before DONE and before reusing source
SGE storage. The requester consumes every DONE before GPU reorder/completion.
No pending READ/MKey lifetime rule is relaxed; READ keeps its ACK protocol.
--fast-plan 1 (merged only) removes per-token diagnostic counts/histograms that
merged transport does not consume, and uses flat current-index decode/fill.
It still recomputes per-owner row counts, local row addresses, output positions
and all merged groups on each fetch. histogram_recorded=false marks the omitted
histogram; zero histogram bins in this mode are not measured distributions.
All these flags must match the two endpoints; the private hello ABI is version 3.
Additional owner timing: wr_build_ms includes classic user WR allocation/fill;
verbs_post_ms is provider WQE construction/submission; cq_wait_ms is remaining
completion wait AFTER posting, not the entire NIC transfer latency. Data can
progress while WQEs are built/posted. Peer statistics sum batches on one channel;
eight channels and eight ranks overlap, so do not sum them as fetch wall time.
--compact-qp 1 sizes each data SQ for ceil(window_WRs / QPs) plus headroom,
rather than reserving the whole window on every QP. WRITE-only QP0 no longer
requests UMR configuration/inline capabilities. READ+UMR keeps QP0 large enough
for the entire MKey-configuration batch. These are capacity/resource choices,
not current-input preprocessing. They allow a 4096-WR data window without the
previous oversized allocation failing QP creation. Data windows are still
fully drained before owner workspace/QP capacity is reused.
--aligned-counts 1 separates each planner worker's writable counters using
128-byte alignment. It does not change input grouping, row order or wire data.
It is independent of --fast-plan, allowing the two effects to be tested separately.
The compact merged planner omits diagnostic histogram work; it never omits
current-index owner/local-row decoding, bounds checks, routing or output maps.

Measured recommendation (full eight-rank stratified fetch, 2026-11-10):
  --methods write --layout merged --merge-rows 30 --pool 1024 --qps 4 --fast-plan 1 --aligned-counts 1
This retains classic verbs and serial DONE/ACK (the defaults), with no extra
preprocessing or index-based cache. It measured 31.136 GB/s/requester versus
28.797 for the matched original 30-SGE planner, 20 rounds/config in reversed
blocks. The MTU-aligned optimized 16-SGE version measured 30.085 GB/s.
--quick-ack 1 requests TCP_QUICKACK after each exact receive; --coalesce-control 1
uses MSG_MORE for request headers/group lists and nonfinal ACKs. These are
experimental and OFF by default: the 20-round combined setting worsened mean
and tail latency without improving the median. Do not assume a long ACK wait
proves TCP delayed ACK; it also includes peer work and scheduling/resume time.
owner_ack_ms and owner_notify_ms report the previous window's ACK wait and DONE
send through the next DONE, avoiding another control round trip. Their sums
exclude the final window's post-response work, which the requester does not
wait for. Last-window data completion itself remains included and mandatory.
Standalone native WRITE was verified in this tuning; no formal DeepEP Buffer
API integration or NCCL GIN PUT implementation is claimed.
Full tuning report and raw data:
  <LOCAL_ARTIFACT>

Combined experiments (2026-11-11)
-------------------------------
The native implementation is now shared in benchmark_impl.hpp. benchmark.cpp is
its standalone launcher; gpu_request.cpp/.cu add an eight-process MPI launcher.
The existing DeepEP runtime and its transport are unchanged.

Native flags, identical on both endpoints:
  --protocol frame: send all current row indices in one logical request. Owner
    still submits/drains bounded --pool batches, without window DONE/ACK traffic.
  --compact-request 1: merged/frame only; owner derives canonical group lengths
    and packed offsets instead of receiving redundant Group descriptors.
  --done-gpu 1: each data QP publishes its final epoch into requester GPU memory.
    GPU acquire-loads ALL owner/QP epochs before consuming/reordering data. TCP
    responses carry benchmark telemetry after GPU completion, outside e2e.
    Requires queried native GPUDirect OWNER write-ordering support.
  --target-umr strict|adaptive: one persistent logical root MKey per owner and
    requester, with a bounded generic pool of --umr-keys (default 128) root/child
    keys. Every fetch builds its mapping from actual current output positions,
    configures children before root. The root rkey is exchanged ONCE at setup;
    runtime requests carry row indices/count/epoch and indirect-versus-fallback
    mode, not a newly exchanged rkey.
    No output slot has a fixed owner. Equal-stride compression is discovered
    from actual runs, including rare 2/4-row cases. Arbitrary uniform inputs are
    correct too. strict fails clearly if metadata exceeds pool/depth; adaptive
    uses ordinary staging+GPU scatter for that WHOLE fetch when any owner fails
    to fit. Root count does not imply only one physical MKey or one KLM entry.
    Reuse waits for previous WRITE completion and GPU consumption. Concurrent
    outstanding fetches would need separate roots/pools/output slots.
  --vary-indices 1: upstream inputs change every round before the timed fetch.
    All owner routing, destination mapping, requests and transfer remain timed.
  --extended-sge 1: explicit CX7 experiment allowing merge-rows up to 52. This is
    above advertised max_sge=30 and MUST pass actual QP capacity/transfer checks
    on the target system; it is not a portable default.

GPU requester build (optional):
  cmake -S experiments/rdma_gather -B /tmp/engram-rdma-build \
    -DENGRAM_BUILD_GPU_REQUESTER=ON \
    -DNCCL_ROOT=/usr/local/lib/python3.12/dist-packages/nvidia/nccl \
    -DMPI_CXX_COMPILER=/usr/local/mpi/bin/mpic++
  cmake --build /tmp/engram-rdma-build -j4
Requires MPI and matching NCCL host/device headers and library (tested 2.32.3),
CUDA 13 toolchain, Linux CUDA VMM HOST_NUMA/exportable FD support, and permission
for pidfd_getfd between the four local ranks. The PTX target defaults compute_100;
adapt ENGRAM_GPU_ARCH to the actual GPU. Binary finds sibling gpu_request.ptx;
ENGRAM_GPU_PTX can override that path. Eight MPI processes, four per node:
  mpirun ... -np 8 --map-by ppr:4:node --bind-to none engram_gpu_request \
    --ips NODE_A,NODE_B --tokens 131072 --rounds 10 --warmups 1 \
    --methods write --layout merged --merge-rows 30 --pool 1024 --qps 4 \
    --engine ex --compact-qp 1 --protocol frame --compact-request 1 \
    --done-gpu 1 --gpu-request 1 --vary-indices 1 --distribution uniform --port 26700
Use the same library environment as the verified DeepEP runtime. NCCL_CROSS_NIC=0
was used in this experiment; no explicit min_stride override is added. The
prototype checks NCCL-detected ginMinStride=4 and uses CUSTOM_STRIDE connectivity
with that detected value, matching the verified CPU-storage DeepEP path. Plain
RAIL in this installed runtime produced uninitialized compact-peer QP entries;
that failed experiment is retained in logs, not used for throughput conclusions.

GPU route kernels decode/count/prefix/stably pack actual current GPU indices by
owner and build output positions each fetch. --gpu-request 0 copies packed rows
back to CPU and sends a compact TCP request. --gpu-request 1 sends remote requests
through GDAKI GIN PUT+strong VA signal into owner CPU mailboxes. Local owners use
GPU stores into CUDA-mapped CPU mailboxes with system release. All four local
owner inboxes are exported/imported and registered so each rail peer can address
any owner; no requester CPU proxy relays remote indices. Owner CPU builds SGEs
and writes payload with verbs; requester GPU waits DONE and scatters. Data WRITE
is still native verbs, not GIN PUT from CPU tables.

Combining GPU requests with target UMR retains a CPU bridge: the current GPU
position map/counts are copied to CPU, CPU builds/configures dynamic keys, and
the indirect-versus-fallback mode is published to GPU before it sends requests.
Root rkeys are registered once with owners during initialization, outside fetch.
The per-fetch CPU bridge and dynamic mapping configuration are timed and must not
be described as a CPU-free fetch. With target_umr=none and gpu_request=1,
there is no data-dependent requester CPU routing/request/wait within fetch;
CPU still submits kernels and handles validation/telemetry in the harness.
An optional CUDA Graph implementation is available as described below; a
persistent application kernel and the formal EngramBuffer API are not integrated.

In GPU samples gpu_route_ms is CUDA count/prefix/pack time. gpu_tail_ms includes
GPU waiting for owners AND reorder/completion, not just the scatter kernel.
CPU/owner work overlaps; do not sum all diagnostic times as serial stages.
Performance runs require ENGRAM_GPU_DEBUG unset/empty; debug=1 adds synchronization
and device QP diagnostics, and sanitizer runs are correctness evidence only.

The whole-fetch UMR tree may require several SQ-sized configuration batches.
ALL children and the root are configured before the index request is published.
The root is mapped once per fetch; owner data windows never remap that root.
One in-flight fetch per requester is used in these tests. A completed fetch can
remap the same root for the next round without changing its registered rkey.

Token-scaling experiments (2026-11-11)
-------------------------------------
Additional flags for engram_gpu_request:
  --gpu-graph 1: initialize a shape/pointer-dependent driver CUDA Graph once.
    Every replay decodes the CURRENT raw GPU indices and runs all required
    routing, requests, transfer completion, scales and scatter. A GPU epoch
    increments once per replay. No index-dependent plan is captured or cached.
  --adaptive-grid 1: size scale/scatter launches to the current tensor shape.
  --profile-routes 0: disable extra profiling events in primary latency runs.
    gpu_route_ms is -1 when profiling is disabled; this is not zero work.
  --owner-poll-us 0: busy-poll remote CPU request mailboxes instead of sleeping
    after unsuccessful polls. This consumes owner CPU cores even when idle.
  --compact-done 1: publish/wait only QPs that carry data for this owner in this
    fetch. Empty owners still publish one completion. Each active QP remains
    individually ordered; never infer completion of one QP from another.
  --owner-windows N: submit N internal WR batches before draining their full CQE
    count. SQ credits are sized accordingly. This is local submission batching,
    not N outstanding fetches and not per-window network ACKs.
  --hybrid-local 1: keep the vocabulary in CPU HOST_NUMA VMM storage. Each GPU
    imports its four local owners' CPU aliases, reads current local rows directly
    into final output, and overlaps these reads with remote owner WRITE. GPU
    routing packs only remote requests; remote results still need GPU scatter.
    No CPU table replica is kept on GPU. Owner-to-output ownership is dynamic.
    Four remote owner service threads/rank busy-poll; local services sleep.
  --host-vmm-table 1: use the same CPU allocation without enabling local bypass;
    this isolates allocation changes from data-path changes in an ablation.
  --gpu-get 1 --gin-contexts 32: experimental small-request alternative. Requires
    gpu-graph and hybrid-local; currently bounded to <=512 tokens. GPU reads
    local CPU rows directly and issues 256B GIN GETs for remote rows directly to
    final output, then flushes all contexts. No owner CPU data service, index
    request, native data QP, staging scatter or dynamic UMR is used. NCCL owns
    the registered output allocation. This is a separate GET implementation,
    not an unchanged DeepEP kernel.
  --get-batch 1: optional GPU GET issuer. One GPU thread owns each context and
    constructs WQEs from current input rows. Earlier remote rows use aggregate
    requests; the final remote row rings the doorbell. Completion still flushes
    every context. Holding the last remote row handles trailing local rows and
    empty contexts correctly. This trades issuer parallelism for fewer queue
    locks/doorbells; select from measured results. Both GET modes reject shapes
    whose worst-case rows/context exceed the configured 1024-entry queue depth.

Candidate remote WRITE configuration (append to the MPI command above):
  --gpu-graph 1 --adaptive-grid 1 --profile-routes 0 --compact-done 1 \
    --owner-poll-us 0 --owner-windows 1 --hybrid-local 1
Candidate small-request GET configuration:
  --gpu-graph 1 --adaptive-grid 1 --profile-routes 0 --hybrid-local 1 \
    --gpu-get 1 --gin-contexts 32
Choose a path from actual latency measurements for the intended token size;
the prototype does not provide an automatic production dispatcher.

Matched DeepEP comparison:
  compare_deepep.py --vary-indices reads the exact raw per-rank/per-epoch inputs
    exported by benchmark.cpp --dump-indices DIR. Upload/generation is upstream
    of both fetch timers. The input tensor address stays fixed; values change.
  --cuda-graph captures the actual DeepEP buffer.fetch AND both layer hooks;
    replay consumes current input and waits for GPU completion. The normal API
    mode includes Python fetch/hooks. No extra CUDA timing events are added
    unless --profile-cuda is requested explicitly.
  --config-read-requests controls the framework QP recommendation assumption;
    --qp-depth 0 uses its recommended depth, --qps 0 its recommended QP count.
    Never lower QPs just to match the unrelated native verbs QP count.
  Actual DeepEP runtime headers in this test use GIN GET for CPU embeddings,
    including same-node embeddings. Its scale-factor path uses local GPU access.
    Direct local CPU embedding loads are an additional optimization here.
  Primary metric: median of each round's maximum full-fetch wall time across
    eight concurrent ranks; numerator is embedding bytes only, denominator also
    includes scale retrieval and completion. Report latency AND GB/s/requester.
    This useful-output bandwidth includes local traffic and is not NIC wire rate.
  Native C++ versus Python API overhead, local access methods and CPU service
    resources differ. Compare complete implementations and show DeepEP Graph
    separately; do not attribute the entire speedup to WRITE versus READ opcode.
  All embedding bytes, scale values and applicable output guards are checked
    outside the timer. Every fetch recomputes its actual index-dependent work.

Machine-specific detailed measurements and operational reports are retained in the local evidence archive.
