# Third-party source notices

- DeepEP: https://github.com/deepseek-ai/DeepEP , commit `f6aa38fe06cd4a0489e7730365d4e0114970a2f3`. Engram and required shared infrastructure extracted; modified for isolated Engram registration and hybrid CPU loads. Original copyright and MIT license retained in `deepep/LICENSE` and root `LICENSE`.
- DeepJIT: https://github.com/deepseek-ai/DeepJIT , commit `cba802df081985141eeee36803685775c183a047`. Header-only dependency snapshot in `deepep/third-party/deep_jit/include`. That upstream snapshot contains no separate LICENSE file; this archive records its source and does not invent an additional license grant.
- NCCL, CUDA, PyTorch, MPI and RDMA libraries are external runtime/build dependencies, not vendored binaries.
