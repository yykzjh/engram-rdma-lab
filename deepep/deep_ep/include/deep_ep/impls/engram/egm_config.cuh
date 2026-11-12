#pragma once
// Host routing selects this immutable specialization for fetch and wait together.
// Separate JIT cache keys keep pure RDMA and EGM-enabled graphs independent.
#ifndef EP_ENGRAM_EGM_READ
#define EP_ENGRAM_EGM_READ 0
#endif
