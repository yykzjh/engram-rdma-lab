#pragma once

// These are JIT compile-time settings. Use a fresh JIT cache when changing them;
// fetch and fetch_wait must use the same configuration for a Buffer's lifetime.
#ifndef EP_ENGRAM_LOCAL_CPU_READ
#define EP_ENGRAM_LOCAL_CPU_READ 1
#endif

#ifndef EP_ENGRAM_LOCAL_COPY_LANES
#define EP_ENGRAM_LOCAL_COPY_LANES 16
#endif

// Preserve the original pending-row copy as an ablation. The default distributes
// local rows over all local issue warps, including when there is only one tile.
#ifndef EP_ENGRAM_LOCAL_COPY_DISTRIBUTED
#define EP_ENGRAM_LOCAL_COPY_DISTRIBUTED 1
#endif

static_assert(EP_ENGRAM_LOCAL_COPY_LANES > 0 &&
              EP_ENGRAM_LOCAL_COPY_LANES <= 32 &&
              32 % EP_ENGRAM_LOCAL_COPY_LANES == 0,
              "Local copy lanes must divide a warp");
