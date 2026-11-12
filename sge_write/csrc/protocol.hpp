#pragma once
#include <cstdint>

// CPU与GPU共享的请求ABI；只描述请求，不包含embedding数据。
// 每个owner拥有[requester,layer]独立槽，层间不共享ready/DONE。
// +0 ready序号，+8 nr/base（各32位），+16 active_qps，+64 uint32行号列表。
// ready按[requester,layer,owner]独立递增：只给非空请求发signal，空槽不阻塞其他层。
// 每个槽按最大单层查询量预留并按4KiB对齐。每层完成后可独立调用wait hook。
struct alignas(64) RequestHeader {
    uint64_t ready;
    uint32_t rows, base;
    uint32_t active_qps;
    uint32_t reserved[11];
};
static_assert(sizeof(RequestHeader) == 64);
constexpr unsigned kTile = 4096;
constexpr unsigned kMerge = 30;
