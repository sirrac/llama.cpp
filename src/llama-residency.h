#pragma once

#include "llama-arch.h"

#include <algorithm>
#include <cstddef>

// page residency policy for LLAMA_LOAD_MODE_ADAPTIVE
enum llama_residency {
    LLAMA_RESIDENCY_HOT,  // used on every token: populated eagerly and pinned
    LLAMA_RESIDENCY_COLD, // routed MoE experts: left to demand paging, hinted with MADV_COLD
};

// routed experts (and their biases/scales) are exactly the weights consumed by MUL_MAT_ID
// shared experts, the router and norms such as ffn_norm_exps are used on every token
inline llama_residency llama_residency_classify(const llm_tensor_info & info) {
    return info.op == GGML_OP_MUL_MAT_ID ? LLAMA_RESIDENCY_COLD : LLAMA_RESIDENCY_HOT;
}

// round [first, last) outward to whole pages, clamped to limit (page must be a power of 2)
// used for hot ranges: a page shared with a neighbor must still be resident
inline void llama_page_range_outer(size_t & first, size_t & last, size_t page, size_t limit) {
    first = first & ~(page - 1);
    last  = std::min((last + page - 1) & ~(page - 1), limit);
    last  = std::max(last, first);
}

// round [first, last) inward to whole pages, empty if no page is fully covered
// used for cold ranges: GGUF tensors are only 32-byte aligned, so an edge page may belong to a hot neighbor
inline void llama_page_range_inner(size_t & first, size_t & last, size_t page) {
    first = (first + page - 1) & ~(page - 1);
    last  = last & ~(page - 1);
    last  = std::max(last, first);
}
