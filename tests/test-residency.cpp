// tests for LLAMA_LOAD_MODE_ADAPTIVE
//
// usage: test-residency [model.gguf ...]
//
// without arguments only the tensor classification and page range math are tested
// with arguments each model is also loaded with mmap and adaptive and the logits must be bit-identical

#include "common.h"
#include "llama.h"
#include "llama-cpp.h"

#include "../src/llama-arch.h"
#include "../src/llama-residency.h"

#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

static int n_failed = 0;

#define CHECK(cond, ...) do { if (!(cond)) { fprintf(stderr, "FAIL %s:%d: %s: ", __FILE__, __LINE__, #cond); fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); n_failed++; } } while (0)

static llama_residency classify(llm_tensor tensor) {
    return llama_residency_classify(llm_tensor_info_for(tensor));
}

static void test_classify() {
    const llm_tensor cold[] = {
        LLM_TENSOR_FFN_GATE_EXPS, LLM_TENSOR_FFN_DOWN_EXPS, LLM_TENSOR_FFN_UP_EXPS, LLM_TENSOR_FFN_GATE_UP_EXPS,
        LLM_TENSOR_FFN_GATE_CHEXPS, LLM_TENSOR_FFN_DOWN_CHEXPS, LLM_TENSOR_FFN_UP_CHEXPS,
    };
    const llm_tensor hot[] = {
        LLM_TENSOR_TOKEN_EMBD, LLM_TENSOR_OUTPUT, LLM_TENSOR_OUTPUT_NORM,
        LLM_TENSOR_ATTN_Q, LLM_TENSOR_ATTN_K, LLM_TENSOR_ATTN_V, LLM_TENSOR_ATTN_OUT, LLM_TENSOR_ATTN_NORM,
        LLM_TENSOR_FFN_NORM, LLM_TENSOR_FFN_GATE, LLM_TENSOR_FFN_DOWN, LLM_TENSOR_FFN_UP,
        LLM_TENSOR_FFN_GATE_INP, LLM_TENSOR_FFN_EXP_PROBS_B, LLM_TENSOR_FFN_GATE_INP_SHEXP,
        LLM_TENSOR_FFN_GATE_SHEXP, LLM_TENSOR_FFN_DOWN_SHEXP, LLM_TENSOR_FFN_UP_SHEXP,
        LLM_TENSOR_FFN_NORM_EXPS, // arctic: a single {n_embd} norm applied before the experts
    };
    const LLM_TN tn(LLM_ARCH_LLAMA);
    for (llm_tensor t : cold) {
        CHECK(classify(t) == LLAMA_RESIDENCY_COLD, "%s should be cold", tn(t, "weight", 0).str().c_str());
    }
    for (llm_tensor t : hot) {
        CHECK(classify(t) == LLAMA_RESIDENCY_HOT, "%s should be hot", tn(t, "weight", 0).str().c_str());
    }

    // every tensor type of every architecture: routed experts are named *_exps / *_chexps
    int n_cold = 0;
    for (int i = 0; i <= LLM_TENSOR_D2T; i++) {
        const llm_tensor t = (llm_tensor) i;
        llm_tensor_info info;
        try {
            info = llm_tensor_info_for(t);
        } catch (const std::out_of_range &) {
            continue;
        }
        const std::string name = tn(t, "weight", 0).str();
        const bool expert_name = (name.find("_exps.") != std::string::npos || name.find("_chexps.") != std::string::npos)
            && t != LLM_TENSOR_FFN_NORM_EXPS;
        const bool is_cold = llama_residency_classify(info) == LLAMA_RESIDENCY_COLD;
        CHECK(is_cold == expert_name, "%s classified %s", name.c_str(), is_cold ? "cold" : "hot");
        n_cold += is_cold;
    }
    CHECK(n_cold > 0, "no cold tensor types");
}

static void test_page_ranges() {
    const size_t P = 4096;
    // an empty inner range is expected as {0, 0}, its position does not matter
    struct { size_t first, last, limit, out_first, out_last, in_first, in_last; } cases[] = {
        // first   last    limit   outer             inner
        {     0,      1, 1 << 20,     0,   P,         0,   0   }, // sub-page
        {     0,      P, 1 << 20,     0,   P,         0,   P   }, // exactly one page
        {    32,    3*P, 1 << 20,     0, 3*P,         P, 3*P   }, // unaligned start
        {   P-1,    P+1, 1 << 20,     0, 2*P,         0,   0   }, // straddles one boundary
        {   P+1,  3*P-1, 1 << 20,     P, 3*P,         0,   0   }, // two partial pages
        {   P+1,  4*P-1, 1 << 20,     P, 4*P,       2*P, 3*P   }, // inner rounds both ends
        {   100,   5000,    4500,     0, 4500,        0,   0   }, // outer clamped to the end of the mapping
        {   2*P,    2*P, 1 << 20,   2*P, 2*P,         0,   0   }, // empty
    };
    for (const auto & c : cases) {
        size_t f = c.first, l = c.last;
        llama_page_range_outer(f, l, P, c.limit);
        CHECK(f == c.out_first && l == c.out_last, "outer [%zu, %zu) -> [%zu, %zu)", c.first, c.last, f, l);
        f = c.first; l = c.last;
        llama_page_range_inner(f, l, P);
        const bool ok = c.in_first == c.in_last ? f == l : f == c.in_first && l == c.in_last;
        CHECK(ok, "inner [%zu, %zu) -> [%zu, %zu)", c.first, c.last, f, l);
    }

    // random 32-byte aligned layouts: a cold range must never cover a page that holds hot bytes
    // and a hot range must cover all of its tensor
    std::mt19937 rng(42);
    for (int iter = 0; iter < 200; iter++) {
        struct tensor { size_t first, last; bool cold; };
        std::vector<tensor> tensors;
        size_t offs = 0;
        for (int i = 0; i < 64; i++) {
            const size_t size = 1 + rng() % (3*P);
            tensors.push_back({offs, offs + size, (rng() % 2) == 0});
            offs = (offs + size + 31) & ~(size_t) 31;
        }
        const size_t limit = offs;
        std::vector<bool> page_hot((limit + P - 1)/P, false);
        for (const auto & t : tensors) {
            if (!t.cold) {
                for (size_t p = t.first/P; p <= (t.last - 1)/P; p++) {
                    page_hot[p] = true;
                }
            }
        }
        for (const auto & t : tensors) {
            size_t f = t.first, l = t.last;
            if (t.cold) {
                llama_page_range_inner(f, l, P);
                CHECK(f == l || (f % P == 0 && l % P == 0 && f >= t.first && l <= t.last), "cold [%zu, %zu) -> [%zu, %zu)", t.first, t.last, f, l);
                for (size_t p = f/P; p < l/P; p++) {
                    CHECK(!page_hot[p], "cold range [%zu, %zu) covers hot page %zu", f, l, p);
                }
            } else {
                llama_page_range_outer(f, l, P, limit);
                CHECK(f % P == 0 && (l % P == 0 || l == limit) && f <= t.first && l >= t.last, "hot [%zu, %zu) -> [%zu, %zu)", t.first, t.last, f, l);
            }
        }
    }
}

struct log_capture {
    double size_cold = -1.0;
};

static void log_cb(ggml_log_level level, const char * text, void * user_data) {
    auto * cap = (log_capture *) user_data;
    const char * p = strstr(text, "adaptive: hot ");
    if (p) {
        double hot, locked, cold;
        if (sscanf(p, "adaptive: hot %lf MiB (%lf MiB locked), cold %lf MiB", &hot, &locked, &cold) == 3) {
            cap->size_cold = cold;
        }
    }
    if (level >= GGML_LOG_LEVEL_WARN) {
        fputs(text, stderr);
    }
}

static std::vector<float> run_model(const char * path, llama_load_mode load_mode, log_capture & cap) {
    llama_log_set(log_cb, &cap);

    llama_model_params mparams = llama_model_default_params();
    mparams.load_mode    = load_mode;
    mparams.n_gpu_layers = 0; // adaptive only manages weights served from the mmap
    llama_model_ptr model(llama_model_load_from_file(path, mparams));
    if (!model) {
        throw std::runtime_error(std::string("failed to load ") + path);
    }

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx           = 64;
    cparams.n_batch         = 64;
    cparams.n_ubatch        = 64;
    cparams.n_threads       = 4;
    cparams.n_threads_batch = 4;
    llama_context_ptr ctx(llama_init_from_model(model.get(), cparams));
    if (!ctx) {
        throw std::runtime_error(std::string("failed to create context for ") + path);
    }

    const int32_t n_vocab  = llama_vocab_n_tokens(llama_model_get_vocab(model.get()));
    const int32_t n_tokens = 16;
    llama_batch batch = llama_batch_init(n_tokens, 0, 1);
    for (int32_t i = 0; i < n_tokens; i++) {
        common_batch_add(batch, (1 + 7*i) % n_vocab, i, {0}, true);
    }
    const int ret = llama_decode(ctx.get(), batch);
    llama_batch_free(batch);
    if (ret != 0) {
        throw std::runtime_error(std::string("failed to decode with ") + path);
    }

    std::vector<float> logits;
    for (int32_t i = 0; i < n_tokens; i++) {
        const float * l = llama_get_logits_ith(ctx.get(), i);
        logits.insert(logits.end(), l, l + n_vocab);
    }
    return logits;
}

static void test_model(const char * path) {
    log_capture cap_mmap;
    log_capture cap_adaptive;
    const std::vector<float> ref = run_model(path, LLAMA_LOAD_MODE_MMAP,     cap_mmap);
    const std::vector<float> out = run_model(path, LLAMA_LOAD_MODE_ADAPTIVE, cap_adaptive);

    CHECK(cap_mmap.size_cold < 0.0, "%s: residency policy applied with mmap", path);
    CHECK(cap_adaptive.size_cold >= 0.0, "%s: residency policy not applied with adaptive", path);
    if (strstr(path, "-moe.gguf")) {
        CHECK(cap_adaptive.size_cold > 0.0, "%s: no cold tensors in MoE model", path);
    }
    CHECK(ref.size() == out.size() && memcmp(ref.data(), out.data(), ref.size()*sizeof(float)) == 0,
            "%s: adaptive logits differ from mmap", path);
    printf("  %s: cold %.3f MiB, logits %s\n", path, cap_adaptive.size_cold,
            ref.size() == out.size() && memcmp(ref.data(), out.data(), ref.size()*sizeof(float)) == 0 ? "identical" : "DIFFER");
}

int main(int argc, char ** argv) {
    printf("test_classify\n");
    test_classify();
    printf("test_page_ranges\n");
    test_page_ranges();

    if (argc > 1) {
        llama_backend_init();
        printf("test_model\n");
        for (int i = 1; i < argc; i++) {
            try {
                test_model(argv[i]);
            } catch (const std::exception & e) {
                CHECK(false, "%s", e.what());
            }
        }
        llama_backend_free();
    }

    if (n_failed > 0) {
        printf("%d check(s) failed\n", n_failed);
        return 1;
    }
    printf("OK\n");
    return 0;
}
