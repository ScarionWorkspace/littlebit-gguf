// Microbenchmark: GPU time of LittleBit linear layers (or a dense reference) in a ggml graph.
// Builds n_layers chained projections of one shape (output of one is the input of the next, so
// n_in == n_out is required for chaining; for other shapes each layer reads the same input x),
// runs the graph n_iter times on the first GPU backend and reports the time per projection.
// Same graph every iteration, so the CUDA/HIP backend can use graphs, like during decoding.
// Toggle the fused LittleBit path with GGML_CUDA_DISABLE_FUSION=1. MIT license.
//
// usage: lb_bench_linear <mode: lb|q4_k|q1_0> <n_in> <n_out> <r> [n_tokens] [n_layers] [n_iter]
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    if (argc < 5) {
        fprintf(stderr, "usage: %s lb|q4_k|q1_0 n_in n_out r [n_tokens] [n_layers] [n_iter]\n", argv[0]);
        return 1;
    }
    const std::string mode = argv[1];
    const int64_t n_in = atoll(argv[2]), n_out = atoll(argv[3]), r = atoll(argv[4]);
    const int64_t n_tok = argc > 5 ? atoll(argv[5]) : 1;
    const int n_layers = argc > 6 ? atoi(argv[6]) : 36;
    const int n_iter = argc > 7 ? atoi(argv[7]) : 200;
    const int64_t r_pad = (r + 127) / 128 * 128;
    const bool chain = n_in == n_out;

    ggml_backend_load_all();
    ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    if (!dev) { fprintf(stderr, "no GPU backend\n"); return 1; }
    ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);

    ggml_init_params ip = { ggml_tensor_overhead() * (size_t) (n_layers * 16 + 16) + ggml_graph_overhead_custom(8192, false), nullptr, true };
    ggml_context * ctx_w = ggml_init(ip);
    ggml_context * ctx_g = ggml_init(ip);

    ggml_tensor * x = ggml_new_tensor_2d(ctx_w, GGML_TYPE_F32, n_in, n_tok);
    struct layer { ggml_tensor *g[2], *v[2], *u[2], *w; };
    std::vector<layer> layers(n_layers);
    for (auto & l : layers) {
        if (mode == "lb") {
            for (int p = 0; p < 2; ++p) {
                l.g[p] = ggml_new_tensor_1d(ctx_w, GGML_TYPE_F32, n_in);
                l.v[p] = ggml_new_tensor_2d(ctx_w, GGML_TYPE_Q1_0, n_in, r);
                l.u[p] = ggml_new_tensor_2d(ctx_w, GGML_TYPE_Q1_0, r_pad, n_out);
            }
        } else {
            l.w = ggml_new_tensor_2d(ctx_w, mode == "q4_k" ? GGML_TYPE_Q4_K : GGML_TYPE_Q1_0, n_in, n_out);
        }
    }
    ggml_backend_buffer_t buf_w = ggml_backend_alloc_ctx_tensors(ctx_w, backend);
    // random bytes are fine for timing; keep fp16 scales finite by using small values
    for (ggml_tensor * t = ggml_get_first_tensor(ctx_w); t; t = ggml_get_next_tensor(ctx_w, t)) {
        std::vector<uint8_t> data(ggml_nbytes(t));
        if (t->type == GGML_TYPE_F32) {
            float * f = (float *) data.data();
            for (size_t i = 0; i < data.size() / 4; ++i) f[i] = 0.01f * (float) (rand() % 200 - 100) / 100.0f;
        } else {
            for (auto & b : data) b = (uint8_t) (rand() & 0x3f); // keeps fp16 d small and finite
        }
        ggml_backend_tensor_set(t, data.data(), 0, data.size());
    }

    ggml_cgraph * gf = ggml_new_graph_custom(ctx_g, 8192, false);
    ggml_tensor * cur = x;
    for (auto & l : layers) {
        ggml_tensor * inp = chain ? cur : x;
        ggml_tensor * y;
        if (mode == "lb") {
            y = nullptr;
            for (int p = 0; p < 2; ++p) {
                ggml_tensor * t = ggml_mul_mat(ctx_g, l.v[p], ggml_mul(ctx_g, inp, l.g[p]));
                if (r_pad > r) t = ggml_pad(ctx_g, t, (int) (r_pad - r), 0, 0, 0);
                t = ggml_mul_mat(ctx_g, l.u[p], t);
                y = y ? ggml_add(ctx_g, y, t) : t;
            }
        } else {
            y = ggml_mul_mat(ctx_g, l.w, inp);
        }
        ggml_build_forward_expand(gf, y);
        cur = y;
    }
    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    ggml_gallocr_alloc_graph(galloc, gf);

    for (int i = 0; i < 10; ++i) ggml_backend_graph_compute(backend, gf); // warmup, lets the backend capture a graph
    ggml_backend_synchronize(backend);
    const auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < n_iter; ++i) ggml_backend_graph_compute(backend, gf);
    ggml_backend_synchronize(backend);
    const double us = std::chrono::duration<double, std::micro>(std::chrono::high_resolution_clock::now() - t0).count();

    printf("%s n_in=%lld n_out=%lld r=%lld n_tok=%lld: %.2f us per projection (%d nodes/graph)\n", mode.c_str(),
           (long long) n_in, (long long) n_out, (long long) r, (long long) n_tok, us / n_iter / n_layers, ggml_graph_n_nodes(gf));

    ggml_gallocr_free(galloc);
    ggml_backend_buffer_free(buf_w);
    ggml_free(ctx_g);
    ggml_free(ctx_w);
    ggml_backend_free(backend);
    return 0;
}
