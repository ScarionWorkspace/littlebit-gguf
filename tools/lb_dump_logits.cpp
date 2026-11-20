// Dump llama.cpp logits for a fixed token sequence (all positions), for comparison with
// the Python reference. Build: tools/build_tools.sh. MIT license.
//
// usage: lb_dump_logits <model.gguf> <ids.i32> <out.f32> [n_threads] [flash_attn: auto|on|off] [n_gpu_layers] [names]
//   ids.i32: raw little-endian int32 token ids
//   out.f32: raw float32 logits, row-major [n_tokens, n_vocab]
//            if the name ends in ".nll": instead float64 NLL of ids[i+1] given position i, [n_tokens-1]
//   names:   optional comma-separated graph tensor names (e.g. "l_out-0,l_out-14") to dump
//            as <out.f32>.<name> (F32, ggml order: ne0 fastest)
#include "ggml-backend.h"
#include "llama.h"

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <fstream>
#include <set>
#include <sstream>
#include <string>
#include <vector>

struct dump_state {
    std::set<std::string> names;
    std::string prefix;
};

static bool dump_cb(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * st = static_cast<dump_state *>(user_data);
    const bool want = st->names.count(ggml_get_name(t)) > 0 && t->type == GGML_TYPE_F32;
    if (ask) {
        return want;
    }
    if (want) {
        std::vector<float> buf(ggml_nelements(t));
        ggml_backend_tensor_get(t, buf.data(), 0, ggml_nbytes(t));
        std::ofstream f(st->prefix + "." + ggml_get_name(t), std::ios::binary);
        f.write(reinterpret_cast<const char *>(buf.data()), buf.size() * sizeof(float));
        fprintf(stderr, "dumped %s [%lld, %lld]\n", ggml_get_name(t), (long long) t->ne[0], (long long) t->ne[1]);
    }
    return true;
}

int main(int argc, char ** argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s model.gguf ids.i32 out.f32 [n_threads] [flash_attn auto|on|off] [n_gpu_layers]\n", argv[0]);
        return 1;
    }
    const int n_threads = argc > 4 ? atoi(argv[4]) : 8;
    const char * fa = argc > 5 ? argv[5] : "auto";
    const int ngl = argc > 6 ? atoi(argv[6]) : 0;
    dump_state st;
    st.prefix = argv[3];
    if (argc > 7) {
        std::stringstream ss(argv[7]);
        for (std::string item; std::getline(ss, item, ',');) {
            st.names.insert(item);
        }
    }

    std::ifstream fin(argv[2], std::ios::binary | std::ios::ate);
    if (!fin) { fprintf(stderr, "cannot read %s\n", argv[2]); return 1; }
    const size_t nbytes = fin.tellg();
    fin.seekg(0);
    std::vector<llama_token> ids(nbytes / sizeof(int32_t));
    fin.read(reinterpret_cast<char *>(ids.data()), nbytes);
    const int n = (int) ids.size();

    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = ngl;
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) { fprintf(stderr, "failed to load model\n"); return 1; }

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = n;
    cp.n_batch = n;
    cp.n_ubatch = n;
    cp.n_threads = n_threads;
    cp.n_threads_batch = n_threads;
    cp.flash_attn_type = !strcmp(fa, "on") ? LLAMA_FLASH_ATTN_TYPE_ENABLED
                       : !strcmp(fa, "off") ? LLAMA_FLASH_ATTN_TYPE_DISABLED : LLAMA_FLASH_ATTN_TYPE_AUTO;
    if (!st.names.empty()) {
        cp.cb_eval = dump_cb;
        cp.cb_eval_user_data = &st;
    }
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) { fprintf(stderr, "failed to create context\n"); return 1; }

    llama_batch batch = llama_batch_init(n, 0, 1);
    for (int i = 0; i < n; ++i) {
        batch.token[i] = ids[i];
        batch.pos[i] = i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = 1;
    }
    batch.n_tokens = n;
    if (llama_decode(ctx, batch) != 0) { fprintf(stderr, "decode failed\n"); return 1; }

    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    std::ofstream fout(argv[3], std::ios::binary);
    const std::string out_name = argv[3];
    const bool nll_mode = out_name.size() > 4 && out_name.compare(out_name.size() - 4, 4, ".nll") == 0;
    for (int i = 0; i < n; ++i) {
        const float * lg = llama_get_logits_ith(ctx, i);
        if (!nll_mode) {
            fout.write(reinterpret_cast<const char *>(lg), sizeof(float) * n_vocab);
            continue;
        }
        if (i == n - 1) {
            break;
        }
        double mx = lg[0];
        for (int j = 1; j < n_vocab; ++j) mx = std::max(mx, (double) lg[j]);
        double se = 0.0;
        for (int j = 0; j < n_vocab; ++j) se += std::exp((double) lg[j] - mx);
        const double nll = mx + std::log(se) - (double) lg[ids[i + 1]];
        fout.write(reinterpret_cast<const char *>(&nll), sizeof(double));
    }
    fprintf(stderr, "wrote %d x %d logits\n", n, n_vocab);

    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
