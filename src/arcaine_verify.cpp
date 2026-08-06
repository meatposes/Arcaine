// arcaine_verify — numerical gate for kernel A/B changes.
//
// Every fast path in this engine is selected by an environment flag
// (ARCAINE_QWEN35_NVFP4_DPAS, ARCAINE_QWEN35_XMX_ATTENTION,
// ARCAINE_QWEN35_ESIMD_DELTA, DIFF_NVFP4_*, ...). Timing them against each
// other only means something if they compute the same thing, and a benchmark
// cannot tell a kernel that is fast from a kernel that is fast and wrong.
//
// So: `capture` records the logit trajectory of a reference configuration,
// `compare` re-runs the identical token sequence under whatever configuration
// is currently selected and reports how far it drifted.
//
// Both subcommands are teacher-forced — every step is fed the recorded token
// rather than its own sample — so the comparison is deterministic and does not
// depend on the sampler, and so perplexity over the same tokens is directly
// comparable between the two runs. That last number is the gate for changes
// that are *intentionally* lossy (a lower-precision head, quantized
// activations): max-abs-diff will be large by design, and the question is
// whether the model got worse.
//
// Usage:
//   arcaine_verify capture --model DIR --out golden.bin [token source] [opts]
//   arcaine_verify compare --model DIR --golden golden.bin [opts]
//
// Token source (both subcommands; compare takes it from the golden file):
//   --prompt TEXT     tokenize TEXT through the model's chat template
//   --tokens FILE     whitespace-separated token ids
//   (default)         a fixed pseudo-random sequence — fine for comparing
//                     logits, meaningless for perplexity
//
// Typical use: capture with the conservative kernels, compare with the fast
// ones.
//   ZE_AFFINITY_MASK=2 ARCAINE_QWEN35_NVFP4_DPAS=0 ARCAINE_QWEN35_XMX_ATTENTION=0 \
//     ARCAINE_QWEN35_ESIMD_DELTA=0 ./build/arcaine_verify capture \
//       --model $M --out golden.bin --prompt "$(cat sample.txt)"
//   ZE_AFFINITY_MASK=2 ./build/arcaine_verify compare --model $M --golden golden.bin

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "common/model_interface.hpp"
#include "common/registry.hpp"
#include "common/gpu/device_select.hpp"

namespace {

constexpr char     kMagic[8]  = {'A','R','C','G','O','L','D','\0'};
constexpr uint32_t kVersion   = 1;
constexpr size_t   kConfigLen = 512;

// The kernel-selection flags, snapshotted into the golden file so a comparison
// can state which configurations it is actually contrasting.
const char* const kTrackedEnv[] = {
    "ARCAINE_QWEN35_NVFP4_DPAS",
    "ARCAINE_QWEN35_XMX_ATTENTION",
    "ARCAINE_QWEN35_SUBGROUP_ATTENTION",
    "ARCAINE_QWEN35_ESIMD_DELTA",
    "ARCAINE_QWEN35_FUSED_ESIMD_DELTA_DECODE",
    "ARCAINE_QWEN35_FUSED_BA_PROJECTION",
    "ARCAINE_QWEN35_FUSED_FP8_PROJECTIONS",
    "ARCAINE_QWEN35_PERSISTENT_IO",
    "ARCAINE_QWEN35_MAX_LAYERS",
    "DIFF_NVFP4_GROUPED_GEMM",
    "DIFF_NVFP4_EXPERT_KERNEL",
    "DIFF_NVFP4_GPU_LAYOUT",
    "DIFF_NVFP4_WEIGHT_LAYOUT",
    "ZE_AFFINITY_MASK",
};

std::string env_snapshot() {
    std::string s;
    for (const char* name : kTrackedEnv) {
        const char* v = std::getenv(name);
        s += name;
        s += '=';
        s += v ? v : "(unset)";
        s += ' ';
    }
    if (s.size() >= kConfigLen) s.resize(kConfigLen - 1);
    return s;
}

struct Golden {
    uint32_t vocab = 0, prefill = 0, records = 0;
    std::string config;
    std::vector<int32_t> tokens;   // prefill + records, teacher-forced
    std::vector<float>   logits;   // records * vocab
};

bool write_golden(const std::string& path, const Golden& g) {
    std::ofstream f(path, std::ios::binary);
    if (!f) return false;
    f.write(kMagic, sizeof kMagic);
    auto u32 = [&](uint32_t v) { f.write((const char*)&v, 4); };
    u32(kVersion);
    u32(g.vocab);
    u32(g.prefill);
    u32(g.records);
    u32((uint32_t)g.tokens.size());
    std::vector<char> cfg(kConfigLen, 0);
    std::memcpy(cfg.data(), g.config.data(),
                std::min(g.config.size(), kConfigLen - 1));
    f.write(cfg.data(), kConfigLen);
    f.write((const char*)g.tokens.data(), (std::streamsize)g.tokens.size() * 4);
    f.write((const char*)g.logits.data(), (std::streamsize)g.logits.size() * 4);
    return (bool)f;
}

bool read_golden(const std::string& path, Golden& g) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    char magic[8];
    f.read(magic, sizeof magic);
    if (std::memcmp(magic, kMagic, sizeof magic) != 0) {
        std::fprintf(stderr, "%s: not an arcaine golden file\n", path.c_str());
        return false;
    }
    auto u32 = [&] { uint32_t v = 0; f.read((char*)&v, 4); return v; };
    uint32_t version = u32();
    if (version != kVersion) {
        std::fprintf(stderr, "%s: golden format v%u, this build reads v%u\n",
                     path.c_str(), version, kVersion);
        return false;
    }
    g.vocab = u32();
    g.prefill = u32();
    g.records = u32();
    uint32_t n_tokens = u32();
    std::vector<char> cfg(kConfigLen);
    f.read(cfg.data(), kConfigLen);
    g.config.assign(cfg.data(), strnlen(cfg.data(), kConfigLen));
    g.tokens.resize(n_tokens);
    f.read((char*)g.tokens.data(), (std::streamsize)n_tokens * 4);
    g.logits.resize((size_t)g.records * g.vocab);
    f.read((char*)g.logits.data(), (std::streamsize)g.logits.size() * 4);
    return (bool)f;
}

// log(sum(exp(x))) computed against the max, so a 248k-wide logit vector does
// not overflow before it is normalized.
double logsumexp(const float* x, int n) {
    float m = -INFINITY;
    for (int i = 0; i < n; ++i) m = std::max(m, x[i]);
    if (!std::isfinite(m)) return (double)m;
    double sum = 0.0;
    for (int i = 0; i < n; ++i) sum += std::exp((double)x[i] - m);
    return (double)m + std::log(sum);
}

int argmax(const float* x, int n) {
    int best = 0;
    for (int i = 1; i < n; ++i) if (x[i] > x[best]) best = i;
    return best;
}

// Mean negative log-likelihood of each recorded target under its own logits.
// Meaningful only when the tokens came from real text.
double perplexity(const std::vector<float>& logits, uint32_t records,
                  uint32_t vocab, const std::vector<int32_t>& tokens,
                  uint32_t prefill) {
    double nll = 0.0;
    uint32_t counted = 0;
    for (uint32_t r = 0; r < records; ++r) {
        const float* row = logits.data() + (size_t)r * vocab;
        int target = tokens[prefill + r];
        if (target < 0 || (uint32_t)target >= vocab) continue;
        nll += logsumexp(row, (int)vocab) - (double)row[target];
        ++counted;
    }
    return counted ? std::exp(nll / counted) : 0.0;
}

// Teacher-forced trajectory: one prefill over the first `prefill` tokens, then
// one single-token decode per remaining position. Each record is the logit
// vector that predicts tokens[prefill + r].
std::vector<float> run_trajectory(Model& model, const std::vector<int32_t>& tokens,
                                  uint32_t prefill, uint32_t records,
                                  uint32_t vocab) {
    std::vector<float> out((size_t)records * vocab);
    model.reset_cache();

    std::vector<int> prompt(tokens.begin(), tokens.begin() + prefill);
    std::vector<float> logits = model.forward(ForwardInput{prompt, 0});
    if (logits.size() != vocab)
        throw std::runtime_error("forward returned an unexpected logit count");
    std::copy(logits.begin(), logits.end(), out.begin());

    for (uint32_t r = 1; r < records; ++r) {
        std::vector<int> step{(int)tokens[prefill + r - 1]};
        logits = model.forward(ForwardInput{step, (int)(prefill + r - 1)});
        std::copy(logits.begin(), logits.end(),
                  out.begin() + (size_t)r * vocab);
        if (r % 8 == 0) { std::printf("."); std::fflush(stdout); }
    }
    std::printf("\n");
    return out;
}

std::vector<int32_t> tokens_from_file(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("cannot read token file: " + path);
    std::vector<int32_t> out;
    long v;
    while (f >> v) out.push_back((int32_t)v);
    return out;
}

// Dependency-free default. Exercises the same kernels as real text; only the
// perplexity number is meaningless, which the report says out loud.
std::vector<int32_t> synthetic_tokens(int count, int vocab) {
    std::vector<int32_t> out;
    out.reserve(count);
    uint64_t state = 0x243f6a8885a308d3ull;
    for (int i = 0; i < count; ++i) {
        state = state * 6364136223846793005ull + 1442695040888963407ull;
        out.push_back((int32_t)((state >> 33) % (uint64_t)vocab));
    }
    return out;
}

void usage() {
    std::fputs(
        "Usage:\n"
        "  arcaine_verify capture --model DIR --out FILE [token source] [opts]\n"
        "  arcaine_verify compare --model DIR --golden FILE [opts]\n"
        "\n"
        "Token source (capture only):\n"
        "  --prompt TEXT     tokenize through the model's chat template\n"
        "  --tokens FILE     whitespace-separated token ids\n"
        "  (default)         fixed pseudo-random ids; perplexity is then\n"
        "                    meaningless and is reported as such\n"
        "\n"
        "Options:\n"
        "  --steps N         logit records to capture      (default: 32)\n"
        "  --prefill N       tokens in the prefill call    (default: 16)\n"
        "  --max-seq N       KV capacity                   (default: auto)\n"
        "  --device N        restrict to one Level Zero GPU\n"
        "  --tol-abs A       max |Δlogit| before failing   (default: 0.05)\n"
        "  --tol-top1 R      min top-1 agreement           (default: 1.0)\n",
        stderr);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) { usage(); return 1; }
    std::string mode = argv[1];
    if (mode == "-h" || mode == "--help") { usage(); return 0; }
    if (mode != "capture" && mode != "compare") {
        std::fprintf(stderr, "unknown subcommand: %s\n\n", mode.c_str());
        usage();
        return 1;
    }

    std::string model_dir, out_path, golden_path, prompt, tokens_file, device_index;
    int steps = 32, prefill = 16, max_seq = -1;
    double tol_abs = 0.05, tol_top1 = 1.0;
    bool device_index_set = false;

    for (int i = 2; i < argc; ++i) {
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) { std::fprintf(stderr, "missing value for %s\n", argv[i]); std::exit(1); }
            return argv[++i];
        };
        std::string a = argv[i];
        if      (a == "--model")     model_dir   = next();
        else if (a == "--out")       out_path    = next();
        else if (a == "--golden")    golden_path = next();
        else if (a == "--prompt")    prompt      = next();
        else if (a == "--tokens")    tokens_file = next();
        else if (a == "--steps")     steps       = std::stoi(next());
        else if (a == "--prefill")   prefill     = std::stoi(next());
        else if (a == "--max-seq")   max_seq     = std::stoi(next());
        else if (a == "--device")  { device_index = next(); device_index_set = true; }
        else if (a == "--tol-abs")   tol_abs     = std::stod(next());
        else if (a == "--tol-top1")  tol_top1    = std::stod(next());
        else if (a == "-h" || a == "--help") { usage(); return 0; }
        else { std::fprintf(stderr, "unknown argument: %s\n", a.c_str()); usage(); return 1; }
    }

    if (model_dir.empty()) { usage(); return 1; }
    if (mode == "capture" && out_path.empty()) { usage(); return 1; }
    if (mode == "compare" && golden_path.empty()) { usage(); return 1; }
    if (steps < 1 || prefill < 1) {
        std::fputs("--steps and --prefill must be positive\n", stderr);
        return 1;
    }

    try {
        if (device_index_set) gpu_device_control::apply_device_index(device_index);

        Golden golden;
        if (mode == "compare" && !read_golden(golden_path, golden)) {
            std::fprintf(stderr, "failed to read %s\n", golden_path.c_str());
            return 1;
        }

        register_builtin_architectures();

        // The golden run fixes the sequence length, so the cache must be at
        // least that deep for the comparison to replay it.
        int needed = (mode == "compare")
                         ? (int)golden.tokens.size() + 1
                         : prefill + steps + 1;
        if (max_seq < 0) max_seq = std::max(needed, 512);
        if (max_seq < needed) {
            std::fprintf(stderr, "--max-seq %d is below the %d tokens this run needs\n",
                         max_seq, needed);
            return 1;
        }

        std::printf("[verify] loading %s\n", model_dir.c_str());
        std::unique_ptr<Model> model =
            ModelRegistry::instance().create(model_dir, max_seq);
        const ModelInfo& info = model->info();
        const uint32_t vocab = (uint32_t)info.vocab_size;
        std::printf("[verify] %s\n", info.description.c_str());

        std::vector<int32_t> tokens;
        bool real_text = false;

        if (mode == "compare") {
            if (golden.vocab != vocab) {
                std::fprintf(stderr,
                             "golden vocab %u != model vocab %u; different model\n",
                             golden.vocab, vocab);
                return 1;
            }
            tokens  = golden.tokens;
            prefill = (int)golden.prefill;
            steps   = (int)golden.records;
            real_text = true;   // unknown; the golden's own ppl is the reference
        } else if (!tokens_file.empty()) {
            tokens = tokens_from_file(tokens_file);
            real_text = true;
        } else if (!prompt.empty()) {
            PreparedInput prepared = model->prepare_input(prompt, {}, {}, "");
            tokens.assign(prepared.tokens.begin(), prepared.tokens.end());
            real_text = true;
        } else {
            tokens = synthetic_tokens(prefill + steps, (int)vocab);
        }

        if ((int)tokens.size() < prefill + steps) {
            if (mode == "compare") {
                std::fputs("golden file is truncated\n", stderr);
                return 1;
            }
            steps = (int)tokens.size() - prefill;
            if (steps < 1) {
                std::fprintf(stderr,
                             "token source has %d tokens, need more than --prefill %d\n",
                             (int)tokens.size(), prefill);
                return 1;
            }
            std::printf("[verify] token source is short; capturing %d records\n", steps);
        }
        tokens.resize(prefill + steps);

        std::printf("[verify] prefill %d, %d records, vocab %u\n",
                    prefill, steps, vocab);
        std::printf("[verify] config: %s\n", env_snapshot().c_str());

        std::vector<float> logits =
            run_trajectory(*model, tokens, (uint32_t)prefill, (uint32_t)steps, vocab);

        if (mode == "capture") {
            Golden g;
            g.vocab   = vocab;
            g.prefill = (uint32_t)prefill;
            g.records = (uint32_t)steps;
            g.config  = env_snapshot();
            g.tokens  = tokens;
            g.logits  = std::move(logits);
            if (!write_golden(out_path, g)) {
                std::fprintf(stderr, "failed to write %s\n", out_path.c_str());
                return 1;
            }
            double ppl = perplexity(g.logits, g.records, vocab, tokens,
                                    (uint32_t)prefill);
            std::printf("[verify] wrote %s (%.1f MiB)\n", out_path.c_str(),
                        (double)g.logits.size() * 4 / (1 << 20));
            std::printf("[verify] perplexity %.4f%s\n", ppl,
                        real_text ? "" : "  (synthetic tokens — not meaningful)");
            return 0;
        }

        // ---- compare -------------------------------------------------------
        double max_abs = 0.0, sum_abs = 0.0;
        int    max_abs_step = -1;
        size_t agree = 0;
        double max_kl = 0.0, sum_kl = 0.0;

        for (uint32_t r = 0; r < (uint32_t)steps; ++r) {
            const float* a = golden.logits.data() + (size_t)r * vocab;
            const float* b = logits.data() + (size_t)r * vocab;

            for (uint32_t v = 0; v < vocab; ++v) {
                double d = std::fabs((double)a[v] - (double)b[v]);
                sum_abs += d;
                if (d > max_abs) { max_abs = d; max_abs_step = (int)r; }
            }
            if (argmax(a, (int)vocab) == argmax(b, (int)vocab)) ++agree;

            // KL(golden || current): the distributional distance that actually
            // tracks output quality, unlike a max over raw logits which a
            // single irrelevant token can dominate.
            double lse_a = logsumexp(a, (int)vocab);
            double lse_b = logsumexp(b, (int)vocab);
            double kl = 0.0;
            for (uint32_t v = 0; v < vocab; ++v) {
                double log_pa = (double)a[v] - lse_a;
                if (log_pa < -30.0) continue;    // contributes < 1e-13
                kl += std::exp(log_pa) * (log_pa - ((double)b[v] - lse_b));
            }
            sum_kl += kl;
            if (kl > max_kl) max_kl = kl;
        }

        double top1 = (double)agree / steps;
        double ppl_golden  = perplexity(golden.logits, (uint32_t)steps, vocab,
                                        tokens, (uint32_t)prefill);
        double ppl_current = perplexity(logits, (uint32_t)steps, vocab,
                                        tokens, (uint32_t)prefill);

        std::printf("\ngolden config : %s\n", golden.config.c_str());
        std::printf("current config: %s\n\n", env_snapshot().c_str());
        std::printf("  records            %d\n", steps);
        std::printf("  max |Δlogit|       %.6g  (step %d)\n", max_abs, max_abs_step);
        std::printf("  mean |Δlogit|      %.6g\n", sum_abs / ((double)steps * vocab));
        std::printf("  top-1 agreement    %.4f  (%zu/%d)\n", top1, agree, steps);
        std::printf("  KL max / mean      %.6g / %.6g nats\n", max_kl, sum_kl / steps);
        std::printf("  perplexity         %.4f golden -> %.4f current  (%+.4f)\n",
                    ppl_golden, ppl_current, ppl_current - ppl_golden);

        bool ok = true;
        if (max_abs > tol_abs) {
            std::printf("\nFAIL: max |Δlogit| %.6g exceeds --tol-abs %.6g\n",
                        max_abs, tol_abs);
            ok = false;
        }
        if (top1 < tol_top1) {
            std::printf("FAIL: top-1 agreement %.4f below --tol-top1 %.4f\n",
                        top1, tol_top1);
            ok = false;
        }
        if (ok) std::printf("\nPASS\n");
        return ok ? 0 : 1;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
