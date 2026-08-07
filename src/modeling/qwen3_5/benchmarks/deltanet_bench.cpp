// Qwen3.5 Gated DeltaNet recurrent-core benchmark. Compares the scalar/SIMT and
// ESIMD paths at the checkpoint's 48x128x128 shape, including sequential decode,
// without running end-to-end inference. Registered as `qwen35-deltanet` in the
// unified kernel_bench binary.
//
// Run:
//   ./build/kernel_bench qwen35-deltanet [opts]

#include "benchmarks/registry.hpp"
#include "benchmarks/util.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

#include "runtime/gpu/buffer.hpp"
#include "runtime/gpu/device_select.hpp"
#include "runtime/gpu/engine.hpp"
#include "modeling/qwen3_5/kernels.hpp"

using arcaine::bench::aggregate;
using arcaine::bench::parse_int_csv;
using arcaine::bench::split_csv;
using arcaine::bench::Stat;

namespace {

void usage(const char* program) {
    std::fprintf(stderr,
        "Usage: %s [options]\n"
        "  --p <csv>          prefill lengths       (default: 512,1024)\n"
        "  --n <N>            sequential decode tokens (default: 32)\n"
        "  --w <N>            warmup runs per cell (default: 1)\n"
        "  --r <N>            timed runs per cell  (default: 5)\n"
        "  --kernels <csv>    baseline,esimd       (default: baseline,esimd)\n"
        "  --device <N>       visible Level Zero GPU\n",
        program);
}

int run(int argc, char** argv) {
    std::string p_csv = "512,1024";
    std::string kernels_csv = "baseline,esimd";
    std::string device;
    int decode_tokens = 32;
    int warmup = 1;
    int runs = 5;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) throw std::runtime_error("missing value for " + arg);
            return argv[++i];
        };
        if (arg == "-h" || arg == "--help") { usage(argv[0]); return 0; }
        else if (arg == "-p" || arg == "--p") p_csv = next();
        else if (arg == "-n" || arg == "--n") decode_tokens = std::stoi(next());
        else if (arg == "-w" || arg == "--w") warmup = std::stoi(next());
        else if (arg == "-r" || arg == "--r") runs = std::stoi(next());
        else if (arg == "--kernels") kernels_csv = next();
        else if (arg == "--device") device = next();
        else { std::fprintf(stderr, "unknown arg: %s\n", arg.c_str()); return 1; }
    }
    if (!device.empty()) gpu_device_control::apply_device_index(device);
    std::vector<int> prefills = parse_int_csv(p_csv);
    std::vector<std::string> kernels = split_csv(kernels_csv);
    if (prefills.empty() || kernels.empty() || decode_tokens <= 0 ||
        warmup < 0 || runs <= 0)
        throw std::runtime_error("invalid benchmark arguments");

    constexpr int heads = 48;
    constexpr int key_dim = 128;
    constexpr int value_dim = 128;
    int max_tokens = std::max(decode_tokens,
        *std::max_element(prefills.begin(), prefills.end()));
    size_t vectors = (size_t)max_tokens * heads * key_dim;
    size_t state_values = (size_t)heads * key_dim * value_dim;
    auto& queue = GpuEngine::get(0).queue;

    std::vector<bf16> host_q(vectors), host_k(vectors), host_v(vectors);
    std::vector<bf16> host_beta((size_t)max_tokens * heads);
    std::vector<bf16> host_g(host_beta.size());
    uint32_t random = 0x35d31a5u;
    auto sample = [&]() {
        random = random * 1664525u + 1013904223u;
        return (static_cast<int>((random >> 16) & 2047u) - 1024) / 32768.0f;
    };
    for (int token = 0; token < max_tokens; ++token) {
        for (int head = 0; head < heads; ++head) {
            float q_norm = 0.0f, k_norm = 0.0f;
            size_t base = ((size_t)token * heads + head) * key_dim;
            for (int dim = 0; dim < key_dim; ++dim) {
                float q = sample(), k = sample();
                host_q[base + dim] = float_to_bf16(q);
                host_k[base + dim] = float_to_bf16(k);
                host_v[base + dim] = float_to_bf16(sample());
                q_norm += q * q;
                k_norm += k * k;
            }
            float q_scale = 0.08838834764831845f / std::sqrt(q_norm + 1e-6f);
            float k_scale = 1.0f / std::sqrt(k_norm + 1e-6f);
            for (int dim = 0; dim < key_dim; ++dim) {
                host_q[base + dim] = float_to_bf16(
                    bf16_to_float(host_q[base + dim]) * q_scale);
                host_k[base + dim] = float_to_bf16(
                    bf16_to_float(host_k[base + dim]) * k_scale);
            }
            size_t gate = (size_t)token * heads + head;
            host_beta[gate] = float_to_bf16(0.25f + std::fabs(sample()) * 8.0f);
            host_g[gate] = float_to_bf16(-0.001f - std::fabs(sample()) * 0.25f);
        }
    }

    GpuBuffer<bf16> q(host_q.size(), queue), k(host_k.size(), queue),
                      v(host_v.size(), queue), beta(host_beta.size(), queue),
                      g(host_g.size(), queue), output(vectors, queue),
                      reference(vectors, queue);
    GpuBuffer<float> baseline_state(state_values, queue),
                     esimd_state(state_values, queue);
    q.upload(host_q.data(), host_q.size());
    k.upload(host_k.data(), host_k.size());
    v.upload(host_v.data(), host_v.size());
    beta.upload(host_beta.data(), host_beta.size());
    g.upload(host_g.data(), host_g.size());

    auto reset = [&](const std::string& kernel) {
        if (kernel == "baseline") baseline_state.zero();
        else if (kernel == "esimd") esimd_state.zero();
        else throw std::runtime_error("unknown kernel: " + kernel);
    };
    auto run_prefill = [&](const std::string& kernel, bf16* destination, int seq) {
        if (kernel == "baseline")
            qwen35_recurrent_delta(queue, q.data(), k.data(), v.data(), beta.data(),
                                   g.data(), baseline_state.data(), destination, seq,
                                   heads, key_dim, value_dim);
        else if (kernel == "esimd")
            qwen35_recurrent_delta_esimd(queue, q.data(), k.data(), v.data(),
                                         beta.data(), g.data(), esimd_state.data(),
                                         destination, seq, heads, key_dim, value_dim);
        else
            throw std::runtime_error("unknown kernel: " + kernel);
    };
    auto run_decode = [&](const std::string& kernel, bf16* destination, int tokens) {
        for (int token = 0; token < tokens; ++token) {
            size_t vector_offset = (size_t)token * heads * key_dim;
            size_t gate_offset = (size_t)token * heads;
            if (kernel == "baseline")
                qwen35_recurrent_delta(
                    queue, q.data() + vector_offset, k.data() + vector_offset,
                    v.data() + vector_offset, beta.data() + gate_offset,
                    g.data() + gate_offset, baseline_state.data(),
                    destination + vector_offset, 1, heads, key_dim, value_dim);
            else if (kernel == "esimd")
                qwen35_recurrent_delta_esimd(
                    queue, q.data() + vector_offset, k.data() + vector_offset,
                    v.data() + vector_offset, beta.data() + gate_offset,
                    g.data() + gate_offset, esimd_state.data(),
                    destination + vector_offset, 1, heads, key_dim, value_dim);
            else
                throw std::runtime_error("unknown kernel: " + kernel);
        }
    };

    auto correctness = [&](int tokens, bool decode) {
        reset("baseline");
        if (decode) run_decode("baseline", reference.data(), tokens);
        else run_prefill("baseline", reference.data(), tokens);
        queue.wait();
        reset("esimd");
        if (decode) run_decode("esimd", output.data(), tokens);
        else run_prefill("esimd", output.data(), tokens);
        queue.wait();
        size_t count = (size_t)tokens * heads * value_dim;
        std::vector<bf16> expected(count), actual(count);
        reference.download(expected.data(), count);
        output.download(actual.data(), count);
        float max_abs = 0.0f, max_rel = 0.0f;
        for (size_t index = 0; index < count; ++index) {
            float e = bf16_to_float(expected[index]);
            float a = bf16_to_float(actual[index]);
            float error = std::fabs(e - a);
            max_abs = std::max(max_abs, error);
            max_rel = std::max(max_rel, error / std::max(1e-3f, std::fabs(e)));
        }
        return std::pair<float, float>{max_abs, max_rel};
    };

    // ---------------------------------------------------------------------
    // fp64 CPU oracle for the gated delta rule.
    //
    // `correctness` above compares the ESIMD kernel against the scalar kernel,
    // which establishes only that they differ - it names neither as wrong.
    // Two implementations summing 128 products in different orders are
    // *expected* to disagree, and in a recurrence that disagreement compounds,
    // so their distance carries no verdict on its own.
    //
    // This computes the same recurrence in double precision on the host from
    // the identical bf16 inputs the kernels receive, so it is the value both
    // are approximating. Distance from it is each kernel's own arithmetic
    // error, and the two can finally be ranked rather than merely contrasted.
    //
    // It is O(seq * heads * value_dim * key_dim) scalar work - fine for the
    // sequence lengths a correctness check needs, far too slow for a sweep.
    auto oracle = [&](int tokens) {
        std::vector<double> out((size_t)tokens * heads * value_dim);
        std::vector<double> state((size_t)key_dim * value_dim);
        for (int head = 0; head < heads; ++head) {
            std::fill(state.begin(), state.end(), 0.0);
            for (int token = 0; token < tokens; ++token) {
                size_t qk = ((size_t)token * heads + head) * key_dim;
                size_t vb = ((size_t)token * heads + head) * value_dim;
                size_t gate = (size_t)token * heads + head;
                double decay = std::exp((double)bf16_to_float(host_g[gate]));
                double b = (double)bf16_to_float(host_beta[gate]);
                for (int value_index = 0; value_index < value_dim; ++value_index) {
                    double memory = 0.0;
                    for (int dim = 0; dim < key_dim; ++dim) {
                        double& cell = state[(size_t)dim * value_dim + value_index];
                        cell *= decay;
                        memory += cell * (double)bf16_to_float(host_k[qk + dim]);
                    }
                    double delta =
                        ((double)bf16_to_float(host_v[vb + value_index]) - memory) * b;
                    double result = 0.0;
                    for (int dim = 0; dim < key_dim; ++dim) {
                        double& cell = state[(size_t)dim * value_dim + value_index];
                        cell += (double)bf16_to_float(host_k[qk + dim]) * delta;
                        result += cell * (double)bf16_to_float(host_q[qk + dim]);
                    }
                    out[vb + value_index] = result;
                }
            }
        }
        return out;
    };

    auto against_oracle = [&](int tokens, bool decode) {
        std::vector<double> truth = oracle(tokens);
        size_t count = (size_t)tokens * heads * value_dim;
        std::vector<bf16> actual(count);
        std::printf("[oracle] fp64 host reference, tokens=%d decode=%d\n",
                    tokens, decode ? 1 : 0);
        for (const auto& kernel : kernels) {
            reset(kernel);
            if (decode) run_decode(kernel, output.data(), tokens);
            else run_prefill(kernel, output.data(), tokens);
            queue.wait();
            output.download(actual.data(), count);
            double max_abs = 0.0, sum_sq = 0.0, scale = 0.0;
            for (size_t index = 0; index < count; ++index) {
                double t = truth[index];
                double a = (double)bf16_to_float(actual[index]);
                double error = std::fabs(t - a);
                max_abs = std::max(max_abs, error);
                sum_sq += error * error;
                scale = std::max(scale, std::fabs(t));
            }
            // The kernels emit bf16, which alone costs ~2^-8 relative. Quoting
            // error against that floor says whether a kernel is merely storing
            // its answer in bf16 or is actually computing a different one.
            double rms = std::sqrt(sum_sq / (double)count);
            std::printf("  kernel=%-8s max_abs=%.6f rms=%.6f peak|truth|=%.4f "
                        "max_abs/bf16_ulp_of_peak=%.2f\n",
                        kernel.c_str(), max_abs, rms, scale,
                        max_abs / std::max(1e-30, scale * 0.00390625));
        }
    };

    std::printf("[bench] Qwen3.5 DeltaNet core: heads=48 K=128 V=128 state=FP32\n");
    if (const char* value = std::getenv("ARCAINE_DELTANET_ORACLE")) {
        if (std::atoi(value) != 0) {
            int tokens = std::min(prefills.front(), 32);
            against_oracle(tokens, false);
            against_oracle(std::min(decode_tokens, 32), true);
        }
    }
    auto benchmark = [&](const char* kind, int tokens, bool decode) {
        auto errors = correctness(tokens, decode);
        for (const auto& kernel : kernels) {
            for (int i = 0; i < warmup; ++i) {
                reset(kernel);
                if (decode) run_decode(kernel, output.data(), tokens);
                else run_prefill(kernel, output.data(), tokens);
            }
            queue.wait();
            std::vector<double> samples;
            for (int i = 0; i < runs; ++i) {
                reset(kernel);
                auto start = std::chrono::steady_clock::now();
                if (decode) run_decode(kernel, output.data(), tokens);
                else run_prefill(kernel, output.data(), tokens);
                queue.wait();
                samples.push_back(std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - start).count());
            }
            Stat stat = aggregate(samples);
            double per_token = stat.mean / tokens;
            std::printf("qwen35_deltanet kind=%s kernel=%s tokens=%d mean_ms=%.3f "
                        "sd_ms=%.3f ms_per_token=%.6f max_abs=%.6f max_rel=%.6f\n",
                        kind, kernel.c_str(), tokens, stat.mean, stat.sd,
                        per_token, errors.first, errors.second);
        }
    };

    for (int seq : prefills) benchmark("prefill", seq, false);
    benchmark("decode", decode_tokens, true);
    return 0;
}

}  // namespace

REGISTER_BENCH("qwen35-deltanet",
    "Qwen3.5 Gated DeltaNet recurrent core (baseline vs ESIMD) at 48x128x128",
    run)

