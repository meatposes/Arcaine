// Qwen3.5 full-attention kernel benchmark. Compares the scalar online-softmax
// baseline with subgroup and XMX/DPAS paths at 24Q/4KV/D256. Registered as
// `qwen35-attention` in the unified kernel_bench binary.
//
// Run:
//   ./build/kernel_bench qwen35-attention [opts]

#include "benchmarks/registry.hpp"
#include "benchmarks/util.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <cstring>
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
        "  --p <csv>          prefill query lengths       (default: 128,512)\n"
        "  --d <csv>          decode KV depths           (default: 0,512,1024)\n"
        "  --w <N>            warmup runs per cell       (default: 1)\n"
        "  --r <N>            timed runs per cell        (default: 5)\n"
        "  --kernels <csv>    baseline,subgroup,xmx,xmx-gqa (default: baseline,xmx)\n"
        "  --device <N>       visible Level Zero GPU\n",
        program);
}

int run(int argc, char** argv) {
    std::string p_csv = "128,512";
    std::string d_csv = "0,512,1024";
    std::string kernels_csv = "baseline,xmx";
    std::string device;
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
        else if (arg == "-d" || arg == "--d") d_csv = next();
        else if (arg == "-w" || arg == "--w") warmup = std::stoi(next());
        else if (arg == "-r" || arg == "--r") runs = std::stoi(next());
        else if (arg == "--kernels") kernels_csv = next();
        else if (arg == "--device") device = next();
        else { std::fprintf(stderr, "unknown arg: %s\n", arg.c_str()); return 1; }
    }
    if (!device.empty()) gpu_device_control::apply_device_index(device);
    std::vector<int> prefills = parse_int_csv(p_csv);
    std::vector<int> depths = parse_int_csv(d_csv);
    std::vector<std::string> kernels = split_csv(kernels_csv);
    if (prefills.empty() || depths.empty() || kernels.empty() || warmup < 0 || runs <= 0)
        throw std::runtime_error("invalid benchmark arguments");

    constexpr int query_heads = 24;
    constexpr int key_heads = 4;
    constexpr int head_dim = 256;
    constexpr float scale = 0.0625f;
    int max_query = *std::max_element(prefills.begin(), prefills.end());
    int max_depth = *std::max_element(depths.begin(), depths.end());
    int max_kv = std::max(max_query, max_depth + 1);
    auto& queue = GpuEngine::get(0).queue;

    std::vector<bf16> host_q((size_t)max_query * query_heads * head_dim);
    std::vector<bf16> host_k((size_t)max_kv * key_heads * head_dim);
    std::vector<bf16> host_v(host_k.size());
    uint32_t state = 42;
    auto fill = [&](std::vector<bf16>& values) {
        for (bf16& value : values) {
            state = state * 1664525u + 1013904223u;
            float x = (static_cast<int>((state >> 16) & 1023u) - 512) / 1024.0f;
            value = float_to_bf16(x);
        }
    };
    fill(host_q); fill(host_k); fill(host_v);
    GpuBuffer<bf16> q(host_q.size(), queue), k(host_k.size(), queue),
                      v(host_v.size(), queue);
    GpuBuffer<bf16> output((size_t)max_query * query_heads * head_dim, queue);
    GpuBuffer<bf16> reference(output.count(), queue);
    q.upload(host_q.data(), host_q.size());
    k.upload(host_k.data(), host_k.size());
    v.upload(host_v.data(), host_v.size());

    std::printf("[bench] Qwen3.5 full attention: q_heads=24 kv_heads=4 head_dim=256\n");
    auto run_kernel = [&](const std::string& kernel, bf16* destination,
                          int seq, int past) {
        if (kernel == "baseline")
            qwen35_online_attention(queue, q.data(), k.data(), v.data(), destination,
                                    seq, past, query_heads, key_heads, head_dim, scale);
        else if (kernel == "subgroup")
            qwen35_online_attention_subgroup(
                queue, q.data(), k.data(), v.data(), destination, seq, past,
                query_heads, key_heads, head_dim, scale);
        else if (kernel == "xmx")
            qwen35_xmx_attention(queue, q.data(), k.data(), v.data(), destination,
                                 seq, past, query_heads, key_heads, head_dim, scale);
        else if (kernel == "xmx-gqa" && seq == 1)
            qwen35_xmx_attention_decode_gqa(
                queue, q.data(), k.data(), v.data(), destination, past,
                query_heads, key_heads, head_dim, scale);
        else
            throw std::runtime_error("unknown kernel: " + kernel);
    };

    // ---------------------------------------------------------------------
    // fp64 CPU oracle for causal GQA attention.
    //
    // `reference` below is the baseline GPU kernel, so the comparison it feeds
    // establishes only that two kernels differ - it cannot say which is right,
    // and two implementations reducing 256-element dot products and a softmax
    // in different orders are expected to differ.
    //
    // This computes softmax(QK^T * scale + causal mask) V in double precision
    // from the same bf16 inputs, so each kernel can be ranked against the value
    // it is approximating.
    //
    // Query i sits at absolute position past + i and may attend to keys
    // 0..past+i inclusive. Query head h reads kv head h / (query_heads /
    // key_heads). Getting either of those wrong is the failure this is here to
    // catch, so both are written out rather than shared with the kernels.
    auto oracle = [&](int seq, int past) {
        int group = query_heads / key_heads;
        std::vector<double> out((size_t)seq * query_heads * head_dim);
        for (int i = 0; i < seq; ++i) {
            int absolute = past + i;
            for (int h = 0; h < query_heads; ++h) {
                int kv = h / group;
                const bf16* qrow =
                    host_q.data() + ((size_t)i * query_heads + h) * head_dim;
                std::vector<double> logits((size_t)absolute + 1);
                double top = -std::numeric_limits<double>::infinity();
                for (int j = 0; j <= absolute; ++j) {
                    const bf16* krow =
                        host_k.data() + ((size_t)j * key_heads + kv) * head_dim;
                    double dot = 0.0;
                    for (int d = 0; d < head_dim; ++d)
                        dot += (double)bf16_to_float(qrow[d]) *
                               (double)bf16_to_float(krow[d]);
                    logits[j] = dot * (double)scale;
                    top = std::max(top, logits[j]);
                }
                double total = 0.0;
                for (int j = 0; j <= absolute; ++j) {
                    logits[j] = std::exp(logits[j] - top);
                    total += logits[j];
                }
                double* orow = out.data() + ((size_t)i * query_heads + h) * head_dim;
                for (int d = 0; d < head_dim; ++d) orow[d] = 0.0;
                for (int j = 0; j <= absolute; ++j) {
                    double weight = logits[j] / total;
                    const bf16* vrow =
                        host_v.data() + ((size_t)j * key_heads + kv) * head_dim;
                    for (int d = 0; d < head_dim; ++d)
                        orow[d] += weight * (double)bf16_to_float(vrow[d]);
                }
            }
        }
        return out;
    };

    auto against_oracle = [&](int seq, int past) {
        std::vector<double> truth = oracle(seq, past);
        size_t count = (size_t)seq * query_heads * head_dim;
        std::vector<bf16> actual(count);
        for (const auto& kernel : kernels) {
            if (kernel == "xmx-gqa" && seq != 1) continue;
            run_kernel(kernel, output.data(), seq, past);
            queue.wait();
            output.download(actual.data(), count);
            double max_abs = 0.0, sum_sq = 0.0, peak = 0.0;
            for (size_t index = 0; index < count; ++index) {
                double t = truth[index];
                double a = (double)bf16_to_float(actual[index]);
                double error = std::fabs(t - a);
                max_abs = std::max(max_abs, error);
                sum_sq += error * error;
                peak = std::max(peak, std::fabs(t));
            }
            // bf16 carries 8 mantissa bits, so storing a correct answer already
            // costs ~2^-8 relative. Quoting error in those units separates "this
            // kernel rounds its output" from "this kernel computes something
            // else". A structural error - wrong mask, wrong kv head - lands
            // orders of magnitude above 1.
            std::printf("  seq=%-4d past=%-6d kernel=%-9s max_abs=%.6f rms=%.6f "
                        "peak=%.4f ulps=%.2f\n",
                        seq, past, kernel.c_str(), max_abs,
                        std::sqrt(sum_sq / (double)count), peak,
                        max_abs / std::max(1e-30, peak * 0.00390625));
        }
    };

    auto benchmark_cell = [&](const char* kind, int seq, int past) {
        run_kernel("baseline", reference.data(), seq, past);
        queue.wait();
        std::vector<bf16> host_reference((size_t)seq * query_heads * head_dim);
        reference.download(host_reference.data(), host_reference.size());
        for (const std::string& kernel : kernels) {
            if (kernel == "xmx-gqa" && seq != 1) continue;
            run_kernel(kernel, output.data(), seq, past);
            queue.wait();
            std::vector<bf16> host_output(host_reference.size());
            output.download(host_output.data(), host_output.size());
            float max_abs = 0.0f;
            float max_rel = 0.0f;
            for (size_t i = 0; i < host_output.size(); ++i) {
                float expected = bf16_to_float(host_reference[i]);
                float actual = bf16_to_float(host_output[i]);
                float error = std::fabs(expected - actual);
                max_abs = std::max(max_abs, error);
                max_rel = std::max(max_rel, error / std::max(1e-3f, std::fabs(expected)));
            }
            for (int i = 0; i < warmup; ++i) run_kernel(kernel, output.data(), seq, past);
            queue.wait();
            std::vector<double> samples;
            for (int i = 0; i < runs; ++i) {
                auto start = std::chrono::steady_clock::now();
                run_kernel(kernel, output.data(), seq, past);
                queue.wait();
                samples.push_back(std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - start).count());
            }
            Stat stat = aggregate(samples);
            std::printf("qwen35_attention kind=%s kernel=%s q=%d kv=%d mean_ms=%.3f "
                        "sd_ms=%.3f max_abs=%.6f max_rel=%.6f\n",
                        kind, kernel.c_str(), seq, past + seq, stat.mean, stat.sd,
                        max_abs, max_rel);
        }
    };

    if (const char* value = std::getenv("ARCAINE_ATTENTION_ORACLE")) {
        if (std::atoi(value) != 0) {
            std::printf("[oracle] fp64 host reference, causal GQA\n");
            // Sequence lengths that are not multiples of the XMX kernel's
            // 8-query tile or its 16-wide subgroup, and depths that are not
            // multiples of either, because a ragged final tile and a
            // non-aligned `past` are where an indexing error would hide. The
            // one confirmed bug in this model was invisible at every shape but
            // the one nobody tested.
            for (int seq : {1, 2, 7, 8, 9, 15, 16, 17, 33})
                against_oracle(seq, 0);
            for (int past : {0, 1, 7, 15, 31, 100})
                against_oracle(1, past);
            for (int seq : {3, 8, 9})
                for (int past : {1, 13, 64})
                    against_oracle(seq, past);
            return 0;
        }
    }

    for (int seq : prefills) benchmark_cell("prefill", seq, 0);
    for (int depth : depths) benchmark_cell("decode", 1, depth);
    return 0;
}

}  // namespace

REGISTER_BENCH("qwen35-attention",
    "Qwen3.5 full-attention baseline/subgroup/xmx at 24Q/4KV/D256", run)

