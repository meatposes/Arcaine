// src/bench/qwen35_dense_projection_bench.cpp
//
// Dense NVFP4 projection roofline at Qwen3.5's real shapes, against a BF16
// reference of the same logical shape.
//
// The question this exists to answer: at M=1, is the NVFP4 path limited by the
// bytes it reads or by decompressing them? End-to-end profiling put decode's
// NVFP4 GEMVs near 158 GB/s while the BF16 lm_head GEMV in the same run reached
// roughly 600 GB/s, but those are different shapes, so the comparison proves
// nothing on its own. Here both dtypes run the same K:N with the same M, so the
// only difference is the weight format.
//
// BF16 reads 4x the bytes of NVFP4 for the same shape. If NVFP4 is
// bandwidth-bound its GB/s should match BF16's and its wall time should be ~4x
// lower; if it is decompression-bound its GB/s will be far below BF16's, and
// the headroom is real rather than a property of the card.
//
// `nvfp4` is what decode runs today (pack the activation, then the packed
// matmul). `nvfp4-nopack` skips the pack to size that step separately, since it
// is charged to every projection under W4A4.
//
// Random weights: this measures time, not numerics, and the values do not
// affect either kernel's cost. DIFF_NVFP4_WEIGHT_LAYOUT must stay unset (Raw)
// so random weights are not oneDNN-reordered.
//
// Registered as `qwen35-dense-projection` in the arcaine_kbench binary.
//
// Run:
//   ONEAPI_DEVICE_SELECTOR=level_zero:2 ./build/arcaine_kbench \
//       qwen35-dense-projection -p 1,8,512
//
// One shape only, more repetitions:
//   ./build/arcaine_kbench qwen35-dense-projection --shapes gate_up -p 1 -r 50

#include "common/bench/registry.hpp"
#include "common/bench/util.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <random>
#include <string>
#include <vector>

#include "common/gpu/engine.hpp"
#include "common/gpu/nvfp4.hpp"
#include "common/gpu/ops.hpp"

namespace {

using arcaine::bench::aggregate;
using arcaine::bench::elapsed_ms;
using arcaine::bench::parse_int_csv;
using arcaine::bench::split_csv;
using arcaine::bench::Stat;

// Every dense projection the qwen3_5 forward path drives, at this checkpoint's
// config (hidden 5120, intermediate 17408, 24 q heads / 4 kv heads x 256,
// linear-attn key 16x128 value 48x128, vocab 248320).
struct Shape {
    const char* name;
    int K;
    int N;
};
const Shape kShapes[] = {
    {"gate_up",      5120,  34816},  // fused gate+up
    {"down",        17408,   5120},
    {"qkv",          5120,  14336},  // q(2x24x256) + k + v, fused
    {"o_proj",       6144,   5120},
    {"in_proj_qkv",  5120,  10240},
    {"in_proj_z",    5120,   6144},
    {"out_proj",     6144,   5120},
    {"lm_head",      5120, 248320},
};

enum class Kernel { Nvfp4, Nvfp4NoPack, Bf16, DequantBf16 };

const char* kernel_name(Kernel k) {
    switch (k) {
        case Kernel::Nvfp4:       return "nvfp4";
        case Kernel::Nvfp4NoPack: return "nvfp4-nopack";
        case Kernel::Bf16:        return "bf16";
        case Kernel::DequantBf16: return "dequant+bf16";
    }
    return "?";
}

// Expand an NVFP4 weight into a BF16 scratch buffer, then let the BF16 GEMM run.
// Deliberately naive -- one work item per packed byte -- because the point is to
// price the strategy, not to optimize the expansion. If a throwaway elementwise
// kernel plus a library GEMM already beats oneDNN's f4 matmul, that is worth
// knowing before anyone writes a real f4 kernel.
//
// Reconstruction matches the matmul convention: oneDNN's DST scale divides, so
// w = e2m1(nibble) * e4m3(group scale) / weight_global_scale. Scales arrive
// transposed to [K/16][N], as upload_nvfp4_scales_transposed leaves them.
void dequantize_nvfp4_weight(sycl::queue& q, const uint8_t* packed,
                             const uint8_t* scales, bf16* out,
                             int N, int K, float weight_global_scale) {
    const size_t bytes = (size_t)N * (size_t)K / 2;
    const int half_k = K / 2;
    const float inv_global = 1.0f / weight_global_scale;
    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>(bytes), [=](sycl::id<1> id) {
            const size_t b = id[0];
            const int n = (int)(b / (size_t)half_k);
            const int pair = (int)(b % (size_t)half_k);
            const int k0 = pair * 2;
            const int group = k0 / 16;
            const float scale =
                nvfp4_e4m3_fast(scales[(size_t)group * N + n]) * inv_global;
            const uint8_t byte = packed[b];
            const size_t base = (size_t)n * K + k0;
            out[base]     = float_to_bf16(nvfp4_e2m1_fast(byte & 0x0f) * scale);
            out[base + 1] = float_to_bf16(nvfp4_e2m1_fast(byte >> 4) * scale);
        });
    });
}

// Weight bytes moved per call. Activations are negligible at these M and are
// excluded so the number is comparable across dtypes.
double weight_bytes(Kernel k, int K, int N) {
    const double elements = (double)K * (double)N;
    const double packed = elements * 0.5 + elements / 16.0;
    const double dense = elements * 2.0;
    switch (k) {
        case Kernel::Bf16: return dense;
        // Read the packed weight, write the expansion, then read it back in the
        // GEMM. Counting all three is what makes the comparison honest: the
        // strategy trades traffic for a kernel that uses XMX.
        case Kernel::DequantBf16: return packed + dense * 2.0;
        default: return packed;
    }
}

const char* kUsage =
    "Usage: qwen35-dense-projection [options]\n"
    "  -p <csv>          M values to sweep (default 1,8,512)\n"
    "  --shapes <csv>    projection names or K:N (default: all)\n"
    "  --kernels <csv>   nvfp4,nvfp4-nopack,bf16,dequant+bf16 (default: all)\n"
    "  -r <N>            timed iterations after warmup (default 20)\n"
    "  -w <N>            warmup iterations (default 3)\n"
    "  --seed <N>        RNG seed (default 42)\n"
    "  --check           dequant+bf16 vs nvfp4 deviation (W4A16 vs W4A4, see note)\n"
    "  -h, --help        show this help\n";

}  // namespace

static int run(int argc, char** argv) {
    std::vector<int> ms{1, 8, 512};
    std::vector<std::string> shape_names;
    std::vector<std::string> kernel_names{"nvfp4", "nvfp4-nopack", "bf16",
                                         "dequant+bf16"};
    int iters = 20;
    int warmup = 3;
    unsigned seed = 42;
    bool check = false;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) { std::fputs(kUsage, stderr); std::exit(1); }
            return argv[++i];
        };
        if (a == "-p")            ms = parse_int_csv(next());
        else if (a == "--shapes") shape_names = split_csv(next());
        else if (a == "--kernels") kernel_names = split_csv(next());
        else if (a == "-r")       iters = std::atoi(next().c_str());
        else if (a == "-w")       warmup = std::atoi(next().c_str());
        else if (a == "--seed")   seed = (unsigned)std::atoi(next().c_str());
        else if (a == "--check")  check = true;
        else if (a == "-h" || a == "--help") { std::fputs(kUsage, stdout); return 0; }
        else { std::fprintf(stderr, "unknown argument: %s\n", a.c_str());
               std::fputs(kUsage, stderr); return 1; }
    }

    std::vector<Kernel> kernels;
    for (const std::string& name : kernel_names) {
        if (name == "nvfp4")             kernels.push_back(Kernel::Nvfp4);
        else if (name == "nvfp4-nopack") kernels.push_back(Kernel::Nvfp4NoPack);
        else if (name == "bf16")         kernels.push_back(Kernel::Bf16);
        else if (name == "dequant+bf16") kernels.push_back(Kernel::DequantBf16);
        else { std::fprintf(stderr, "unknown kernel: %s\n", name.c_str()); return 1; }
    }

    std::vector<Shape> shapes;
    if (shape_names.empty()) {
        for (const Shape& s : kShapes) shapes.push_back(s);
    } else {
        for (const std::string& name : shape_names) {
            size_t colon = name.find(':');
            if (colon != std::string::npos) {
                shapes.push_back({"custom", std::atoi(name.substr(0, colon).c_str()),
                                  std::atoi(name.substr(colon + 1).c_str())});
                continue;
            }
            bool found = false;
            for (const Shape& s : kShapes)
                if (name == s.name) { shapes.push_back(s); found = true; break; }
            if (!found) { std::fprintf(stderr, "unknown shape: %s\n", name.c_str()); return 1; }
        }
    }

    if (const char* layout = std::getenv("DIFF_NVFP4_WEIGHT_LAYOUT"))
        std::fprintf(stderr,
            "[qwen35-dense-projection] WARNING: DIFF_NVFP4_WEIGHT_LAYOUT=%s is set;\n"
            "  random weights would be reordered and the NVFP4 timings invalid.\n", layout);

    GpuEngine& ctx = GpuEngine::get(0);
    sycl::queue& q = ctx.queue;
    std::printf("[qwen35-dense-projection] device: %s | iters=%d warmup=%d\n",
                q.get_device().get_info<sycl::info::device::name>().c_str(),
                iters, warmup);
    std::printf("%-14s %-13s %7s %6s %7s %11s %9s %9s\n",
                "shape", "kernel", "K", "N", "M", "ms", "GB/s", "MB/call");

    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> nibble(0, 15), scale(40, 110);
    std::uniform_int_distribution<int> bf16_bits(0x3d00, 0x3f00);

    for (const Shape& shape : shapes) {
        const int K = shape.K;
        const int N = shape.N;
        if (K % 16 != 0) {
            std::fprintf(stderr, "skipping %s: K=%d not divisible by 16\n", shape.name, K);
            continue;
        }
        const int G = K / 16;

        Nvfp4Linear w4;
        w4.in_features = K;
        w4.out_features = N;
        w4.input_global_scale = 1.0f;
        w4.weight_global_scale = 1.0f;
        {
            std::vector<uint8_t> packed((size_t)N * K / 2);
            std::vector<uint8_t> scales((size_t)G * N);
            for (auto& b : packed) b = (uint8_t)(nibble(rng) | (nibble(rng) << 4));
            for (auto& b : scales) b = (uint8_t)scale(rng);
            w4.weight_packed = GpuBuffer<uint8_t>(packed.size(), q);
            w4.weight_packed.upload(packed.data(), packed.size());
            w4.weight_scale = GpuBuffer<uint8_t>(scales.size(), q);
            w4.weight_scale.upload(scales.data(), scales.size());
            w4.dst_scale = GpuBuffer<float>(1, q);
            const float one = 1.0f;
            w4.dst_scale.upload(&one, 1);
        }

        GpuBuffer<bf16> w16((size_t)N * K, q);
        {
            std::vector<bf16> host((size_t)N * K);
            for (auto& v : host) v = (bf16)bf16_bits(rng);
            w16.upload(host.data(), host.size());
        }

        for (int M : ms) {
            GpuBuffer<bf16> A((size_t)M * K, q);
            {
                std::vector<bf16> host((size_t)M * K);
                for (auto& v : host) v = (bf16)bf16_bits(rng);
                A.upload(host.data(), host.size());
            }
            GpuBuffer<bf16> C((size_t)M * N, q);
            GpuBuffer<uint8_t> a_packed((size_t)M * K / 2, q);
            GpuBuffer<uint8_t> a_scale((size_t)M * G, q);
            // Scratch for the dequant strategy: one projection's expansion, so
            // the footprint is bounded by the largest weight, not the model.
            GpuBuffer<bf16> expanded((size_t)N * K, q);
            // Pre-pack once for the nopack variant.
            pack_bf16_to_nvfp4(q, A.data(), a_packed.data(), a_scale.data(), M, K,
                               w4.input_global_scale);
            q.wait();

            // NOT a weight-rounding check, and the numbers must not be read as
            // one. The nvfp4 path is W4A4 -- pack_bf16_to_nvfp4 quantizes the
            // activation to FP4 -- while dequant+bf16 feeds the activation in
            // full BF16. The two compute different functions, and the deviation
            // below is dominated by activation quantization, not by rounding the
            // weight to BF16.
            //
            // It is reported because the direction is the useful part: switching
            // a projection to dequant+bf16 moves it from W4A4 to W4A16, which is
            // more accurate, not less. Isolating weight rounding needs an
            // activation drawn from the exactly-representable FP4 magnitudes so
            // the pack is lossless; until that exists, treat this as a wiring
            // check that the two paths produce the same order of magnitude.
            if (check) {
                std::vector<bf16> ref((size_t)M * N), alt((size_t)M * N);
                matmul_nvfp4_packed(a_packed.data(), a_scale.data(), M, K, w4,
                                    C.data(), ctx);
                q.wait();
                C.download(ref.data(), ref.size());
                dequantize_nvfp4_weight(q, w4.weight_packed.data(),
                                        w4.weight_scale.data(), expanded.data(),
                                        N, K, w4.weight_global_scale);
                matmul_bf16(A.data(), M, K, expanded.data(), N, C.data(), ctx);
                q.wait();
                C.download(alt.data(), alt.size());
                double max_abs = 0.0, max_rel = 0.0;
                for (size_t i = 0; i < ref.size(); ++i) {
                    const float a = bf16_to_float(ref[i]);
                    const float b = bf16_to_float(alt[i]);
                    const double abs_err = std::fabs((double)a - (double)b);
                    max_abs = std::max(max_abs, abs_err);
                    const double denom = std::fabs((double)a);
                    if (denom > 1e-3) max_rel = std::max(max_rel, abs_err / denom);
                }
                std::printf("%-14s %-13s %7d %6d %7d   max_abs=%.6g max_rel=%.6g"
                            "  (W4A16 vs W4A4, not weight rounding)\n",
                            shape.name, "check", K, N, M, max_abs, max_rel);
            }

            for (Kernel k : kernels) {
                auto once = [&] {
                    switch (k) {
                        case Kernel::Nvfp4:
                            pack_bf16_to_nvfp4(q, A.data(), a_packed.data(),
                                               a_scale.data(), M, K,
                                               w4.input_global_scale);
                            matmul_nvfp4_packed(a_packed.data(), a_scale.data(),
                                                M, K, w4, C.data(), ctx);
                            break;
                        case Kernel::Nvfp4NoPack:
                            matmul_nvfp4_packed(a_packed.data(), a_scale.data(),
                                                M, K, w4, C.data(), ctx);
                            break;
                        case Kernel::Bf16:
                            matmul_bf16(A.data(), M, K, w16.data(), N, C.data(), ctx);
                            break;
                        case Kernel::DequantBf16:
                            dequantize_nvfp4_weight(q, w4.weight_packed.data(),
                                                    w4.weight_scale.data(),
                                                    expanded.data(), N, K,
                                                    w4.weight_global_scale);
                            matmul_bf16(A.data(), M, K, expanded.data(), N,
                                        C.data(), ctx);
                            break;
                    }
                };
                for (int i = 0; i < warmup; ++i) once();
                q.wait();

                std::vector<double> samples;
                samples.reserve(iters);
                for (int i = 0; i < iters; ++i)
                    samples.push_back(elapsed_ms(q, 1, once));
                Stat stat = aggregate(samples);

                const double bytes = weight_bytes(k, K, N);
                const double gbps = stat.mean > 0.0
                    ? bytes / (stat.mean * 1e-3) / 1e9 : 0.0;
                std::printf("%-14s %-13s %7d %6d %7d %8.4f%s %9.1f %9.1f\n",
                            shape.name, kernel_name(k), K, N, M, stat.mean,
                            stat.sd > 0.05 * stat.mean ? "*" : " ",
                            gbps, bytes / 1e6);
            }
        }
    }
    std::printf("\nGB/s counts weight bytes only. nvfp4 moves ~1/4 of bf16 for the\n"
                "same shape, so equal GB/s means equal efficiency per byte and a ~4x\n"
                "wall-time win; lower GB/s means decompression, not bandwidth, is the\n"
                "limit. '*' marks sd above 5%% of the mean.\n");
    return 0;
}

REGISTER_BENCH("qwen35-dense-projection",
    "Dense NVFP4 vs BF16 projection roofline at Qwen3.5 shapes (random weights)",
    run)
