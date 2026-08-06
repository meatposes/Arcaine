// bench.cpp — Gemma4 inference benchmark, styled after llama-bench.
//
// Measures real GPU throughput for prefill (PP) and decode (TG) at
// various KV-cache depths.  Each row times N actual forward passes and
// reports mean t/s ± σ so you can see how decode slows as the cache fills.
//
// Usage: ./build/bench --model <model_dir> [options]
//   -p, --p P,... prefill prompt sizes to test   (default: 128,512)
//   -n, --n N,... new-token counts to test       (default: 128)
//   -d D,...      starting KV-cache depth(s)     (default: 0,512,1024,2048)
//   -r, --r R     timed repetitions              (default: 3)
//   -w, --w W     discarded warmup runs          (default: 1)
//   --max-seq N   KvCache allocation size        (default: auto)
//   --device N    restrict visible GPUs to one Level Zero device

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <algorithm>

#include <memory>

#include <sycl/sycl.hpp>

#include "common/model_interface.hpp"
#include "common/registry.hpp"
#include "common/gpu/device_select.hpp"
#include "common/gpu/engine.hpp"

// ---------------------------------------------------------------------------
using Clk = std::chrono::high_resolution_clock;
using Ms  = std::chrono::duration<double, std::milli>;

static double now_ms() { return Ms(Clk::now().time_since_epoch()).count(); }

// Per-benchmark statistics computed from a set of wall-clock timings.
struct Stats {
    double mean_ms, sd_ms;     // raw timing
    double mean_tps, sd_tps;   // derived tokens/sec (each rep computed independently)
    double ms_per_tok;         // mean_ms / n_toks
    double med_ms, iqr_ms;     // robust equivalents; a single scheduling stall
                               // moves the mean but not the median
    int    n_toks;
};

// Linear-interpolated quantile over a copy the caller has already sorted.
static double quantile(const std::vector<double>& sorted, double q) {
    if (sorted.empty()) return 0.0;
    if (sorted.size() == 1) return sorted[0];
    double pos = q * (sorted.size() - 1);
    size_t lo = (size_t)pos;
    size_t hi = std::min(lo + 1, sorted.size() - 1);
    return sorted[lo] + (pos - lo) * (sorted[hi] - sorted[lo]);
}

static Stats compute_stats(const std::vector<double>& ms, int n_toks) {
    int n = (int)ms.size();

    // ms stats
    double s = 0, s2 = 0;
    for (double t : ms) { s += t; s2 += t * t; }
    double mean_ms = s / n;
    double sd_ms   = std::sqrt(std::max(0.0, s2 / n - mean_ms * mean_ms));

    // t/s stats (computed per-rep to get σ in t/s space)
    double ts = 0, ts2 = 0;
    for (double t : ms) {
        double v = n_toks / (t * 0.001);
        ts += v; ts2 += v * v;
    }
    double mean_tps = ts / n;
    double sd_tps   = std::sqrt(std::max(0.0, ts2 / n - mean_tps * mean_tps));

    std::vector<double> sorted(ms);
    std::sort(sorted.begin(), sorted.end());
    double med = quantile(sorted, 0.5);
    double iqr = quantile(sorted, 0.75) - quantile(sorted, 0.25);

    return {mean_ms, sd_ms, mean_tps, sd_tps, mean_ms / n_toks, med, iqr, n_toks};
}

// ---------------------------------------------------------------------------
// Measured device bandwidth.
//
// The roofline denominator has to be what the card actually sustains, not its
// datasheet figure: decode is a stream of large weight reads, so that is what
// is timed here. Reported alongside the achieved number so the gap between
// them is visible rather than assumed.
struct Bandwidth { double read_gbs = 0.0, copy_gbs = 0.0; };

// Two details decide whether this number means anything.
//
// The buffer is filled with incompressible noise. This GPU compresses memory
// losslessly, so a uniform or memset buffer reports ~2400 GB/s where the same
// probe over random data reports ~590 GB/s on the same card — the first figure
// is the compressor, not the bus. Weights are incompressible, so noise is the
// fill that matches the workload being measured.
//
// Every lane stores its own partial sum rather than writing under a condition.
// A guard like `if ((id & mask) == 0 && sum == magic)` short-circuits on the id
// term, which makes the accumulation dead code for most lanes and lets the
// compiler delete the loads entirely.
static Bandwidth measure_bandwidth(sycl::queue& q, size_t bytes, int reps) {
    using v4 = sycl::vec<float, 4>;
    const size_t n = bytes / sizeof(v4);
    const size_t threads = 1u << 20;
    v4* src = sycl::malloc_device<v4>(n, q);
    v4* dst = sycl::malloc_device<v4>(n, q);
    float* partial = sycl::malloc_device<float>(threads, q);
    if (!src || !dst || !partial) {
        sycl::free(src, q); sycl::free(dst, q); sycl::free(partial, q);
        return {};
    }

    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>(n), [=](sycl::id<1> id) {
            uint64_t x = (uint64_t)id[0] * 6364136223846793005ull +
                         1442695040888963407ull;
            x ^= x >> 33; x *= 0xff51afd7ed558ccdull; x ^= x >> 33;
            v4 v;
            for (int k = 0; k < 4; ++k)
                v[k] = (float)(int32_t)(uint32_t)(x >> (k * 8));
            src[id[0]] = v;
        });
    }).wait();

    auto read_once = [&] {
        q.submit([&](sycl::handler& h) {
            h.parallel_for(sycl::range<1>(threads), [=](sycl::id<1> id) {
                size_t i = id[0];
                v4 acc(0.f);
                for (size_t j = i; j < n; j += threads) acc += src[j];
                partial[i] = acc[0] + acc[1] + acc[2] + acc[3];
            });
        });
    };
    auto copy_once = [&] { q.memcpy(dst, src, n * sizeof(v4)); };

    auto time = [&](auto&& fn, double bytes_moved) {
        fn(); q.wait();                       // warm
        double t0 = now_ms();
        for (int r = 0; r < reps; ++r) fn();
        q.wait();
        double ms = now_ms() - t0;
        return bytes_moved * reps / (ms * 1e6);   // GB/s
    };

    Bandwidth bw;
    bw.read_gbs = time(read_once, (double)n * sizeof(v4));
    bw.copy_gbs = time(copy_once, (double)n * sizeof(v4) * 2.0);

    sycl::free(src, q); sycl::free(dst, q); sycl::free(partial, q);
    return bw;
}

// Parse "a,b,c" → {a,b,c}.
static std::vector<int> parse_list(const char* s) {
    std::vector<int> v;
    char buf[512]; strncpy(buf, s, 511); buf[511] = 0;
    for (char* tok = strtok(buf, ","); tok; tok = strtok(nullptr, ","))
        v.push_back(atoi(tok));
    return v;
}

// ---------------------------------------------------------------------------
static bool g_roofline = false;
// Bytes a decode step reads once, plus the per-cached-position KV cost, plus
// the bandwidth the roofline is measured against. Zero disables the columns.
static double g_fixed_bytes = 0.0, g_kv_bytes_per_pos = 0.0, g_peak_gbs = 0.0;

static void print_header() {
    printf("\n %-18s %9s   %10s   %7s   %9s   %8s",
           "test", "kv-depth", "t/s", "± sd", "ms/tok", "time(s)");
    if (g_roofline) printf("   %8s   %8s   %7s", "GB/tok", "GB/s", "%roof");
    printf("\n");
    printf(" %-18s %9s   %10s   %7s   %9s   %8s",
           "──────────────────", "─────────",
           "──────────", "───────", "─────────", "────────");
    if (g_roofline) printf("   %8s   %8s   %7s", "────────", "────────", "───────");
    printf("\n");
}

// `kv_depth` is the mean cache depth over the timed window, so the KV term
// reflects what was actually re-read rather than the starting depth.
static void print_row(const char* test, const char* depth,
                      const Stats& s, bool skipped = false,
                      double kv_depth = -1.0) {
    if (skipped) {
        printf(" %-18s %9s   [skipped: depth+tg > max_seq]\n", test, depth);
        return;
    }
    printf(" %-18s %9s   %10.2f   %7.2f   %9.3f   %8.3f",
           test, depth, s.mean_tps, s.sd_tps, s.ms_per_tok, s.mean_ms * 0.001);
    if (g_roofline) {
        if (g_fixed_bytes > 0.0 && kv_depth >= 0.0) {
            // Median, not mean: the roofline fraction is a property of the
            // steady state, and one stalled rep should not move it.
            double per_tok = g_fixed_bytes + g_kv_bytes_per_pos * kv_depth;
            double ms_tok  = s.med_ms / s.n_toks;
            double gbs     = per_tok / (ms_tok * 1e6);
            printf("   %8.3f   %8.1f", per_tok / 1e9, gbs);
            if (g_peak_gbs > 0.0) printf("   %6.1f%%", 100.0 * gbs / g_peak_gbs);
            else                  printf("   %7s", "—");
        } else {
            printf("   %8s   %8s   %7s", "—", "—", "—");
        }
    }
    printf("\n");
}

// ---------------------------------------------------------------------------
int main(int argc, char* argv[]) {
    static const char* USAGE =
        "Usage: bench --model <model_dir> [options]\n"
        "  -h, --help    show this help text\n"
        "  -p, --p P,... prefill sizes          (default: 128,512)\n"
        "  -n, --n N,... new-token counts       (default: 128)\n"
        "  -d D,...      KV-cache depths        (default: 0,512,1024,2048)\n"
        "  -r, --r R     timed repetitions      (default: 3)\n"
        "  -w, --w W     warmup runs            (default: 1)\n"
        "  --max-seq N   KvCache capacity       (default: auto)\n"
        "  --device N    run with one visible Level Zero GPU\n"
        "  --roofline    report bytes/token and % of measured bandwidth\n"
        "  --bw-mib N    bandwidth probe buffer  (default: 512)\n";

    std::string model_dir;
    std::vector<int> pp_list  = {128, 512};
    std::vector<int> tg_list  = {128};
    std::vector<int> depths   = {0, 512, 1024, 2048};
    std::string device_index;
    int reps    = 3;
    int warmup  = 1;
    int max_seq = -1;  // computed after arg parsing
    bool device_index_set = false;
    // Smaller buffers understate: 64 MiB reads ~20% low against the figure
    // that 1-4 GiB converge on.
    int bw_mib = 1024;

    for (int i = 1; i < argc; ++i) {
        if      (!strcmp(argv[i], "--model")    && i+1<argc) model_dir = argv[++i];
        else if ((!strcmp(argv[i], "-p") || !strcmp(argv[i], "--p")) && i+1<argc)
            pp_list = parse_list(argv[++i]);
        else if ((!strcmp(argv[i], "-n") || !strcmp(argv[i], "--n")) && i+1<argc)
            tg_list = parse_list(argv[++i]);
        else if (!strcmp(argv[i], "-d")         && i+1<argc) depths  = parse_list(argv[++i]);
        else if ((!strcmp(argv[i], "-r") || !strcmp(argv[i], "--r") ||
                  !strcmp(argv[i], "--reps")) && i+1<argc) reps = atoi(argv[++i]);
        else if ((!strcmp(argv[i], "-w") || !strcmp(argv[i], "--w") ||
                  !strcmp(argv[i], "--warmup")) && i+1<argc) warmup = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--max-seq") && i+1<argc) max_seq = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--device")  && i+1<argc) { device_index = argv[++i]; device_index_set = true; }
        else if (!strcmp(argv[i], "--roofline")) g_roofline = true;
        else if (!strcmp(argv[i], "--bw-mib")  && i+1<argc) bw_mib = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")) { fputs(USAGE, stderr); return 0; }
        else if (argv[i][0] != '-' && model_dir.empty()) model_dir = argv[i];
        else { fprintf(stderr, "Unknown argument: %s\n", argv[i]); fputs(USAGE, stderr); return 1; }
    }

    if (model_dir.empty() || pp_list.empty() || tg_list.empty() || depths.empty() ||
        reps <= 0 || warmup < 0) {
        fputs(USAGE, stderr);
        return 1;
    }
    for (int value : pp_list)
        if (value <= 0) { fputs("--p values must be positive\n", stderr); return 1; }
    for (int value : tg_list)
        if (value <= 0) { fputs("--n values must be positive\n", stderr); return 1; }

    try {
        if (device_index_set) gpu_device_control::apply_device_index(device_index);
    } catch (const std::exception& e) {
        fprintf(stderr, "%s\n", e.what());
        return 1;
    }

    // Derive max_seq from the largest depth + n, plus the largest pp size,
    // if the user did not override it with --max-seq.
    if (max_seq < 0) {
        int max_d = *std::max_element(depths.begin(), depths.end());
        int max_p = *std::max_element(pp_list.begin(), pp_list.end());
        int max_n = *std::max_element(tg_list.begin(), tg_list.end());
        max_seq = std::max(max_d + max_n, max_p);
    }

    // ── Load ────────────────────────────────────────────────────────────────
    register_builtin_architectures();

    printf("loading model from %s ...\n", model_dir.c_str());
    double t0 = now_ms();
    std::unique_ptr<Model> model = ModelRegistry::instance().create(model_dir, max_seq);
    double load_s = (now_ms() - t0) * 0.001;

    const ModelInfo& info = model->info();
    printf("model   : %s\n", info.description.c_str());
    printf("backend : SYCL + oneDNN | GPUs: %d | max_seq: %d",
           GpuEngine::count(), max_seq);
    if (const char* active_gpus = gpu_device_control::active_gpus_spec())
        printf(" | ZE_AFFINITY_MASK=%s", active_gpus);
    printf("\n");
    printf("load    : %.1f s\n", load_s);

    if (g_roofline) {
        const DecodeTraffic& traffic = info.decode_traffic;
        if (traffic.empty()) {
            printf("\nroofline: architecture reports no traffic accounting; "
                   "columns disabled\n");
            g_roofline = false;
        } else {
            g_fixed_bytes      = (double)traffic.fixed_bytes();
            g_kv_bytes_per_pos = (double)traffic.bytes_per_kv_position;

            printf("\ntraffic per decode step (analytic, from resident tensors)\n");
            for (const auto& c : traffic.fixed)
                printf("  %-16s %10.3f GB   %5.1f%%\n", c.name.c_str(),
                       c.bytes / 1e9, 100.0 * c.bytes / g_fixed_bytes);
            printf("  %-16s %10.3f GB\n", "fixed total", g_fixed_bytes / 1e9);
            printf("  %-16s %10.1f KiB per cached position\n", "kv",
                   g_kv_bytes_per_pos / 1024.0);

            // Each device is probed separately: a pipeline split runs them one
            // after another, so the slowest single device sets the ceiling for
            // the whole step rather than their sum.
            printf("\nmeasured bandwidth (%d MiB buffer)\n", bw_mib);
            for (int g = 0; g < GpuEngine::count(); ++g) {
                sycl::queue& q = GpuEngine::get(g).queue;
                sycl::device dev = q.get_device();
                Bandwidth bw = measure_bandwidth(q, (size_t)bw_mib << 20, 20);
                printf("  GPU %d   read %7.1f GB/s   copy %7.1f GB/s   %s, %zu GiB\n",
                       g, bw.read_gbs, bw.copy_gbs,
                       dev.get_info<sycl::info::device::name>().c_str(),
                       dev.get_info<sycl::info::device::global_mem_size>() >> 30);
                if (bw.read_gbs > 0.0)
                    g_peak_gbs = (g_peak_gbs == 0.0)
                                     ? bw.read_gbs
                                     : std::min(g_peak_gbs, bw.read_gbs);
            }
            if (GpuEngine::count() > 1)
                printf("  ceiling %7.1f GB/s (slowest device; the layer split "
                       "runs them serially)\n", g_peak_gbs);
        }
    }

    // Placeholder token: BOS from the loaded model config.
    const int bos_id = info.bos_token_id;
    const std::vector<int> single(1, bos_id);

    print_header();

    // ── Prefill (PP) ────────────────────────────────────────────────────────
    for (int pp : pp_list) {
        if (pp > max_seq) {
            char name[32]; snprintf(name, sizeof name, "pp %d", pp);
            printf(" %-18s %9s   [skip: pp=%d > max_seq=%d]\n", name, "—", pp, max_seq);
            continue;
        }
        const std::vector<int> prompt(pp, bos_id);
        std::vector<double> times;

        for (int r = 0; r < warmup + reps; ++r) {
            model->reset_cache();
            double t = now_ms();
            model->forward(ForwardInput{prompt, 0});
            double dt = now_ms() - t;
            if (r >= warmup) times.push_back(dt);
        }

        char name[32]; snprintf(name, sizeof name, "pp %d", pp);
        print_row(name, "—", compute_stats(times, pp));
    }

    // ── Decode (TG) at various KV-cache depths ──────────────────────────────
    for (int tg : tg_list) {
      for (int depth : depths) {
        char name[32]; snprintf(name, sizeof name, "tg %d", tg);
        char dstr[32]; snprintf(dstr, sizeof dstr, "%d", depth);

        if (depth + tg > max_seq) {
            print_row(name, dstr, {}, /*skipped=*/true);
            continue;
        }

        std::vector<double> times;

        for (int r = 0; r < warmup + reps; ++r) {
            model->reset_cache();

            // Pre-fill KV cache to `depth` tokens in chunks (untimed).
            // A single forward of `depth` tokens allocates scores of shape
            // (nq, depth, depth) — O(depth²) and gigantic at depth=16k+.
            // Chunking caps peak activation memory at O(CHUNK × kv_len).
            if (depth > 0) {
                constexpr int CHUNK = 512;
                for (int pos = 0; pos < depth; pos += CHUNK) {
                    int sz = std::min(CHUNK, depth - pos);
                    std::vector<int> chunk(sz, bos_id);
                    model->forward(ForwardInput{chunk, pos});
                }
            }

            // Time `tg` single-token decode steps.
            // The cache grows from depth to depth+tg during this window,
            // which is realistic for mid-generation latency measurement.
            double t = now_ms();
            for (int step = 0; step < tg; ++step)
                model->forward(ForwardInput{single, depth + step});
            double dt = now_ms() - t;

            if (r >= warmup) times.push_back(dt);
        }

        // The cache grows from `depth` to `depth+tg` across the timed window.
        print_row(name, dstr, compute_stats(times, tg), false,
                  depth + (tg - 1) / 2.0);
      }
    }

    printf("\n");
    return 0;
}
