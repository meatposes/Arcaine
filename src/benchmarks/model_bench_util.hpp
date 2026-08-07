#pragma once
//
// Shared harness for the llama-bench-style prefill/decode model benchmarks.
// Pure timing + formatting helpers (no model logic); each model-owned
// model_bench.cpp drives its concrete engine's forward()/reset_cache() and
// reports through these helpers so output stays consistent across models.
//
//   reset -> prefill synthetic prompt (PP)        -> time forward(past=0)
//   reset -> populate KV to depth in chunks        -> time single-token decode (TG)
//

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>

#include "benchmarks/util.hpp"   // arcaine::bench::parse_int_csv

namespace arcaine::bench {

struct PpTgStats {
    double mean_ms, sd_ms;     // raw timing
    double mean_tps, sd_tps;   // derived tokens/sec (per-rep)
    double ms_per_tok;         // mean_ms / n_toks
    double med_ms, iqr_ms;     // robust equivalents; one scheduling stall moves
                               // the mean but not the median
    int    n_toks;
};

// Linear-interpolated quantile over an already-sorted copy.
inline double pp_tg_quantile(const std::vector<double>& sorted, double q) {
    if (sorted.empty()) return 0.0;
    if (sorted.size() == 1) return sorted[0];
    double pos = q * (double)(sorted.size() - 1);
    size_t lo = (size_t)pos;
    size_t hi = std::min(lo + 1, sorted.size() - 1);
    return sorted[lo] + (pos - (double)lo) * (sorted[hi] - sorted[lo]);
}

inline PpTgStats compute_pp_tg_stats(const std::vector<double>& ms, int n_toks) {
    int n = (int)ms.size();
    double s = 0, s2 = 0;
    for (double t : ms) { s += t; s2 += t * t; }
    double mean_ms = s / n;
    double sd_ms   = std::sqrt(std::max(0.0, s2 / n - mean_ms * mean_ms));
    double ts = 0, ts2 = 0;
    for (double t : ms) {
        double v = n_toks / (t * 0.001);
        ts += v; ts2 += v * v;
    }
    double mean_tps = ts / n;
    double sd_tps   = std::sqrt(std::max(0.0, ts2 / n - mean_tps * mean_tps));
    std::vector<double> sorted(ms);
    std::sort(sorted.begin(), sorted.end());
    double med = pp_tg_quantile(sorted, 0.5);
    double iqr = pp_tg_quantile(sorted, 0.75) - pp_tg_quantile(sorted, 0.25);
    return {mean_ms, sd_ms, mean_tps, sd_tps, mean_ms / n_toks, med, iqr, n_toks};
}

// Roofline columns are opt-in per row. A model that has not implemented
// per-decode-step byte accounting passes nothing and the columns stay off, so
// enabling this for one model does not disturb the others.
//
// The denominator belongs to the caller because it is measured, not looked up:
// see benchmarks/device_bandwidth.hpp for why a datasheet constant is wrong.
inline void print_pp_tg_header(bool roofline = false) {
    std::printf("\n %-18s %9s   %10s   %7s   %9s   %8s",
                "test", "kv-depth", "t/s", "± sd", "ms/tok", "time(s)");
    if (roofline) std::printf("   %8s   %8s   %7s", "GB/tok", "GB/s", "%roof");
    std::printf("\n");
    std::printf(" %-18s %9s   %10s   %7s   %9s   %8s",
                "──────────────────", "─────────",
                "──────────", "───────", "─────────", "────────");
    if (roofline) std::printf("   %8s   %8s   %7s", "────────", "────────", "───────");
    std::printf("\n");
}

// `bytes_per_token` <= 0 leaves the roofline columns blank for this row, which
// is what prefill wants: its weights are amortized across the whole batch, so a
// per-token byte count would be misleading rather than merely absent.
inline void print_pp_tg_row(const char* test, const char* depth,
                            const PpTgStats& s, bool skipped = false,
                            bool roofline = false,
                            double bytes_per_token = -1.0,
                            double peak_gbps = 0.0) {
    if (skipped) {
        std::printf(" %-18s %9s   [skipped: depth+tg > max_seq]\n", test, depth);
        return;
    }
    std::printf(" %-18s %9s   %10.2f   %7.2f   %9.3f   %8.3f",
                test, depth, s.mean_tps, s.sd_tps, s.ms_per_tok, s.mean_ms * 0.001);
    if (roofline) {
        if (bytes_per_token > 0.0 && s.n_toks > 0) {
            // Median, not mean: the roofline fraction is a property of the
            // steady state and one stalled rep should not move it.
            double ms_tok = s.med_ms / s.n_toks;
            double gbs = ms_tok > 0.0 ? bytes_per_token / (ms_tok * 1e6) : 0.0;
            std::printf("   %8.3f   %8.1f", bytes_per_token / 1e9, gbs);
            if (peak_gbps > 0.0) std::printf("   %6.1f%%", 100.0 * gbs / peak_gbps);
            else                 std::printf("   %7s", "—");
        } else {
            std::printf("   %8s   %8s   %7s", "—", "—", "—");
        }
    }
    std::printf("\n");
}

}  // namespace arcaine::bench
