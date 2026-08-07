#pragma once

// Requantize an NVFP4 weight tensor to E4M3 FP8, in place of the checkpoint's
// packed form, at load time.
//
// Why: at M=1 the f4 path is bandwidth-inefficient enough that the wider format
// wins outright. Measured per layer group on Qwen3.6-27B, the NVFP4 MLP layers
// run at ~36% of achievable bandwidth while the checkpoint's own FP8 layers
// reach ~88%, so FP8 moves 44% more bytes and is still 1.69x faster per layer.
// oneDNN's f4 GEMM and a hand-written ESIMD f4 GEMV converge at the same ~250
// GB/s, so this is a property of the format at M=1 and not of a kernel.
// See notes/qwen3_5_27b/decode_bandwidth_gap.md.
//
// What changes numerically, in both directions:
//
//   the scale grid coarsens. NVFP4 carries one E4M3 scale per 16 inputs; FP8
//   carries one BF16 scale per output channel. That is the loss.
//
//   the mantissa widens. E2M1 has one mantissa bit, E4M3 has three.
//
//   activations stop being quantized. The f4 path is W4A4 and packs
//   activations to fp4 (pack_bf16_to_nvfp4); matmul_fp8 consumes BF16
//   directly. That is a large gain and it applies to every layer converted.
//
// The net is not predictable from the formats alone and must be measured —
// `arcaine_mbench --golden` exists for exactly this.
//
// The conversion is a straight dequantize/requantize, so it costs load time and
// VRAM (FP8 is 2x the packed bytes) and nothing at inference.

#include <cstdint>

#include <sycl/sycl.hpp>

#include "runtime/gpu/buffer.hpp"
#include "runtime/quantization/fp8.hpp"
#include "runtime/quantization/nvfp4.hpp"

// E4M3 has no sign handling in the positive-only encoder used for scales.
// Weights are signed, so the magnitude is encoded and the sign reapplied. A
// zero magnitude keeps a clear sign bit rather than producing -0.
inline uint8_t nvfp4_encode_e4m3_signed(float x) {
    uint8_t magnitude = nvfp4_encode_e4m3_positive(sycl::fabs(x));
    if (magnitude == 0) return 0;
    return (x < 0.0f) ? (uint8_t)(magnitude | 0x80) : magnitude;
}

// Largest finite E4M3 magnitude. Row scales target this so the row uses the
// full range.
inline constexpr float kE4m3Max = 448.0f;

// Row scale = clip * absmax / 448. A clip below 1 saturates the largest
// weights in the row and spends the E4M3 range on the bulk of the
// distribution instead. Whether that trade pays depends on how outlier-heavy
// the weights are, which is a per-checkpoint question, so it is measured
// rather than assumed.
inline Fp8Linear requantize_nvfp4_to_fp8(const Nvfp4Linear& source,
                                         sycl::queue& queue, float clip = 1.0f) {
    const int N = source.out_features;
    const int K = source.in_features;

    Fp8Linear out;
    out.in_features = K;
    out.out_features = N;
    out.weight = GpuBuffer<uint8_t>((size_t)N * K, queue);
    out.weight_scale = GpuBuffer<bf16>((size_t)N, queue);

    const uint8_t* packed = source.weight_packed.data();
    const uint8_t* scales = source.weight_scale.data();
    const float inverse_global = 1.0f / source.weight_global_scale;
    uint8_t* dst = out.weight.data();
    bf16* dst_scale = out.weight_scale.data();

    // One work-item per output row. Two passes over the row: the first finds
    // the magnitude the row scale has to cover, the second encodes against it.
    // Reading each row twice is a few tens of milliseconds over the whole model
    // and happens once at load.
    queue.submit([&](sycl::handler& handler) {
        handler.parallel_for(sycl::range<1>((size_t)N), [=](sycl::id<1> id) {
            const int n = (int)id[0];
            const uint8_t* row = packed + (size_t)n * (K / 2);

            float max_abs = 0.0f;
            for (int k = 0; k < K; k += 2) {
                uint8_t byte = row[k / 2];
                // Scales are stored k-major, [K/16, N], the layout oneDNN
                // wants; both nibbles of a byte share a 16-wide block.
                float scale = nvfp4_e4m3_to_float(
                    scales[(size_t)(k / 16) * N + n]);
                float lo = nvfp4_e2m1_to_float(byte & 0x0f) * scale;
                float hi = nvfp4_e2m1_to_float((byte >> 4) & 0x0f) * scale;
                max_abs = sycl::fmax(max_abs, sycl::fabs(lo));
                max_abs = sycl::fmax(max_abs, sycl::fabs(hi));
            }
            max_abs *= inverse_global;

            float row_scale = max_abs > 0.0f ? clip * max_abs / kE4m3Max : 1.0f;
            // Round-trip through BF16 before using it. The scale is stored as
            // BF16 and the matmul will dequantize with the rounded value, so
            // encoding against the unrounded one would bias every weight in
            // the row by the rounding error.
            bf16 stored_scale = float_to_bf16(row_scale);
            dst_scale[n] = stored_scale;
            float inverse_row_scale = 1.0f / bf16_to_float(stored_scale);

            uint8_t* out_row = dst + (size_t)n * K;
            for (int k = 0; k < K; k += 2) {
                uint8_t byte = row[k / 2];
                float scale = nvfp4_e4m3_to_float(
                    scales[(size_t)(k / 16) * N + n]) * inverse_global;
                // Clamp before encoding: with clip < 1 the largest weights
                // exceed the representable range and must saturate rather
                // than wrap through the encoder.
                float lo = nvfp4_e2m1_to_float(byte & 0x0f) * scale * inverse_row_scale;
                float hi = nvfp4_e2m1_to_float((byte >> 4) & 0x0f) * scale *
                           inverse_row_scale;
                lo = sycl::fmin(sycl::fmax(lo, -kE4m3Max), kE4m3Max);
                hi = sycl::fmin(sycl::fmax(hi, -kE4m3Max), kE4m3Max);
                out_row[k] = nvfp4_encode_e4m3_signed(lo);
                out_row[k + 1] = nvfp4_encode_e4m3_signed(hi);
            }
        });
    }).wait();

    return out;
}
