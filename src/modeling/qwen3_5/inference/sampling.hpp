#pragma once

#include <random>

namespace arcaine::qwen3_5 {

// Qwen3.5-owned top-k / top-p / greedy sampler. A copy of the project sampler
// kept inside this module so the model owns its sampling policy.
// The distribution sample_token draws from, after temperature / top-k / top-p
// warping, as a sparse list. Speculative decoding needs the distribution
// itself and not just a draw from it: accepting a draft requires the target
// and draft probabilities of the same token, and rejecting it requires
// sampling from the residual between them.
//
// `ids` and `probs` are in nucleus order (probability descending) — the order
// sample_token has always drawn in — so routing sample_token through this
// leaves its output for a given seed unchanged.
struct SamplingDistribution {
    std::vector<int>   ids;
    std::vector<float> probs;   // normalized over the nucleus

    // Zero outside the support. The nucleus is at most top_k entries, so a
    // scan beats building an index.
    float probability_of(int id) const {
        for (size_t i = 0; i < ids.size(); ++i)
            if (ids[i] == id) return probs[i];
        return 0.0f;
    }
    int sample(std::mt19937& rng) const;
};

// Applies the same warping sample_token applies. At temperature <= 0 this is a
// point mass on the argmax, which is what makes greedy a special case of
// speculative sampling rather than a separate path.
SamplingDistribution warp_logits(const float* logits, int vocab_size,
                                 float temperature, int top_k, float top_p);

// Standard speculative sampling correction (Leviathan et al. / Chen et al.).
// `draft` was drawn from `q`; `p` is the target. Accepts with probability
// min(1, p(draft)/q(draft)) and otherwise draws from the normalized residual
// max(0, p - q). The token returned is distributed exactly as `p`, which is
// what makes speculation lossless rather than merely close.
//
// Accepting whenever the draft matches the target's argmax — the obvious rule —
// is only correct at temperature 0. Under sampling it over-weights whatever the
// draft head prefers and silently serves a different distribution than the
// request asked for.
//
// Split out from the decode path so it can be tested against a known p without
// a GPU or a model.
int speculative_correct(const SamplingDistribution& q,
                        const SamplingDistribution& p, int draft,
                        std::mt19937& rng, bool* accepted);

int sample_token(const float* logits, int vocab_size,
                 float temperature, int top_k, float top_p,
                 std::mt19937& rng);

}  // namespace arcaine::qwen3_5
