#include "sampling.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>
#include <vector>

namespace arcaine::qwen3_5 {

int SamplingDistribution::sample(std::mt19937& rng) const {
    if (ids.empty()) return 0;
    if (ids.size() == 1) return ids[0];
    std::discrete_distribution<int> dist(probs.begin(), probs.end());
    return ids[dist(rng)];
}

SamplingDistribution warp_logits(const float* logits, int vocab_size,
                                 float temperature, int top_k, float top_p) {
    SamplingDistribution out;
    if (temperature <= 0.0f) {
        int best = static_cast<int>(
            std::max_element(logits, logits + vocab_size) - logits);
        out.ids   = {best};
        out.probs = {1.0f};
        return out;
    }

    std::vector<float> scaled(vocab_size);
    for (int i = 0; i < vocab_size; ++i)
        scaled[i] = logits[i] / temperature;

    int k = std::min(top_k, vocab_size);
    if (k <= 0) k = vocab_size;
    std::vector<int> indices(vocab_size);
    std::iota(indices.begin(), indices.end(), 0);
    std::partial_sort(indices.begin(), indices.begin() + k, indices.end(),
        [&](int a, int b) { return scaled[a] > scaled[b]; });
    indices.resize(k);

    float max_v = scaled[indices[0]];
    std::vector<float> probs(k);
    float sum = 0.0f;
    for (int i = 0; i < k; ++i) {
        probs[i] = std::exp(scaled[indices[i]] - max_v);
        sum += probs[i];
    }
    for (float& p : probs) p /= sum;

    std::vector<int> sorted_k(k);
    std::iota(sorted_k.begin(), sorted_k.end(), 0);
    std::sort(sorted_k.begin(), sorted_k.end(),
        [&](int a, int b) { return probs[a] > probs[b]; });

    float cumsum = 0.0f;
    int nucleus_size = 0;
    for (int i = 0; i < k; ++i) {
        cumsum += probs[sorted_k[i]];
        ++nucleus_size;
        if (cumsum >= top_p) break;
    }

    out.probs.resize(nucleus_size);
    out.ids.resize(nucleus_size);
    float final_sum = 0.0f;
    for (int i = 0; i < nucleus_size; ++i) {
        int idx = sorted_k[i];
        out.probs[i] = probs[idx];
        out.ids[i]   = indices[idx];
        final_sum += probs[idx];
    }
    for (float& p : out.probs) p /= final_sum;
    return out;
}

int speculative_correct(const SamplingDistribution& q,
                        const SamplingDistribution& p, int draft,
                        std::mt19937& rng, bool* accepted) {
    if (accepted) *accepted = false;
    float p_draft = p.probability_of(draft);
    float q_draft = q.probability_of(draft);

    // A certain accept and a certain reject are decided without drawing, so the
    // rng stream is untouched at temperature 0 where both are point masses.
    if (q_draft > 0.0f) {
        float ratio = p_draft / q_draft;
        bool take = ratio >= 1.0f;
        if (!take && ratio > 0.0f)
            take = std::uniform_real_distribution<float>(0.0f, 1.0f)(rng) < ratio;
        if (take) {
            if (accepted) *accepted = true;
            return draft;
        }
    }

    SamplingDistribution residual;
    float total = 0.0f;
    for (size_t i = 0; i < p.ids.size(); ++i) {
        float excess = p.probs[i] - q.probability_of(p.ids[i]);
        if (excess <= 0.0f) continue;
        residual.ids.push_back(p.ids[i]);
        residual.probs.push_back(excess);
        total += excess;
    }
    // p dominated by q everywhere it has mass. The acceptance test makes this
    // vanishingly unlikely; fall back to the target so the round still makes
    // progress.
    if (residual.ids.empty() || total <= 0.0f) return p.sample(rng);
    for (float& v : residual.probs) v /= total;
    return residual.sample(rng);
}

int sample_token(const float* logits, int vocab_size,
                 float temperature, int top_k, float top_p,
                 std::mt19937& rng) {
    if (temperature <= 0.0f) {
        return static_cast<int>(
            std::max_element(logits, logits + vocab_size) - logits);
    }
    return warp_logits(logits, vocab_size, temperature, top_k, top_p).sample(rng);
}

}  // namespace arcaine::qwen3_5
