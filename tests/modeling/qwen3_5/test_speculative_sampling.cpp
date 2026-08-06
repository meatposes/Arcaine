// Speculative sampling preserves the target distribution.
//
// The whole point of the accept/reject rule is that the emitted token is
// distributed exactly as the target p, not merely close to it. The obvious
// alternative — accept when the draft equals the target's argmax — is only
// correct at temperature 0; under sampling it over-weights whatever the draft
// head prefers, and the result is a served distribution that quietly differs
// from the one the request asked for. That failure is invisible in any
// single-sequence check, so it needs a distributional one.
//
// Draws many (draft ~ q, correct against p) pairs and compares the empirical
// histogram against p. Pure C++: no GPU, no model, runs under ctest.

#include <cmath>
#include <cstdio>
#include <map>
#include <random>
#include <vector>

#include "check.hpp"
#include "modeling/qwen3_5/inference/sampling.hpp"

using arcaine::qwen3_5::SamplingDistribution;
using arcaine::qwen3_5::speculative_correct;

namespace {

SamplingDistribution make(std::vector<int> ids, std::vector<float> probs) {
    SamplingDistribution d;
    d.ids = std::move(ids);
    d.probs = std::move(probs);
    float sum = 0.0f;
    for (float p : d.probs) sum += p;
    for (float& p : d.probs) p /= sum;
    return d;
}

// Total variation distance between the empirical distribution of the corrected
// token and the target. Sampling error at n draws is O(1/sqrt(n)) per outcome,
// so the tolerance is set well above that and far below the bias the
// argmax-matching rule would introduce.
double total_variation(const SamplingDistribution& q,
                       const SamplingDistribution& p, int draws,
                       uint32_t seed, double* accept_rate) {
    std::mt19937 rng(seed);
    std::map<int, int> counts;
    int accepts = 0;
    for (int i = 0; i < draws; ++i) {
        int draft = q.sample(rng);
        bool accepted = false;
        int token = speculative_correct(q, p, draft, rng, &accepted);
        if (accepted) ++accepts;
        ++counts[token];
    }
    double tv = 0.0;
    std::map<int, double> target;
    for (size_t i = 0; i < p.ids.size(); ++i) target[p.ids[i]] = p.probs[i];
    for (const auto& kv : counts)
        target[kv.first] += 0.0;   // ensure every observed id is compared
    for (const auto& kv : target) {
        double observed = (double)counts[kv.first] / draws;
        tv += std::fabs(observed - kv.second);
    }
    if (accept_rate) *accept_rate = (double)accepts / draws;
    return 0.5 * tv;
}

void check_case(const char* label, const SamplingDistribution& q,
                const SamplingDistribution& p, double tolerance) {
    constexpr int kDraws = 200000;
    double accept_rate = 0.0;
    double tv = total_variation(q, p, kDraws, 0xC0FFEEu, &accept_rate);
    std::printf("  %-34s TV %.5f (tol %.3f), accept %.3f\n",
                label, tv, tolerance, accept_rate);
    CHECK(tv < tolerance);
}

}  // namespace

int main() {
    std::printf("qwen3_5 speculative sampling\n");

    // Draft agrees with the target. Everything is accepted and the output is
    // trivially p.
    {
        SamplingDistribution p = make({1, 2, 3}, {0.5f, 0.3f, 0.2f});
        check_case("identical draft and target", p, p, 0.01);
    }

    // Draft is skewed away from the target. This is the case the argmax rule
    // gets wrong: it would emit the draft's favourite far too often.
    {
        SamplingDistribution q = make({1, 2, 3}, {0.8f, 0.1f, 0.1f});
        SamplingDistribution p = make({1, 2, 3}, {0.2f, 0.3f, 0.5f});
        check_case("skewed draft", q, p, 0.01);
    }

    // Supports only partly overlap: the draft proposes tokens the target has
    // ruled out, and the target has mass the draft never proposes. The residual
    // has to cover the difference.
    {
        SamplingDistribution q = make({1, 2, 9}, {0.4f, 0.3f, 0.3f});
        SamplingDistribution p = make({2, 3, 4}, {0.5f, 0.25f, 0.25f});
        check_case("partially disjoint supports", q, p, 0.01);
    }

    // Draft is a point mass, the target is broad. Every miss must be corrected
    // through the residual.
    {
        SamplingDistribution q = make({7}, {1.0f});
        SamplingDistribution p = make({5, 6, 7, 8}, {0.25f, 0.25f, 0.25f, 0.25f});
        check_case("point-mass draft, broad target", q, p, 0.01);
    }

    // Temperature 0 on both sides: two point masses that disagree. The rule
    // must reject and emit the target's token every time, which is exactly the
    // greedy behaviour the end-to-end test relies on.
    {
        SamplingDistribution q = make({3}, {1.0f});
        SamplingDistribution p = make({4}, {1.0f});
        std::mt19937 rng(1234);
        for (int i = 0; i < 100; ++i) {
            bool accepted = true;
            int token = speculative_correct(q, p, 3, rng, &accepted);
            CHECK(!accepted);
            CHECK_EQ(token, 4);
        }
        std::printf("  %-34s ok\n", "greedy disagreement");
    }

    // Temperature 0 agreeing: accept without consuming randomness, so a greedy
    // speculative decode leaves the rng stream untouched.
    {
        SamplingDistribution q = make({4}, {1.0f});
        SamplingDistribution p = make({4}, {1.0f});
        std::mt19937 rng(1234), untouched(1234);
        bool accepted = false;
        CHECK_EQ(speculative_correct(q, p, 4, rng, &accepted), 4);
        CHECK(accepted);
        CHECK(rng == untouched);
        std::printf("  %-34s ok\n", "greedy agreement, rng untouched");
    }

    RETURN_TESTS();
}
