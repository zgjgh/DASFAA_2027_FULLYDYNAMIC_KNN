#include "kmeans.h"

#include <algorithm>
#include <cstring>

namespace dualtree {

namespace {
constexpr int kMaxIterations = 100;
constexpr std::uint32_t kUnassigned = 0xFFFFFFFFu;
}  // namespace

KMeansResult kmeans(const std::vector<PointId>& ids, PointVectors vectors, int dim, int k, std::uint64_t seed) {
    KMeansResult result;
    const std::size_t n = ids.size();
    if (n == 0 || k <= 0) return result;
    const std::size_t clusters = std::min<std::size_t>(static_cast<std::size_t>(k), n);
    const std::size_t d = static_cast<std::size_t>(dim);

    std::vector<float> x(n * d);
    for (std::size_t i = 0; i < n; ++i) std::memcpy(&x[i * d], vectors(ids[i]), d * sizeof(float));

    // Initial centers: distinct random points (partial Fisher-Yates shuffle).
    std::vector<std::size_t> order(n);
    for (std::size_t i = 0; i < n; ++i) order[i] = i;
    Random rng(seed);
    for (std::size_t i = 0; i < clusters; ++i) std::swap(order[i], order[i + rng.below(n - i)]);
    std::vector<float> centers(clusters * d);
    for (std::size_t c = 0; c < clusters; ++c) std::memcpy(&centers[c * d], &x[order[c] * d], d * sizeof(float));

    std::vector<std::uint32_t> assign(n, kUnassigned);
    std::vector<double> sums(clusters * d);
    std::vector<std::size_t> counts(clusters);
    const bool parallel = n * clusters * d > (std::size_t(1) << 22);
    for (int iter = 0; iter < kMaxIterations; ++iter) {
        std::size_t changed = 0;
#pragma omp parallel for schedule(static) reduction(+ : changed) if (parallel)
        for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(n); ++i) {
            const float* xi = &x[static_cast<std::size_t>(i) * d];
            std::uint32_t best = 0;
            float bestDist = kInf;
            for (std::size_t c = 0; c < clusters; ++c) {
                const float dist = squaredDistance(xi, &centers[c * d], d);
                if (dist < bestDist) {
                    bestDist = dist;
                    best = static_cast<std::uint32_t>(c);
                }
            }
            if (assign[i] != best) {
                assign[i] = best;
                ++changed;
            }
        }
        if (changed == 0) break;
        std::fill(sums.begin(), sums.end(), 0.0);
        std::fill(counts.begin(), counts.end(), 0);
        for (std::size_t i = 0; i < n; ++i) {
            const std::size_t c = assign[i];
            ++counts[c];
            for (std::size_t j = 0; j < d; ++j) sums[c * d + j] += x[i * d + j];
        }
        for (std::size_t c = 0; c < clusters; ++c) {
            if (counts[c] == 0) continue;   // an empty cluster keeps its center
            for (std::size_t j = 0; j < d; ++j)
                centers[c * d + j] = static_cast<float>(sums[c * d + j] / static_cast<double>(counts[c]));
        }
    }

    std::vector<std::uint32_t> slot(clusters, kUnassigned);
    for (std::size_t i = 0; i < n; ++i) {
        const std::size_t c = assign[i];
        if (slot[c] == kUnassigned) {
            slot[c] = static_cast<std::uint32_t>(result.members.size());
            result.centers.emplace_back(centers.begin() + c * d, centers.begin() + (c + 1) * d);
            result.members.emplace_back();
        }
        result.members[slot[c]].push_back(ids[i]);
    }
    return result;
}

}  // namespace dualtree
