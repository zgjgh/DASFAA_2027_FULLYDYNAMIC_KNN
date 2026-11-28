// k-means partitioning used to build the trees and the iDistance partitions.
#pragma once

#include <cstdint>
#include <vector>

#include "common.h"

namespace dualtree {

// Read-only view of per-point vectors: point p starts at base + p * stride.
struct PointVectors {
    const float* base = nullptr;
    std::size_t stride = 0;
    const float* operator()(PointId p) const { return base + static_cast<std::size_t>(p) * stride; }
};

struct KMeansResult {
    std::vector<std::vector<float>> centers;      // non-empty clusters only
    std::vector<std::vector<PointId>> members;    // members[c] belong to centers[c]
};

// Lloyd's algorithm on the first `dim` coordinates of the given points. The
// initial centers are min(k, |ids|) distinct points chosen at random.
KMeansResult kmeans(const std::vector<PointId>& ids, PointVectors vectors, int dim, int k, std::uint64_t seed);

}  // namespace dualtree
