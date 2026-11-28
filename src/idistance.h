// iDistance baseline for kNN search over I.
#pragma once

#include <utility>
#include <vector>

#include "common.h"
#include "dataset.h"

namespace dualtree {

// The references are partitioned around m reference points found by
// k-means. A point is keyed by its distance to the reference point of its
// partition, and every partition keeps its keys sorted (the one-dimensional
// index). kNN search grows a query sphere by a fixed step and scans, in every
// partition, the key interval that the sphere covers.
class IDistance {
public:
    IDistance(const Dataset& data, int partitions, double radiusStep, Tolerance tol, std::uint64_t seed);

    void build(const std::vector<PointId>& refs);
    void insert(PointId q);
    void remove(PointId q);
    void knn(const float* x, int k, std::vector<Neighbor>& out) const;

private:
    struct Partition {
        std::vector<float> center;                       // reference point O_i
        std::vector<std::pair<float, PointId>> keys;     // (dist(p, O_i), p), sorted
    };
    std::size_t nearest(const float* x, float* dist) const;
    void add(std::size_t part, float key, PointId p);

    const Dataset& data_;
    int numPartitions_;
    double radiusStep_;
    Tolerance tol_;
    std::uint64_t seed_;
    std::vector<Partition> parts_;
    std::vector<std::uint32_t> partOf_;
    std::vector<float> keyOf_;
    float step_ = 0.0f;
};

}  // namespace dualtree
