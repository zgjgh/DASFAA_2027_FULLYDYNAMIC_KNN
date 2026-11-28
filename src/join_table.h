// The materialized kNN join table kNN_join(U, I) (Definition 4).
#pragma once

#include <algorithm>
#include <vector>

#include "common.h"

namespace dualtree {

// What an RkNN search for a reference q looks for.
enum class RknnKind {
    Insertion,   // q is new: queries whose kNN list q enters
    Deletion     // q is removed: queries whose kNN list contains q
};

// One row of k neighbors per query. A row is a max-heap under the
// (distance, id) order, so its first entry is the k-th neighbor and its
// distance is dknn(u, I).
class JoinTable {
public:
    JoinTable(std::size_t numIds, int k)
        : k_(k), cells_(numIds * static_cast<std::size_t>(k)), knnDist_(numIds, kInf) {}

    int k() const { return k_; }

    // Writes the row of u from its k nearest references (in any order).
    void setRow(PointId u, const std::vector<Neighbor>& neighbors) {
        Neighbor* row = rowOf(u);
        const std::size_t m = std::min(neighbors.size(), static_cast<std::size_t>(k_));
        std::copy(neighbors.begin(), neighbors.begin() + static_cast<std::ptrdiff_t>(m), row);
        std::fill(row + m, row + k_, Neighbor{});   // fewer than k references: row not full
        std::make_heap(row, row + k_);
        knnDist_[u] = row[0].dist;
    }

    void clearRow(PointId u) {
        Neighbor* row = rowOf(u);
        std::fill(row, row + k_, Neighbor{});
        knnDist_[u] = kInf;
    }

    // Case 3: the new reference n.id replaces the k-th neighbor of u.
    void replaceKth(PointId u, const Neighbor& n) {
        Neighbor* row = rowOf(u);
        std::pop_heap(row, row + k_);
        row[k_ - 1] = n;
        std::push_heap(row, row + k_);
        knnDist_[u] = row[0].dist;
    }

    float knnDist(PointId u) const { return knnDist_[u]; }
    const Neighbor& kth(PointId u) const { return cells_[static_cast<std::size_t>(u) * k_]; }
    // True if n precedes the k-th neighbor of u, i.e. n belongs to kNN(u).
    bool admits(PointId u, const Neighbor& n) const { return n < kth(u); }

    bool contains(PointId u, PointId ref) const {
        const Neighbor* row = &cells_[static_cast<std::size_t>(u) * k_];
        for (int i = 0; i < k_; ++i)
            if (row[i].id == ref && row[i].dist < kInf) return true;
        return false;
    }

    std::vector<Neighbor> sortedRow(PointId u) const {
        const Neighbor* row = &cells_[static_cast<std::size_t>(u) * k_];
        std::vector<Neighbor> r(row, row + k_);
        std::sort(r.begin(), r.end());
        return r;
    }

    const float* knnDists() const { return knnDist_.data(); }

private:
    Neighbor* rowOf(PointId u) { return &cells_[static_cast<std::size_t>(u) * k_]; }

    int k_;
    std::vector<Neighbor> cells_;   // row u occupies [u k, (u + 1) k)
    std::vector<float> knnDist_;    // dknn(u, I); +inf if u has no row
};

}  // namespace dualtree
