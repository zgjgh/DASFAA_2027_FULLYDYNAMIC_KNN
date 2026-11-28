// Linear scans, used by the baselines for the search direction they do not index.
#pragma once

#include <vector>

#include "common.h"
#include "dataset.h"
#include "join_table.h"

namespace dualtree {

// The current members of U or I.
class LiveSet {
public:
    explicit LiveSet(std::size_t numIds) : pos_(numIds, kAbsent) {}

    void insert(PointId p) {
        pos_[p] = static_cast<std::uint32_t>(items_.size());
        items_.push_back(p);
    }
    void erase(PointId p) {
        const std::uint32_t i = pos_[p];
        items_[i] = items_.back();
        pos_[items_[i]] = i;
        items_.pop_back();
        pos_[p] = kAbsent;
    }
    void insert(const std::vector<PointId>& ps) {
        for (PointId p : ps) insert(p);
    }
    const std::vector<PointId>& items() const { return items_; }

private:
    static constexpr std::uint32_t kAbsent = 0xFFFFFFFFu;
    std::vector<PointId> items_;
    std::vector<std::uint32_t> pos_;
};

// Exact kNN of x over `refs` (k entries, in heap order).
void scanKnn(const Dataset& data, const float* x, const std::vector<PointId>& refs, int k,
             std::vector<Neighbor>& out);

// RkNN search for the reference q by a scan over all queries: one
// full-dimensional distance per query, compared with its dknn.
void scanRknn(const Dataset& data, PointId q, RknnKind kind, const JoinTable& table,
              const std::vector<PointId>& queries, const Tolerance& tol, std::vector<Neighbor>& out);

}  // namespace dualtree
