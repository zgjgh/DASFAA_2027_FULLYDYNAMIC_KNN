// Delta-Tree over the reference set I.
#pragma once

#include <vector>

#include "cluster_tree.h"

namespace dualtree {

// kNN search visits the clusters best first by their reduced-space lower
// bounds, discards a cluster once its bound exceeds the current k-th best
// distance, and verifies the points of the surviving leaves in R^D.
class DeltaTree {
public:
    DeltaTree(const Dataset& data, const Projections& proj, const LevelDims& dims, int fanout, int leafSize,
              Tolerance tol, std::uint64_t seed);

    void build(const std::vector<PointId>& refs) { tree_.build(refs); }
    void insert(PointId q) { tree_.insert(q, Descent::NearestCenter); }
    void remove(PointId q) { tree_.remove(q); }

    // Exact kNN(u, I) written to `out` (k entries, in heap order). With
    // pruneDim > 0, a leaf point v is skipped when dist_PC(u, v) on the
    // leading pruneDim components already exceeds the k-th best distance.
    void knn(PointId u, int k, int pruneDim, std::vector<Neighbor>& out, PruneProbe* probe = nullptr) const;

    const ClusterTree& tree() const { return tree_; }

private:
    const Dataset& data_;
    const Projections& proj_;
    ClusterTree tree_;
};

}  // namespace dualtree
