// HDR-Tree over the query set U with the dynamic extension of Section 4.1.
#pragma once

#include <vector>

#include "cluster_tree.h"
#include "join_table.h"

namespace dualtree {

// RkNN search discards a cluster C when its reduced-space lower bound
// exceeds dist_MkNN(C) and verifies the queries of the surviving leaves in
// R^D. The tree follows changes to U (Algorithms 1 and 2) and to the stored
// kNN distances (Algorithm 3); it reads dknn(u, I) from the join table.
//
// With proj = null, LevelDims::fullSpace and Descent::NearestCenter the same
// class gives the Spheretree baseline: one cluster tree over R^D without
// dimensionality reduction.
class HdrTree {
public:
    HdrTree(const Dataset& data, const Projections* proj, const LevelDims& dims, const JoinTable& table, int fanout,
            int leafSize, Tolerance tol, std::uint64_t seed, Descent descent = Descent::SmallestRadiusPlusKnnDist);

    void build(const std::vector<PointId>& queries) { tree_.build(queries); }
    void insert(PointId u) { tree_.insert(u, descent_); }                       // Algorithm 1
    void remove(PointId u) { tree_.remove(u); }                                 // Algorithm 2
    void refresh(const std::vector<PointId>& repaired) { tree_.refresh(repaired); }   // Algorithm 3

    // RkNN search for the reference q. Appends (dist(u, q), u) for every
    // query u whose row the update changes. With pruneDim > 0, a leaf query v
    // is skipped when dist_PC(q, v) on the leading pruneDim components already
    // exceeds dknn(v, I).
    void rknn(PointId q, RknnKind kind, int pruneDim, std::vector<Neighbor>& out,
              PruneProbe* probe = nullptr) const;

    const ClusterTree& tree() const { return tree_; }

private:
    const Dataset& data_;
    const Projections* proj_;
    const JoinTable& table_;
    Descent descent_;
    ClusterTree tree_;
};

}  // namespace dualtree
