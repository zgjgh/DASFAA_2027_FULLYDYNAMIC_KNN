// Hierarchical structure shared by Delta-Tree, HDR-Tree and Spheretree
// (Section 3.2) and the update procedures of Section 4.1.
#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "common.h"
#include "dataset.h"
#include "kmeans.h"
#include "pca.h"

namespace dualtree {

struct TreeNode;

// A cluster stored at a non-leaf node: center and radius in the space of its
// level, the set points(C) of its points, and the child node that holds
// them. Trees over U also keep dist_MkNN(C), the largest dknn(u, I) over the
// queries in points(C).
struct TreeCluster {
    std::vector<float> center;
    float radius = 0.0f;
    float maxKnnDist = -kInf;
    std::vector<PointId> points;
    TreeNode* owner = nullptr;          // node holding this cluster
    std::unique_ptr<TreeNode> child;    // node generated from points(C)
    int level = 1;                      // level of the owner node
    std::uint64_t mark = 0;             // visit marker of the refresh pass
};

// A tree node. A non-leaf node holds clusters. A leaf holds no clusters;
// the points it stores are points(parent).
struct TreeNode {
    int level = 1;
    TreeCluster* parent = nullptr;      // cluster this node was generated from; null at the root
    std::vector<std::unique_ptr<TreeCluster>> clusters;
    bool isLeaf() const { return clusters.empty(); }
};

// Cluster chosen at each level when a point is inserted.
enum class Descent {
    NearestCenter,              // smallest dist_PC(center(C), p)
    SmallestRadiusPlusKnnDist   // smallest r_C + m_C (Algorithm 1)
};

struct TreeParams {
    int fanout = 10;            // f: at most f clusters per non-leaf node
    int leafSize = 20;          // theta: a leaf holds at most theta points
    bool knnBounds = false;     // maintain dist_MkNN (trees over U)
};

// Counts of the leaf-level check at two dimensionalities d1 < d2, collected
// by the sampled searches that fit the cost model (Section 5). A probed search
// skips the full-dimensional verification of a point pruned at d2.
struct PruneProbe {
    int d1 = 0;
    int d2 = 0;
    std::uint64_t visited = 0;
    std::uint64_t pruned1 = 0;
    std::uint64_t pruned2 = 0;
};

// Points are partitioned top down with k-means; the clusters of level l live
// in the PCA subspace of dimensionality d_l, or in R^D when d_l = D. Each
// point keeps leaf(p), a pointer to its leaf, and each node a pointer to the
// cluster it was generated from. Deletion and refresh therefore start at the
// point and walk a single path bottom up.
class ClusterTree {
public:
    // proj may be null when every level uses R^D. knnDist gives dknn(u, I)
    // per query identifier (trees over U only).
    ClusterTree(const Dataset& data, const Projections* proj, const LevelDims& dims, TreeParams params,
                Tolerance tol, std::uint64_t seed, const float* knnDist = nullptr);
    ~ClusterTree();
    ClusterTree(const ClusterTree&) = delete;
    ClusterTree& operator=(const ClusterTree&) = delete;

    void build(const std::vector<PointId>& ids);
    void insert(PointId p, Descent rule);                  // Algorithm 1 (and plain insertion)
    void remove(PointId p);                                // Algorithm 2
    void refresh(const std::vector<PointId>& repaired);    // Algorithm 3

    const TreeNode& root() const { return *root_; }
    const LevelDims& dims() const { return dims_; }
    const Tolerance& tolerance() const { return tol_; }

    // dist_PC(center(c), p) in the space of c's level.
    float centerDistance(const TreeCluster& c, PointId p) const;
    // Lower bound of dist(p, v) over all v in points(c).
    float lowerBound(const TreeCluster& c, PointId p) const {
        const float d = centerDistance(c, p) - c.radius;
        return d > 0.0f ? d : 0.0f;
    }

    // Calls visit(leafCluster) for every leaf; leafCluster.points are the leaf's points.
    template <class Visit>
    void forEachLeaf(Visit&& visit) const {
        std::vector<const TreeNode*> stack{root_.get()};
        while (!stack.empty()) {
            const TreeNode* node = stack.back();
            stack.pop_back();
            if (node->isLeaf()) {
                visit(*node->parent);
                continue;
            }
            for (const auto& c : node->clusters) stack.push_back(c->child.get());
        }
    }

private:
    struct PointState {
        TreeNode* leaf = nullptr;            // leaf(p)
        std::vector<std::uint32_t> slot;     // slot[l - 1]: position of p in points(C) at level l
    };

    int dimOf(int level) const;
    PointVectors spaceOf(int level) const;
    std::unique_ptr<TreeNode> partition(int level, TreeCluster* parent, const std::vector<PointId>& ids,
                                        bool requireSplit);
    std::unique_ptr<TreeNode> makeLeaf(TreeCluster* parent);
    void attach(TreeCluster& c, PointId p);
    void detach(TreeCluster& c, PointId p);
    bool growSubtree(TreeCluster& c);
    void collapse(TreeCluster& c);
    void recomputeRadius(TreeCluster& c);
    void recomputeMaxKnnDist(TreeCluster& c);
    void initMaxKnnDist(TreeNode& node);

    const Dataset& data_;
    const Projections* proj_;
    LevelDims dims_;
    TreeParams params_;
    Tolerance tol_;
    std::uint64_t seed_;
    std::uint64_t partitions_ = 0;
    const float* knnDist_;
    std::unique_ptr<TreeNode> root_;
    std::vector<PointState> state_;
    std::uint64_t epoch_ = 0;
    std::vector<std::vector<TreeCluster*>> pending_;   // clusters to check per level (Algorithm 3)
};

}  // namespace dualtree
