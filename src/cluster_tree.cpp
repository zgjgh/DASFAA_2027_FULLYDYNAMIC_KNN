#include "cluster_tree.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace dualtree {

ClusterTree::ClusterTree(const Dataset& data, const Projections* proj, const LevelDims& dims, TreeParams params,
                         Tolerance tol, std::uint64_t seed, const float* knnDist)
    : data_(data), proj_(proj), dims_(dims), params_(params), tol_(tol), seed_(seed), knnDist_(knnDist) {
    if (params_.fanout < 2 || params_.leafSize < 1) throw std::invalid_argument("fanout >= 2 and leaf size >= 1 required");
    if (params_.knnBounds && !knnDist_) throw std::invalid_argument("a tree over U needs the kNN distances");
}

ClusterTree::~ClusterTree() = default;

int ClusterTree::dimOf(int level) const {
    const int d = dims_.at(level);
    return (proj_ && d < data_.dim()) ? d : data_.dim();
}

PointVectors ClusterTree::spaceOf(int level) const {
    if (proj_ && dims_.at(level) < data_.dim()) return {(*proj_)(0), static_cast<std::size_t>(proj_->length())};
    return {data_[0], static_cast<std::size_t>(data_.dim())};
}

float ClusterTree::centerDistance(const TreeCluster& c, PointId p) const {
    return std::sqrt(squaredDistance(spaceOf(c.level)(p), c.center.data(), static_cast<std::size_t>(dimOf(c.level))));
}

void ClusterTree::build(const std::vector<PointId>& ids) {
    if (ids.empty()) throw std::invalid_argument("cannot build a tree over an empty set");
    state_.assign(data_.size(), PointState{});
    root_ = partition(1, nullptr, ids, false);
    if (params_.knnBounds) initMaxKnnDist(*root_);
}

// Splits ids with k-means into the clusters of a new node at `level` and
// builds the subtree of every cluster holding more than theta points. With
// requireSplit, returns null if k-means cannot separate the points (e.g.
// duplicates), so that the caller keeps them in one leaf.
std::unique_ptr<TreeNode> ClusterTree::partition(int level, TreeCluster* parent, const std::vector<PointId>& ids,
                                                 bool requireSplit) {
    KMeansResult km = kmeans(ids, spaceOf(level), dimOf(level), params_.fanout, seed_ + 7919 * ++partitions_);
    if (requireSplit && km.members.size() < 2) return nullptr;
    auto node = std::make_unique<TreeNode>();
    node->level = level;
    node->parent = parent;
    node->clusters.reserve(km.members.size());
    for (std::size_t g = 0; g < km.members.size(); ++g) {
        auto c = std::make_unique<TreeCluster>();
        c->center = std::move(km.centers[g]);
        c->owner = node.get();
        c->level = level;
        c->points.reserve(km.members[g].size());
        for (PointId p : km.members[g]) attach(*c, p);
        recomputeRadius(*c);
        if (c->points.size() > static_cast<std::size_t>(params_.leafSize))
            c->child = partition(level + 1, c.get(), c->points, true);
        if (!c->child) c->child = makeLeaf(c.get());
        node->clusters.push_back(std::move(c));
    }
    return node;
}

std::unique_ptr<TreeNode> ClusterTree::makeLeaf(TreeCluster* parent) {
    auto leaf = std::make_unique<TreeNode>();
    leaf->level = parent->level + 1;
    leaf->parent = parent;
    for (PointId p : parent->points) state_[p].leaf = leaf.get();
    return leaf;
}

void ClusterTree::attach(TreeCluster& c, PointId p) {
    std::vector<std::uint32_t>& slot = state_[p].slot;
    if (slot.size() < static_cast<std::size_t>(c.level)) slot.resize(c.level);
    slot[c.level - 1] = static_cast<std::uint32_t>(c.points.size());
    c.points.push_back(p);
}

void ClusterTree::detach(TreeCluster& c, PointId p) {
    const std::uint32_t i = state_[p].slot[c.level - 1];
    const PointId last = c.points.back();
    c.points[i] = last;
    state_[last].slot[c.level - 1] = i;
    c.points.pop_back();
}

void ClusterTree::recomputeRadius(TreeCluster& c) {
    float r = 0.0f;
    for (PointId p : c.points) r = std::max(r, centerDistance(c, p));
    c.radius = r;
}

// dist_MkNN(C): from the queries of the leaf below C, or from the clusters
// directly below C.
void ClusterTree::recomputeMaxKnnDist(TreeCluster& c) {
    float m = -kInf;
    if (c.child->isLeaf()) {
        for (PointId u : c.points) m = std::max(m, knnDist_[u]);
    } else {
        for (const auto& below : c.child->clusters) m = std::max(m, below->maxKnnDist);
    }
    c.maxKnnDist = m;
}

void ClusterTree::initMaxKnnDist(TreeNode& node) {
    for (auto& c : node.clusters) {
        if (!c->child->isLeaf()) initMaxKnnDist(*c->child);
        recomputeMaxKnnDist(*c);
    }
}

// Algorithm 1, lines 12-13: k-means grows a subtree below c from points(c),
// which already include the new point. Levels deeper than the initial height
// use R^D. Returns false if the points cannot be separated.
bool ClusterTree::growSubtree(TreeCluster& c) {
    std::unique_ptr<TreeNode> subtree = partition(c.level + 1, &c, c.points, true);
    if (!subtree) return false;
    c.child = std::move(subtree);
    if (params_.knnBounds) initMaxKnnDist(*c.child);
    return true;
}

// Algorithm 2, lines 5-7: the subtree below c collapses into one leaf, and the
// leaf pointers of points(c) move to it.
void ClusterTree::collapse(TreeCluster& c) {
    c.child = makeLeaf(&c);
    for (PointId p : c.points) state_[p].slot.resize(c.level);
}

// Algorithm 1 for Descent::SmallestRadiusPlusKnnDist. The centers of existing
// clusters are kept; the point descends along a single path, and every
// cluster on it takes the point and updates its radius (and dist_MkNN).
void ClusterTree::insert(PointId p, Descent rule) {
    const float du = params_.knnBounds ? knnDist_[p] : 0.0f;   // d_u = dknn(u, I)
    TreeNode* node = root_.get();
    while (!node->isLeaf()) {
        TreeCluster* best = nullptr;
        float bestScore = kInf;
        float bestRadius = 0.0f;
        float bestKnn = -kInf;
        for (auto& c : node->clusters) {
            const float delta = centerDistance(*c, p);
            const float r = std::max(c->radius, delta);                               // r_C
            const float m = params_.knnBounds ? std::max(c->maxKnnDist, du) : -kInf;  // m_C
            const float score = rule == Descent::NearestCenter ? delta : r + m;
            if (!best || score < bestScore) {
                best = c.get();
                bestScore = score;
                bestRadius = r;
                bestKnn = m;
            }
        }
        attach(*best, p);
        best->radius = bestRadius;
        if (params_.knnBounds) best->maxKnnDist = bestKnn;
        // The leaf below held theta points before p arrived: grow a subtree and stop.
        if (best->child->isLeaf() && best->points.size() > static_cast<std::size_t>(params_.leafSize) &&
            growSubtree(*best))
            return;
        node = best->child.get();
    }
    state_[p].leaf = node;
}

// Algorithm 2. The leaf pointer leads to the leaf of p; the walk then follows
// the parent pointers. The points of a leaf are points(C) of its parent
// cluster, so the first iteration deletes p from its leaf.
void ClusterTree::remove(PointId p) {
    TreeCluster* c = state_[p].leaf->parent;
    while (c) {
        detach(*c, p);
        if (!c->child->isLeaf() && c->points.size() < static_cast<std::size_t>(params_.leafSize)) collapse(*c);
        if (centerDistance(*c, p) >= c->radius) recomputeRadius(*c);   // p lay on the boundary
        if (params_.knnBounds) recomputeMaxKnnDist(*c);
        c = c->owner->parent;
    }
    state_[p].leaf = nullptr;
    state_[p].slot.clear();
}

// Algorithm 3. Starts from the parent clusters of the leaves holding the
// repaired queries and climbs level by level. A cluster is checked at most
// once; its parent is checked only if its dist_MkNN changed.
void ClusterTree::refresh(const std::vector<PointId>& repaired) {
    ++epoch_;
    int deepest = 0;
    for (PointId u : repaired) {
        TreeCluster* c = state_[u].leaf->parent;
        if (c->mark == epoch_) continue;
        c->mark = epoch_;
        if (pending_.size() <= static_cast<std::size_t>(c->level)) pending_.resize(c->level + 1);
        pending_[c->level].push_back(c);
        deepest = std::max(deepest, c->level);
    }
    for (int level = deepest; level >= 1; --level) {
        for (TreeCluster* c : pending_[level]) {
            const float old = c->maxKnnDist;
            recomputeMaxKnnDist(*c);
            if (c->maxKnnDist == old) continue;
            TreeCluster* parent = c->owner->parent;
            if (parent && parent->mark != epoch_) {
                parent->mark = epoch_;
                pending_[parent->level].push_back(parent);
            }
        }
        pending_[level].clear();
    }
}

}  // namespace dualtree
