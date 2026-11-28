#include "delta_tree.h"

#include <algorithm>
#include <cmath>

namespace dualtree {

DeltaTree::DeltaTree(const Dataset& data, const Projections& proj, const LevelDims& dims, int fanout, int leafSize,
                     Tolerance tol, std::uint64_t seed)
    : data_(data), proj_(proj), tree_(data, &proj, dims, TreeParams{fanout, leafSize, false}, tol, seed) {}

void DeltaTree::knn(PointId u, int k, int pruneDim, std::vector<Neighbor>& out, PruneProbe* probe) const {
    struct Entry {
        float bound;
        const TreeNode* node;
    };
    const auto later = [](const Entry& a, const Entry& b) { return a.bound > b.bound; };
    thread_local std::vector<Entry> queue;      // non-leaf nodes, smallest bound first
    thread_local std::vector<Entry> children;   // surviving clusters of the expanded node

    const Tolerance& tol = tree_.tolerance();
    const std::size_t D = static_cast<std::size_t>(data_.dim());
    const float* xu = data_[u];
    const float* yu = (pruneDim > 0 || probe) ? proj_(u) : nullptr;
    float kth = kInf;   // current k-th best distance: the pruning threshold b_p
    out.clear();
    queue.clear();
    queue.push_back({0.0f, &tree_.root()});

    const auto scanLeaf = [&](const std::vector<PointId>& points) {
        for (PointId v : points) {
            const float t = tol.widen(kth);
            if (probe) {
                ++probe->visited;
                const float s1 = squaredDistance(yu, proj_(v), static_cast<std::size_t>(probe->d1));
                const float s2 = s1 + squaredDistance(yu + probe->d1, proj_(v) + probe->d1,
                                                      static_cast<std::size_t>(probe->d2 - probe->d1));
                if (s1 > t * t) ++probe->pruned1;
                if (s2 > t * t) {
                    ++probe->pruned2;
                    continue;
                }
            } else if (pruneDim > 0 && squaredDistance(yu, proj_(v), static_cast<std::size_t>(pruneDim)) > t * t) {
                continue;   // leaf-level check (Section 5)
            }
            const Neighbor n{std::sqrt(squaredDistance(xu, data_[v], D)), v};
            if (static_cast<int>(out.size()) < k) {
                out.push_back(n);
                std::push_heap(out.begin(), out.end());
                if (static_cast<int>(out.size()) == k) kth = out.front().dist;
            } else if (n < out.front()) {
                std::pop_heap(out.begin(), out.end());
                out.back() = n;
                std::push_heap(out.begin(), out.end());
                kth = out.front().dist;
            }
        }
    };

    while (!queue.empty()) {
        std::pop_heap(queue.begin(), queue.end(), later);
        const Entry e = queue.back();
        queue.pop_back();
        if (tol.exceeds(e.bound, kth)) break;   // every remaining cluster is farther

        // Clusters of the node in increasing order of their lower bounds. Leaves
        // are scanned right away; non-leaf children wait in the queue.
        children.clear();
        for (const auto& c : e.node->clusters) {
            if (c->points.empty()) continue;
            const float bound = tree_.lowerBound(*c, u);
            if (!tol.exceeds(bound, kth)) children.push_back({bound, c->child.get()});
        }
        std::sort(children.begin(), children.end(), [](const Entry& a, const Entry& b) { return a.bound < b.bound; });
        for (const Entry& child : children) {
            if (tol.exceeds(child.bound, kth)) break;   // the k-th distance shrank while scanning
            if (child.node->isLeaf()) {
                scanLeaf(child.node->parent->points);
            } else {
                queue.push_back(child);
                std::push_heap(queue.begin(), queue.end(), later);
            }
        }
    }
}

}  // namespace dualtree
