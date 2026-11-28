#include "hdr_tree.h"

#include <cmath>

namespace dualtree {

HdrTree::HdrTree(const Dataset& data, const Projections* proj, const LevelDims& dims, const JoinTable& table,
                 int fanout, int leafSize, Tolerance tol, std::uint64_t seed, Descent descent)
    : data_(data),
      proj_(proj),
      table_(table),
      descent_(descent),
      tree_(data, proj, dims, TreeParams{fanout, leafSize, true}, tol, seed, table.knnDists()) {}

void HdrTree::rknn(PointId q, RknnKind kind, int pruneDim, std::vector<Neighbor>& out, PruneProbe* probe) const {
    thread_local std::vector<const TreeNode*> stack;
    const Tolerance& tol = tree_.tolerance();
    const std::size_t D = static_cast<std::size_t>(data_.dim());
    const float* xq = data_[q];
    const float* yq = (pruneDim > 0 || probe) ? (*proj_)(q) : nullptr;
    stack.clear();
    stack.push_back(&tree_.root());

    while (!stack.empty()) {
        const TreeNode* node = stack.back();
        stack.pop_back();
        if (!node->isLeaf()) {
            for (const auto& c : node->clusters) {
                if (c->points.empty()) continue;
                if (tol.exceeds(tree_.lowerBound(*c, q), c->maxKnnDist)) continue;
                stack.push_back(c->child.get());
            }
            continue;
        }

        for (PointId v : node->parent->points) {
            const float t = tol.widen(table_.knnDist(v));   // b_p = dknn(v, I)
            if (probe) {
                ++probe->visited;
                const float* yv = (*proj_)(v);
                const float s1 = squaredDistance(yq, yv, static_cast<std::size_t>(probe->d1));
                const float s2 = s1 + squaredDistance(yq + probe->d1, yv + probe->d1,
                                                      static_cast<std::size_t>(probe->d2 - probe->d1));
                if (s1 > t * t) ++probe->pruned1;
                if (s2 > t * t) {
                    ++probe->pruned2;
                    continue;
                }
            } else if (pruneDim > 0 &&
                       squaredDistance(yq, (*proj_)(v), static_cast<std::size_t>(pruneDim)) > t * t) {
                continue;   // leaf-level check (Section 5)
            }
            const float d = std::sqrt(squaredDistance(xq, data_[v], D));
            if (d > t) continue;   // dist(q, v) > dknn(v, I): the row of v does not change
            if (kind == RknnKind::Insertion) {
                if (table_.admits(v, Neighbor{d, q})) out.push_back({d, v});
            } else if (table_.contains(v, q)) {
                out.push_back({d, v});
            }
        }
    }
}

}  // namespace dualtree
