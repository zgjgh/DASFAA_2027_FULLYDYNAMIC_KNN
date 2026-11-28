#include "scan.h"

#include <algorithm>
#include <cmath>

namespace dualtree {

void scanKnn(const Dataset& data, const float* x, const std::vector<PointId>& refs, int k,
             std::vector<Neighbor>& out) {
    const std::size_t D = static_cast<std::size_t>(data.dim());
    out.clear();
    float limit = kInf;   // squared distance that a candidate must not exceed
    for (PointId v : refs) {
        const float s = squaredDistance(x, data[v], D);
        if (s > limit) continue;
        const Neighbor n{std::sqrt(s), v};
        if (static_cast<int>(out.size()) < k) {
            out.push_back(n);
            std::push_heap(out.begin(), out.end());
        } else if (n < out.front()) {
            std::pop_heap(out.begin(), out.end());
            out.back() = n;
            std::push_heap(out.begin(), out.end());
        } else {
            continue;
        }
        if (static_cast<int>(out.size()) == k) {
            const float kth = out.front().dist;
            limit = kth * kth * 1.0001f;   // squaring may round below a distance equal to kth
        }
    }
}

void scanRknn(const Dataset& data, PointId q, RknnKind kind, const JoinTable& table,
              const std::vector<PointId>& queries, const Tolerance& tol, std::vector<Neighbor>& out) {
    const std::size_t D = static_cast<std::size_t>(data.dim());
    const float* xq = data[q];
    for (PointId u : queries) {
        const float t = tol.widen(table.knnDist(u));
        const float s = squaredDistance(xq, data[u], D);
        if (s > t * t) continue;
        const float d = std::sqrt(s);
        if (kind == RknnKind::Insertion) {
            if (table.admits(u, Neighbor{d, q})) out.push_back({d, u});
        } else if (table.contains(u, q)) {
            out.push_back({d, u});
        }
    }
}

}  // namespace dualtree
