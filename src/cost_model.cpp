#include "cost_model.h"

#include <algorithm>
#include <cmath>

namespace dualtree {
namespace {

// A fixed fraction of the points of every leaf, spread evenly over the leaf.
std::vector<PointId> sampleLeaves(const ClusterTree& tree, double rate) {
    std::vector<PointId> sample;
    tree.forEachLeaf([&](const TreeCluster& leaf) {
        const std::size_t n = leaf.points.size();
        if (n == 0) return;
        const std::size_t share = static_cast<std::size_t>(std::ceil(rate * static_cast<double>(n)));
        const std::size_t m = std::min(n, std::max<std::size_t>(1, share));
        for (std::size_t i = 0; i < m; ++i) sample.push_back(leaf.points[i * n / m]);
    });
    return sample;
}

double frequency(std::uint64_t pruned, std::uint64_t visited) {
    return visited ? static_cast<double>(pruned) / static_cast<double>(visited) : 0.0;
}

}  // namespace

CostModel solveCostModel(int fullDim, int d1, double p1, int d2, double p2) {
    CostModel m;
    m.d1 = d1;
    m.d2 = d2;
    m.p1 = p1;
    m.p2 = p2;
    // P1 = alpha ln d1 + beta and P2 = alpha ln d2 + beta.
    m.alpha = (p2 - p1) / (std::log(static_cast<double>(d2)) - std::log(static_cast<double>(d1)));
    m.beta = p1 - m.alpha * std::log(static_cast<double>(d1));
    const long d = std::lround(fullDim * m.alpha);
    m.optimalDim = static_cast<int>(std::min<long>(fullDim, std::max<long>(1, d)));
    return m;
}

CostModel fitKnnSide(const DeltaTree& delta, const HdrTree& hdr, int k, double sampleRate, int d1, int d2) {
    PruneProbe probe;
    probe.d1 = d1;
    probe.d2 = d2;
    std::vector<Neighbor> result;
    for (PointId u : sampleLeaves(hdr.tree(), sampleRate)) delta.knn(u, k, 0, result, &probe);
    return solveCostModel(delta.tree().dims().full(), d1, frequency(probe.pruned1, probe.visited), d2,
                          frequency(probe.pruned2, probe.visited));
}

CostModel fitRknnSide(const HdrTree& hdr, const DeltaTree& delta, double sampleRate, int d1, int d2) {
    PruneProbe probe;
    probe.d1 = d1;
    probe.d2 = d2;
    std::vector<Neighbor> result;
    for (PointId q : sampleLeaves(delta.tree(), sampleRate)) {
        result.clear();
        hdr.rknn(q, RknnKind::Deletion, 0, result, &probe);
    }
    return solveCostModel(hdr.tree().dims().full(), d1, frequency(probe.pruned1, probe.visited), d2,
                          frequency(probe.pruned2, probe.visited));
}

std::vector<double> varianceShare(const std::vector<double>& eigenvalues) {
    std::vector<double> share(eigenvalues.size());
    double total = 0.0;
    for (double e : eigenvalues) total += e;
    double kept = 0.0;
    for (std::size_t d = 0; d < eigenvalues.size(); ++d) {
        kept += eigenvalues[d];
        share[d] = total > 0.0 ? kept / total : 1.0;
    }
    return share;
}

ShareFit fitVarianceShare(const std::vector<double>& eigenvalues) {
    const std::vector<double> share = varianceShare(eigenvalues);
    const std::size_t n = share.size();
    ShareFit fit;
    if (n < 2) return fit;
    double sx = 0.0, sy = 0.0, sxx = 0.0, sxy = 0.0;
    for (std::size_t d = 1; d <= n; ++d) {
        const double x = std::log(static_cast<double>(d));
        const double y = share[d - 1];
        sx += x;
        sy += y;
        sxx += x * x;
        sxy += x * y;
    }
    const double m = static_cast<double>(n);
    fit.k = (m * sxy - sx * sy) / (m * sxx - sx * sx);
    fit.b = (sy - fit.k * sx) / m;
    return fit;
}

}  // namespace dualtree
