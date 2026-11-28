// Cost model for the leaf-level pruning dimensionality (Section 5).
#pragma once

#include <vector>

#include "delta_tree.h"
#include "hdr_tree.h"

namespace dualtree {

// Expected cost of a visited leaf point: E_cost(d) = (D - D P_p(d) + d) c,
// with the pruning probability modeled as P_p(d) ~ alpha ln d + beta.
// Measuring P_p at d1 < d2 gives alpha and beta, and dE_cost/dd =
// c (1 - D alpha / d) vanishes at d* = D alpha.
struct CostModel {
    int d1 = 0;
    int d2 = 0;
    double p1 = 0.0;          // pruning frequency measured at d1
    double p2 = 0.0;          // pruning frequency measured at d2
    double alpha = 0.0;
    double beta = 0.0;
    int optimalDim = 1;       // d*, rounded and clamped to [1, D]
};

CostModel solveCostModel(int fullDim, int d1, double p1, int d2, double p2);

// kNN side: the sampled queries of every HDR-Tree leaf run kNN search on the
// initial I (Delta-Tree) with the check at d1 and at d2.
CostModel fitKnnSide(const DeltaTree& delta, const HdrTree& hdr, int k, double sampleRate, int d1, int d2);

// RkNN side: the sampled references of every Delta-Tree leaf run RkNN search
// on the initial U (HDR-Tree) with the check at d1 and at d2.
CostModel fitRknnSide(const HdrTree& hdr, const DeltaTree& delta, double sampleRate, int d1, int d2);

// Share of the variance kept by the leading d components,
// sum_{i <= d} e_i / sum_i e_i, for d = 1 .. D (entry d - 1).
std::vector<double> varianceShare(const std::vector<double>& eigenvalues);

// Least-squares fit of the variance share by k ln d + b over d = 1 .. D. With
// P_p ~ k' share + b', the fitted model gives k' = alpha / k and
// b' = beta - k' b.
struct ShareFit {
    double k = 0.0;
    double b = 0.0;
};
ShareFit fitVarianceShare(const std::vector<double>& eigenvalues);

}  // namespace dualtree
