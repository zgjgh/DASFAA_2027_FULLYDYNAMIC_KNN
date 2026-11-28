// PCA, the per-level dimensionalities of the trees, and the stored
// principal coordinates of the points (Section 3.1).
#pragma once

#include <cstdint>
#include <vector>

#include "common.h"
#include "dataset.h"

namespace dualtree {

// The rotation onto the principal components preserves Euclidean distances.
// Keeping only the leading d components drops nonnegative squared terms, so
// dist_PC(a, b) on the leading d components is a lower bound of dist(a, b)
// and does not decrease as d grows.
class Pca {
public:
    // Covariance matrix of the given points and its eigendecomposition.
    void fit(const Dataset& data, const std::vector<PointId>& ids);

    int dim() const { return dim_; }
    // e_1 >= e_2 >= ... >= e_D.
    const std::vector<double>& eigenvalues() const { return eigenvalues_; }
    // Writes the first `len` principal coordinates of x, P^T (x - mean).
    void project(const float* x, float* out, int len) const;

private:
    int dim_ = 0;
    std::vector<float> mean_;
    std::vector<float> axes_;   // D x D, row j is the j-th principal axis
    std::vector<double> eigenvalues_;
};

// Dimensionality d_l of the clusters at tree level l. Level 1 uses the first
// principal component. Up to the initial height, the share of the variance
// kept by the leading d_l components grows linearly with the level; from the
// initial height on, a level uses the full dimensionality D.
class LevelDims {
public:
    LevelDims() = default;
    LevelDims(const std::vector<double>& eigenvalues, int height);
    static LevelDims fullSpace(int dim);   // every level uses R^D

    int at(int level) const;
    int full() const { return full_; }
    // Largest d_l below D, i.e. the number of principal coordinates the tree reads.
    int maxReduced() const;

private:
    std::vector<int> dims_;   // dims_[l - 1] = d_l for l <= height
    int full_ = 0;
};

// Height of a tree over n points: clusters at levels 1 .. h-1, leaves at h.
int estimatedHeight(std::size_t n, int fanout, int leafSize);

// Leading principal coordinates of every active point, indexed by point id.
class Projections {
public:
    Projections(const Dataset& data, const Pca& pca);

    int length() const { return length_; }
    // Stores at least `len` coordinates per point (re-projects active points).
    void setLength(int len);
    void activate(PointId p);
    void activate(const std::vector<PointId>& ids);
    const float* operator()(PointId p) const {
        return values_.data() + static_cast<std::size_t>(p) * static_cast<std::size_t>(length_);
    }

private:
    const Dataset& data_;
    const Pca& pca_;
    int length_ = 0;
    std::vector<float> values_;
    std::vector<std::uint8_t> active_;
};

}  // namespace dualtree
