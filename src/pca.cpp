#include "pca.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include <Eigen/Dense>

namespace dualtree {

void Pca::fit(const Dataset& data, const std::vector<PointId>& ids) {
    if (ids.empty()) throw std::runtime_error("PCA needs at least one point");
    const int D = data.dim();
    const std::size_t n = ids.size();
    dim_ = D;

    std::vector<double> mean(D, 0.0);
    for (PointId p : ids) {
        const float* x = data[p];
        for (int j = 0; j < D; ++j) mean[j] += x[j];
    }
    for (int j = 0; j < D; ++j) mean[j] /= static_cast<double>(n);

    // Covariance of the centered points, accumulated block by block.
    using RowMatrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    Eigen::MatrixXd cov = Eigen::MatrixXd::Zero(D, D);
    const std::size_t block = 4096;
    RowMatrix x;
    for (std::size_t start = 0; start < n; start += block) {
        const std::size_t m = std::min(block, n - start);
        x.resize(static_cast<Eigen::Index>(m), D);
        for (std::size_t i = 0; i < m; ++i) {
            const float* v = data[ids[start + i]];
            for (int j = 0; j < D; ++j) x(static_cast<Eigen::Index>(i), j) = v[j] - mean[j];
        }
        cov.noalias() += x.transpose() * x;
    }
    cov /= static_cast<double>(n);

    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(cov);
    if (solver.info() != Eigen::Success) throw std::runtime_error("PCA eigendecomposition failed");
    const Eigen::VectorXd& values = solver.eigenvalues();   // ascending
    const Eigen::MatrixXd& vectors = solver.eigenvectors();

    mean_.assign(mean.begin(), mean.end());
    axes_.assign(static_cast<std::size_t>(D) * D, 0.0f);
    eigenvalues_.assign(D, 0.0);
    for (int j = 0; j < D; ++j) {
        const int col = D - 1 - j;   // j-th largest eigenvalue
        eigenvalues_[j] = std::max(0.0, values(col));
        // Fix the sign of each axis so that results do not depend on the solver.
        int largest = 0;
        for (int i = 1; i < D; ++i)
            if (std::abs(vectors(i, col)) > std::abs(vectors(largest, col))) largest = i;
        const double sign = vectors(largest, col) < 0 ? -1.0 : 1.0;
        for (int i = 0; i < D; ++i)
            axes_[static_cast<std::size_t>(j) * D + i] = static_cast<float>(sign * vectors(i, col));
    }
}

void Pca::project(const float* x, float* out, int len) const {
    thread_local std::vector<float> centered;
    centered.resize(dim_);
    for (int i = 0; i < dim_; ++i) centered[i] = x[i] - mean_[i];
    for (int j = 0; j < len; ++j) {
        const float* axis = &axes_[static_cast<std::size_t>(j) * dim_];
        float s = 0.0f;
#pragma omp simd reduction(+ : s)
        for (int i = 0; i < dim_; ++i) s += axis[i] * centered[i];
        out[j] = s;
    }
}

LevelDims::LevelDims(const std::vector<double>& eigenvalues, int height) {
    full_ = static_cast<int>(eigenvalues.size());
    height = std::max(1, height);
    std::vector<double> cumulative(full_ + 1, 0.0);
    for (int i = 0; i < full_; ++i) cumulative[i + 1] = cumulative[i] + eigenvalues[i];
    const double total = cumulative[full_];
    const double first = cumulative[std::min(1, full_)];
    dims_.assign(height, full_);
    for (int level = 1; level < height; ++level) {
        // Variance to keep: e_1 at level 1, rising linearly to the total at the initial height.
        const double target = first + (total - first) * (level - 1) / static_cast<double>(height - 1);
        int d = 1;
        while (d < full_ && cumulative[d] < target * (1.0 - 1e-12)) ++d;
        dims_[level - 1] = d;
    }
    for (int level = 1; level < height; ++level) dims_[level] = std::max(dims_[level], dims_[level - 1]);
}

LevelDims LevelDims::fullSpace(int dim) {
    LevelDims dims;
    dims.full_ = dim;
    return dims;
}

int LevelDims::at(int level) const {
    if (level >= 1 && level <= static_cast<int>(dims_.size())) return dims_[level - 1];
    return full_;
}

int LevelDims::maxReduced() const {
    int m = 0;
    for (int d : dims_)
        if (d < full_) m = std::max(m, d);
    return m;
}

int estimatedHeight(std::size_t n, int fanout, int leafSize) {
    if (n <= static_cast<std::size_t>(leafSize)) return 2;
    const double levels = std::ceil(std::log(static_cast<double>(n) / leafSize) / std::log(static_cast<double>(fanout)));
    return std::max(2, 1 + static_cast<int>(levels));
}

Projections::Projections(const Dataset& data, const Pca& pca)
    : data_(data), pca_(pca), active_(data.size(), 0) {}

void Projections::setLength(int len) {
    len = std::min(len, data_.dim());
    if (len <= length_) return;
    std::vector<PointId> active;
    for (std::size_t p = 0; p < active_.size(); ++p)
        if (active_[p]) active.push_back(static_cast<PointId>(p));
    values_.clear();
    values_.shrink_to_fit();
    length_ = len;
    values_.assign(data_.size() * static_cast<std::size_t>(length_), 0.0f);
    std::fill(active_.begin(), active_.end(), 0);
    activate(active);
}

void Projections::activate(PointId p) {
    if (length_ > 0) pca_.project(data_[p], &values_[static_cast<std::size_t>(p) * length_], length_);
    active_[p] = 1;
}

void Projections::activate(const std::vector<PointId>& ids) {
    const std::ptrdiff_t n = static_cast<std::ptrdiff_t>(ids.size());
#pragma omp parallel for schedule(static)
    for (std::ptrdiff_t i = 0; i < n; ++i) activate(ids[i]);
}

}  // namespace dualtree
