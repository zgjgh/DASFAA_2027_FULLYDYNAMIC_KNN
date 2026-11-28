#include "library_baselines.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

#ifdef DUALTREE_WITH_FAISS
#include <faiss/utils/Heap.h>
#include <faiss/utils/distances.h>
#endif

#ifdef DUALTREE_WITH_NANOFLANN
#include <nanoflann.hpp>
#endif

namespace dualtree {

#if defined(DUALTREE_WITH_FAISS) || defined(DUALTREE_WITH_NANOFLANN)
namespace {

// Vectors of a live set in one contiguous array, as a flat index stores them.
// Removing a vector moves the last one into its slot.
class FlatStore {
public:
    explicit FlatStore(const Dataset& data) : data_(data), dim_(data.dim()), pos_(data.size(), kAbsent) {}

    void insert(PointId p) {
        pos_[p] = static_cast<std::uint32_t>(ids_.size());
        ids_.push_back(p);
        const float* x = data_[p];
        vectors_.insert(vectors_.end(), x, x + dim_);
    }

    void erase(PointId p) {
        const std::size_t i = pos_[p];
        const std::size_t last = ids_.size() - 1;
        if (i != last) {
            std::memcpy(&vectors_[i * dim_], &vectors_[last * dim_], dim_ * sizeof(float));
            ids_[i] = ids_[last];
            pos_[ids_[i]] = static_cast<std::uint32_t>(i);
        }
        ids_.pop_back();
        vectors_.resize(last * dim_);
        pos_[p] = kAbsent;
    }

    std::size_t size() const { return ids_.size(); }
    const float* vectors() const { return vectors_.data(); }
    const float* vector(std::size_t i) const { return vectors_.data() + i * dim_; }
    PointId id(std::size_t i) const { return ids_[i]; }

private:
    static constexpr std::uint32_t kAbsent = 0xFFFFFFFFu;
    const Dataset& data_;
    std::size_t dim_;
    std::vector<float> vectors_;
    std::vector<PointId> ids_;
    std::vector<std::uint32_t> pos_;
};

// Maintenance shared by the flat baselines. Case 3 compares the new
// reference with every query; Case 4 finds the rows that contain the deleted
// reference by the same scan and recomputes them over all references.
class FlatJoin : public JoinMethod {
public:
    FlatJoin(const Dataset& data, const MethodParams& params)
        : JoinMethod(data, params), refs_(data), queries_(data) {}

    BuildReport build(const std::vector<PointId>& queries, const std::vector<PointId>& refs) override {
        tol_ = Tolerance(dataScale(data_, queries, refs));
        for (PointId q : refs) refs_.insert(q);
        for (PointId u : queries) queries_.insert(u);
        fillRows(table_, queries, [&](PointId u, std::vector<Neighbor>& row) { knn(data_[u], row); });
        return {};
    }

    void insertQuery(PointId u) override {
        knn(data_[u], row_);
        table_.setRow(u, row_);
        queries_.insert(u);
    }

    void deleteQuery(PointId u) override {
        table_.clearRow(u);
        queries_.erase(u);
    }

    void insertReference(PointId q) override {
        distancesToQueries(data_[q], dist_);
        for (std::size_t i = 0; i < queries_.size(); ++i) {
            const PointId u = queries_.id(i);
            const Neighbor n{std::sqrt(dist_[i]), q};
            if (table_.admits(u, n)) table_.replaceKth(u, n);
        }
        refs_.insert(q);
    }

    void deleteReference(PointId q) override {
        refs_.erase(q);
        distancesToQueries(data_[q], dist_);
        affected_.clear();
        for (std::size_t i = 0; i < queries_.size(); ++i) {
            const PointId u = queries_.id(i);
            const float t = tol_.widen(table_.knnDist(u));
            if (dist_[i] <= t * t && table_.contains(u, q)) affected_.push_back(u);
        }
        for (PointId u : affected_) {
            knn(data_[u], row_);
            table_.setRow(u, row_);
        }
    }

protected:
    // kNN of x over all references (k entries).
    virtual void knn(const float* x, std::vector<Neighbor>& out) const = 0;
    // Squared distances from x to all queries, in store order.
    virtual void distancesToQueries(const float* x, std::vector<float>& out) const = 0;

    FlatStore refs_;
    FlatStore queries_;

private:
    std::vector<Neighbor> row_;
    std::vector<float> dist_;
    std::vector<PointId> affected_;
};

}  // namespace
#endif

#ifdef DUALTREE_WITH_FAISS
namespace {

// FAISS flat search: faiss::knn_L2sqr is the exact kernel of IndexFlatL2.
// The vectors stay in a contiguous buffer so that a deletion costs O(D).
class FaissJoin final : public FlatJoin {
public:
    FaissJoin(const Dataset& data, const MethodParams& params) : FlatJoin(data, params) {
        // Compute every distance directly, never through the BLAS expansion
        // of ||x - y||^2, so that the results stay exact for any batch size.
        faiss::distance_compute_blas_threshold = std::numeric_limits<int>::max();
    }

protected:
    void knn(const float* x, std::vector<Neighbor>& out) const override {
        const std::size_t k = static_cast<std::size_t>(params_.k);
        std::vector<float> dist(k);
        std::vector<int64_t> labels(k);
        faiss::float_maxheap_array_t heap = {1, k, labels.data(), dist.data()};
        faiss::knn_L2sqr(x, refs_.vectors(), static_cast<std::size_t>(data_.dim()), 1, refs_.size(), &heap);
        out.clear();
        for (std::size_t i = 0; i < k; ++i)
            if (labels[i] >= 0) out.push_back({std::sqrt(dist[i]), refs_.id(static_cast<std::size_t>(labels[i]))});
    }

    void distancesToQueries(const float* x, std::vector<float>& out) const override {
        out.resize(queries_.size());
        if (!out.empty())
            faiss::fvec_L2sqr_ny(out.data(), x, queries_.vectors(), static_cast<std::size_t>(data_.dim()),
                                 queries_.size());
    }
};

}  // namespace

std::unique_ptr<JoinMethod> makeFaissJoin(const Dataset& data, const MethodParams& params) {
    return std::make_unique<FaissJoin>(data, params);
}
bool haveFaiss() { return true; }
#else
std::unique_ptr<JoinMethod> makeFaissJoin(const Dataset&, const MethodParams&) { return nullptr; }
bool haveFaiss() { return false; }
#endif

#ifdef DUALTREE_WITH_NANOFLANN
namespace {

// nanoflann dataset adaptor over a FlatStore.
struct StoreAdaptor {
    const FlatStore& store;
    std::size_t kdtree_get_point_count() const { return store.size(); }
    float kdtree_get_pt(std::size_t i, std::size_t d) const { return store.vector(i)[d]; }
    template <class BBox>
    bool kdtree_get_bbox(BBox&) const {
        return false;
    }
};

using L2Metric = nanoflann::L2_Adaptor<float, StoreAdaptor, float>;

// nanoflann in brute-force mode: every point is compared through the L2
// metric and the k-NN result set of the library, as in a single leaf.
class NanoflannJoin final : public FlatJoin {
public:
    using FlatJoin::FlatJoin;

protected:
    void knn(const float* x, std::vector<Neighbor>& out) const override {
        const std::size_t k = static_cast<std::size_t>(params_.k);
        const StoreAdaptor adaptor{refs_};
        const L2Metric metric(adaptor);
        std::vector<std::uint32_t> index(k);
        std::vector<float> dist(k);
        nanoflann::KNNResultSet<float, std::uint32_t> result(k);
        result.init(index.data(), dist.data());
        const std::size_t D = static_cast<std::size_t>(data_.dim());
        for (std::size_t i = 0; i < refs_.size(); ++i)
            result.addPoint(metric.evalMetric(x, i, D), static_cast<std::uint32_t>(i));
        out.clear();
        for (std::size_t j = 0; j < result.size(); ++j) out.push_back({std::sqrt(dist[j]), refs_.id(index[j])});
    }

    void distancesToQueries(const float* x, std::vector<float>& out) const override {
        const StoreAdaptor adaptor{queries_};
        const L2Metric metric(adaptor);
        const std::size_t D = static_cast<std::size_t>(data_.dim());
        out.resize(queries_.size());
        for (std::size_t i = 0; i < out.size(); ++i) out[i] = metric.evalMetric(x, i, D);
    }
};

}  // namespace

std::unique_ptr<JoinMethod> makeNanoflannJoin(const Dataset& data, const MethodParams& params) {
    return std::make_unique<NanoflannJoin>(data, params);
}
bool haveNanoflann() { return true; }
#else
std::unique_ptr<JoinMethod> makeNanoflannJoin(const Dataset&, const MethodParams&) { return nullptr; }
bool haveNanoflann() { return false; }
#endif

}  // namespace dualtree
