#include "methods.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "delta_tree.h"
#include "hdr_tree.h"
#include "idistance.h"
#include "library_baselines.h"
#include "pca.h"
#include "scan.h"

namespace dualtree {

JoinMethod::JoinMethod(const Dataset& data, const MethodParams& params)
    : data_(data), params_(params), table_(data.size(), params.k) {}

float dataScale(const Dataset& data, const std::vector<PointId>& a, const std::vector<PointId>& b) {
    const std::size_t n = a.size() + b.size();
    const std::size_t step = std::max<std::size_t>(1, n / 100000);
    const std::size_t D = static_cast<std::size_t>(data.dim());
    const std::vector<float> origin(D, 0.0f);
    float m = 0.0f;
    for (std::size_t i = 0; i < n; i += step) {
        const PointId p = i < a.size() ? a[i] : b[i - a.size()];
        m = std::max(m, std::sqrt(squaredDistance(data[p], origin.data(), D)));
    }
    return m;
}

namespace {

std::vector<PointId> concat(const std::vector<PointId>& a, const std::vector<PointId>& b) {
    std::vector<PointId> all(a);
    all.insert(all.end(), b.begin(), b.end());
    return all;
}

// Dual-Tree (Section 4) and Dual-Tree+, which adds the leaf-level pruning of
// Section 5 to both search directions.
class DualTreeJoin final : public JoinMethod {
public:
    DualTreeJoin(const Dataset& data, const MethodParams& params, bool leafPruning)
        : JoinMethod(data, params), leafPruning_(leafPruning) {}

    BuildReport build(const std::vector<PointId>& queries, const std::vector<PointId>& refs) override {
        BuildReport report;
        const int D = data_.dim();
        Timer buildTimer;
        tol_ = Tolerance(dataScale(data_, queries, refs));
        const std::vector<PointId> all = concat(queries, refs);
        pca_.fit(data_, all);   // covariance of the initial U and I
        const LevelDims refDims(pca_.eigenvalues(),
                                estimatedHeight(refs.size(), params_.refFanout, params_.refLeafSize));
        const LevelDims queryDims(pca_.eigenvalues(),
                                  estimatedHeight(queries.size(), params_.queryFanout, params_.queryLeafSize));
        int length = std::max(refDims.maxReduced(), queryDims.maxReduced());

        const bool fixedDims = params_.pruneDimKnn > 0 || params_.pruneDimRknn > 0;
        int d1 = 0, d2 = 0;
        if (leafPruning_ && fixedDims) {
            dimKnn_ = std::min(std::max(params_.pruneDimKnn, 0), D);
            dimRknn_ = std::min(std::max(params_.pruneDimRknn, 0), D);
            length = std::max({length, dimKnn_, dimRknn_});
        } else if (leafPruning_ && D >= 2) {
            d2 = std::min(D, params_.fitDim2 > 0 ? params_.fitDim2 : std::max(2, D / 4));
            d1 = std::min(d2 - 1, params_.fitDim1 > 0 ? params_.fitDim1 : std::max(1, D / 16));
            length = std::max(length, d2);
        }

        proj_ = std::make_unique<Projections>(data_, pca_);
        proj_->setLength(length);
        proj_->activate(all);
        delta_ = std::make_unique<DeltaTree>(data_, *proj_, refDims, params_.refFanout, params_.refLeafSize, tol_,
                                             params_.seed);
        delta_->build(refs);
        report.buildSeconds = buildTimer.seconds();

        // The rows of the initial queries are not part of the build time;
        // HDR-Tree needs their kNN distances.
        fillRows(table_, queries, [&](PointId u, std::vector<Neighbor>& row) { delta_->knn(u, params_.k, 0, row); });

        Timer hdrTimer;
        hdr_ = std::make_unique<HdrTree>(data_, proj_.get(), queryDims, table_, params_.queryFanout,
                                         params_.queryLeafSize, tol_, params_.seed + 1);
        hdr_->build(queries);
        report.buildSeconds += hdrTimer.seconds();

        // Section 5: one cost model per search direction, fitted before the updates.
        if (leafPruning_ && !fixedDims && d1 > 0) {
            Timer fitTimer;
            report.knnModel = fitKnnSide(*delta_, *hdr_, params_.k, params_.sampleRate, d1, d2);
            report.rknnModel = fitRknnSide(*hdr_, *delta_, params_.sampleRate, d1, d2);
            dimKnn_ = report.knnModel.optimalDim;
            dimRknn_ = report.rknnModel.optimalDim;
            proj_->setLength(std::max(dimKnn_, dimRknn_));
            report.fitSeconds = fitTimer.seconds();
            report.fitted = true;
            report.share = fitVarianceShare(pca_.eigenvalues());
        }
        report.pruneDimKnn = dimKnn_;
        report.pruneDimRknn = dimRknn_;
        return report;
    }

    // Algorithm 4: kNN search on Delta-Tree gives the row and dknn(u, I),
    // which Algorithm 1 needs. No other row changes.
    void insertQuery(PointId u) override {
        proj_->activate(u);
        delta_->knn(u, params_.k, dimKnn_, row_);
        table_.setRow(u, row_);
        hdr_->insert(u);
    }

    // Case 2: the row is removed and Algorithm 2 updates HDR-Tree.
    void deleteQuery(PointId u) override {
        table_.clearRow(u);
        hdr_->remove(u);
    }

    // Algorithm 5: RkNN search reads the kNN distances before any repair; q
    // replaces the k-th neighbor of every affected query; one refresh pass
    // follows. Delta-Tree is not read, so q is inserted into it last.
    void insertReference(PointId q) override {
        proj_->activate(q);
        affected_.clear();
        hdr_->rknn(q, RknnKind::Insertion, dimRknn_, affected_);
        repaired_.clear();
        for (const Neighbor& a : affected_) {
            table_.replaceKth(a.id, Neighbor{a.dist, q});
            repaired_.push_back(a.id);
        }
        hdr_->refresh(repaired_);
        delta_->insert(q);
    }

    // Algorithm 6: q leaves Delta-Tree first, because the repairs run kNN
    // search over I \ {q}; RkNN search finds the rows that contain q.
    void deleteReference(PointId q) override {
        delta_->remove(q);
        affected_.clear();
        hdr_->rknn(q, RknnKind::Deletion, dimRknn_, affected_);
        repaired_.clear();
        for (const Neighbor& a : affected_) {
            delta_->knn(a.id, params_.k, dimKnn_, row_);
            table_.setRow(a.id, row_);
            repaired_.push_back(a.id);
        }
        hdr_->refresh(repaired_);
    }

private:
    bool leafPruning_;
    Pca pca_;
    std::unique_ptr<Projections> proj_;
    std::unique_ptr<DeltaTree> delta_;
    std::unique_ptr<HdrTree> hdr_;
    int dimKnn_ = 0;
    int dimRknn_ = 0;
    std::vector<Neighbor> row_, affected_;
    std::vector<PointId> repaired_;
};

// Delta-Tree baseline: only I is indexed; RkNN search scans U.
class DeltaTreeJoin final : public JoinMethod {
public:
    DeltaTreeJoin(const Dataset& data, const MethodParams& params) : JoinMethod(data, params), queries_(data.size()) {}

    BuildReport build(const std::vector<PointId>& queries, const std::vector<PointId>& refs) override {
        tol_ = Tolerance(dataScale(data_, queries, refs));
        const std::vector<PointId> all = concat(queries, refs);
        pca_.fit(data_, all);
        const LevelDims refDims(pca_.eigenvalues(),
                                estimatedHeight(refs.size(), params_.refFanout, params_.refLeafSize));
        proj_ = std::make_unique<Projections>(data_, pca_);
        proj_->setLength(refDims.maxReduced());
        proj_->activate(all);
        delta_ = std::make_unique<DeltaTree>(data_, *proj_, refDims, params_.refFanout, params_.refLeafSize, tol_,
                                             params_.seed);
        delta_->build(refs);
        fillRows(table_, queries, [&](PointId u, std::vector<Neighbor>& row) { delta_->knn(u, params_.k, 0, row); });
        queries_.insert(queries);
        return {};
    }

    void insertQuery(PointId u) override {
        proj_->activate(u);
        delta_->knn(u, params_.k, 0, row_);
        table_.setRow(u, row_);
        queries_.insert(u);
    }

    void deleteQuery(PointId u) override {
        table_.clearRow(u);
        queries_.erase(u);
    }

    void insertReference(PointId q) override {
        proj_->activate(q);
        affected_.clear();
        scanRknn(data_, q, RknnKind::Insertion, table_, queries_.items(), tol_, affected_);
        for (const Neighbor& a : affected_) table_.replaceKth(a.id, Neighbor{a.dist, q});
        delta_->insert(q);
    }

    void deleteReference(PointId q) override {
        delta_->remove(q);
        affected_.clear();
        scanRknn(data_, q, RknnKind::Deletion, table_, queries_.items(), tol_, affected_);
        for (const Neighbor& a : affected_) {
            delta_->knn(a.id, params_.k, 0, row_);
            table_.setRow(a.id, row_);
        }
    }

private:
    Pca pca_;
    std::unique_ptr<Projections> proj_;
    std::unique_ptr<DeltaTree> delta_;
    LiveSet queries_;
    std::vector<Neighbor> row_, affected_;
};

// Query-side baselines: HDR-Tree (with the extension of Section 4.1) and
// Spheretree index only U; kNN search scans I.
class QueryTreeJoin final : public JoinMethod {
public:
    QueryTreeJoin(const Dataset& data, const MethodParams& params, bool reduced)
        : JoinMethod(data, params), reduced_(reduced), refs_(data.size()) {}

    BuildReport build(const std::vector<PointId>& queries, const std::vector<PointId>& refs) override {
        tol_ = Tolerance(dataScale(data_, queries, refs));
        LevelDims dims = LevelDims::fullSpace(data_.dim());
        if (reduced_) {
            const std::vector<PointId> all = concat(queries, refs);
            pca_.fit(data_, all);
            dims = LevelDims(pca_.eigenvalues(),
                             estimatedHeight(queries.size(), params_.queryFanout, params_.queryLeafSize));
            proj_ = std::make_unique<Projections>(data_, pca_);
            proj_->setLength(dims.maxReduced());
            proj_->activate(all);
        }
        refs_.insert(refs);
        fillRows(table_, queries,
                 [&](PointId u, std::vector<Neighbor>& row) { scanKnn(data_, data_[u], refs_.items(), params_.k, row); });
        tree_ = std::make_unique<HdrTree>(data_, proj_.get(), dims, table_, params_.queryFanout, params_.queryLeafSize,
                                          tol_, params_.seed + 1,
                                          reduced_ ? Descent::SmallestRadiusPlusKnnDist : Descent::NearestCenter);
        tree_->build(queries);
        return {};
    }

    void insertQuery(PointId u) override {
        if (proj_) proj_->activate(u);
        scanKnn(data_, data_[u], refs_.items(), params_.k, row_);
        table_.setRow(u, row_);
        tree_->insert(u);
    }

    void deleteQuery(PointId u) override {
        table_.clearRow(u);
        tree_->remove(u);
    }

    void insertReference(PointId q) override {
        if (proj_) proj_->activate(q);
        affected_.clear();
        tree_->rknn(q, RknnKind::Insertion, 0, affected_);
        repaired_.clear();
        for (const Neighbor& a : affected_) {
            table_.replaceKth(a.id, Neighbor{a.dist, q});
            repaired_.push_back(a.id);
        }
        tree_->refresh(repaired_);
        refs_.insert(q);
    }

    void deleteReference(PointId q) override {
        refs_.erase(q);
        affected_.clear();
        tree_->rknn(q, RknnKind::Deletion, 0, affected_);
        repaired_.clear();
        for (const Neighbor& a : affected_) {
            scanKnn(data_, data_[a.id], refs_.items(), params_.k, row_);
            table_.setRow(a.id, row_);
            repaired_.push_back(a.id);
        }
        tree_->refresh(repaired_);
    }

private:
    bool reduced_;
    Pca pca_;
    std::unique_ptr<Projections> proj_;
    std::unique_ptr<HdrTree> tree_;
    LiveSet refs_;
    std::vector<Neighbor> row_, affected_;
    std::vector<PointId> repaired_;
};

// iDistance baseline: kNN search on iDistance over I; RkNN search scans U.
class IDistanceJoin final : public JoinMethod {
public:
    IDistanceJoin(const Dataset& data, const MethodParams& params) : JoinMethod(data, params), queries_(data.size()) {}

    BuildReport build(const std::vector<PointId>& queries, const std::vector<PointId>& refs) override {
        tol_ = Tolerance(dataScale(data_, queries, refs));
        index_ = std::make_unique<IDistance>(data_, params_.partitions, params_.radiusStep, tol_, params_.seed);
        index_->build(refs);
        fillRows(table_, queries, [&](PointId u, std::vector<Neighbor>& row) { index_->knn(data_[u], params_.k, row); });
        queries_.insert(queries);
        return {};
    }

    void insertQuery(PointId u) override {
        index_->knn(data_[u], params_.k, row_);
        table_.setRow(u, row_);
        queries_.insert(u);
    }

    void deleteQuery(PointId u) override {
        table_.clearRow(u);
        queries_.erase(u);
    }

    void insertReference(PointId q) override {
        affected_.clear();
        scanRknn(data_, q, RknnKind::Insertion, table_, queries_.items(), tol_, affected_);
        for (const Neighbor& a : affected_) table_.replaceKth(a.id, Neighbor{a.dist, q});
        index_->insert(q);
    }

    void deleteReference(PointId q) override {
        index_->remove(q);
        affected_.clear();
        scanRknn(data_, q, RknnKind::Deletion, table_, queries_.items(), tol_, affected_);
        for (const Neighbor& a : affected_) {
            index_->knn(data_[a.id], params_.k, row_);
            table_.setRow(a.id, row_);
        }
    }

private:
    std::unique_ptr<IDistance> index_;
    LiveSet queries_;
    std::vector<Neighbor> row_, affected_;
};

}  // namespace

std::unique_ptr<JoinMethod> makeMethod(const std::string& name, const Dataset& data, const MethodParams& params) {
    if (name == "dual-tree") return std::make_unique<DualTreeJoin>(data, params, false);
    if (name == "dual-tree+") return std::make_unique<DualTreeJoin>(data, params, true);
    if (name == "delta-tree") return std::make_unique<DeltaTreeJoin>(data, params);
    if (name == "hdr-tree") return std::make_unique<QueryTreeJoin>(data, params, true);
    if (name == "spheretree") return std::make_unique<QueryTreeJoin>(data, params, false);
    if (name == "idistance") return std::make_unique<IDistanceJoin>(data, params);
    if (name == "faiss") return makeFaissJoin(data, params);
    if (name == "nanoflann") return makeNanoflannJoin(data, params);
    return nullptr;
}

std::vector<std::string> availableMethods() {
    std::vector<std::string> names{"dual-tree", "dual-tree+", "delta-tree", "hdr-tree", "idistance", "spheretree"};
    if (haveFaiss()) names.push_back("faiss");
    if (haveNanoflann()) names.push_back("nanoflann");
    return names;
}

}  // namespace dualtree
