// Methods that maintain kNN_join(U, I) under insertions and deletions in U and I.
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "cost_model.h"
#include "dataset.h"
#include "join_table.h"

namespace dualtree {

struct MethodParams {
    int k = 10;
    int refFanout = 10;          // Delta-Tree: fanout f
    int refLeafSize = 20;        // Delta-Tree: leaf-size threshold theta
    int queryFanout = 10;        // HDR-Tree and Spheretree: fanout f
    int queryLeafSize = 20;      // HDR-Tree and Spheretree: leaf-size threshold theta
    double sampleRate = 0.1;     // cost model: fraction of every leaf that is sampled
    int fitDim1 = 0;             // cost model: d1 (0 selects D/16)
    int fitDim2 = 0;             // cost model: d2 (0 selects D/4)
    int pruneDimKnn = 0;         // fixed leaf-level pruning dimensionalities for Dual-Tree+;
    int pruneDimRknn = 0;        //   when both are 0 the cost model chooses them
    int partitions = 64;         // iDistance: number of partitions
    double radiusStep = 0.01;    // iDistance: radius step, relative to the largest partition radius
    std::uint64_t seed = 1;
};

// Outcome of build(). Dual-Tree+ also reports its fitted cost models and the
// leaf-level pruning dimensionalities in use.
struct BuildReport {
    double buildSeconds = 0.0;   // building the indexes on the initial sets
    double fitSeconds = 0.0;     // fitting the cost model of both search directions
    bool fitted = false;
    CostModel knnModel;
    CostModel rknnModel;
    ShareFit share;              // variance share of the initial U and I ~ k ln d + b
    int pruneDimKnn = 0;
    int pruneDimRknn = 0;
};

// After every update the table is exact and the method's indexes are current.
class JoinMethod {
public:
    JoinMethod(const Dataset& data, const MethodParams& params);
    virtual ~JoinMethod() = default;

    // Builds the indexes over the initial U and I and fills the rows of U.
    virtual BuildReport build(const std::vector<PointId>& queries, const std::vector<PointId>& refs) = 0;
    virtual void insertQuery(PointId u) = 0;        // Case 1
    virtual void deleteQuery(PointId u) = 0;        // Case 2
    virtual void insertReference(PointId q) = 0;    // Case 3
    virtual void deleteReference(PointId q) = 0;    // Case 4

    const JoinTable& table() const { return table_; }

protected:
    const Dataset& data_;
    MethodParams params_;
    JoinTable table_;
    Tolerance tol_;
};

// dual-tree, dual-tree+, delta-tree, hdr-tree, idistance, spheretree, and,
// when built with the libraries, faiss and nanoflann.
std::unique_ptr<JoinMethod> makeMethod(const std::string& name, const Dataset& data, const MethodParams& params);
std::vector<std::string> availableMethods();

// Magnitude of the data vectors, for the rounding margin of the pruning tests.
float dataScale(const Dataset& data, const std::vector<PointId>& a, const std::vector<PointId>& b);

// Fills the rows of `queries`, in parallel, with knn(u, row).
template <class KnnFn>
void fillRows(JoinTable& table, const std::vector<PointId>& queries, KnnFn knn) {
    const std::ptrdiff_t n = static_cast<std::ptrdiff_t>(queries.size());
#pragma omp parallel
    {
        std::vector<Neighbor> row;
#pragma omp for schedule(dynamic, 64)
        for (std::ptrdiff_t i = 0; i < n; ++i) {
            knn(queries[i], row);
            table.setRow(queries[i], row);
        }
    }
}

}  // namespace dualtree
