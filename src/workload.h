// Workloads of Section 6 and their execution.
#pragma once

#include <string>
#include <vector>

#include "methods.h"

namespace dualtree {

// All sizes are fractions of the dataset size S, and every split into
// initial sets and update batches is drawn uniformly at random.
//   mixed      |U| = |I| = X S; insert Y S queries, then insert Y S references,
//              then delete Y S random references of the enlarged I
//   knn-side   insert Y S queries into an initial U of X S; I is fixed and
//              holds the remaining (1 - X - Y) S records
//   rknn-side  insert Y S references into an initial I of X S; U is fixed and
//              holds the remaining (1 - X - Y) S records
struct WorkloadSpec {
    std::string kind = "mixed";
    double init = 0.25;             // X
    double update = 0.25;           // Y
    double queryDeletions = 0.0;    // mixed only: optional last batch of random query deletions
    std::uint64_t seed = 1;
};

struct Workload {
    std::vector<PointId> queries;         // initial U
    std::vector<PointId> refs;            // initial I
    std::vector<PointId> insertQueries;
    std::vector<PointId> insertRefs;
    std::vector<PointId> deleteRefs;
    std::vector<PointId> deleteQueries;
};

Workload makeWorkload(std::size_t S, const WorkloadSpec& spec, int k);

// Applies the batches in this order: query insertions, reference insertions,
// reference deletions, query deletions. Returns the total maintenance time.
double runWorkload(JoinMethod& method, const Workload& w);

// U and I after all batches.
void finalSets(const Workload& w, std::vector<PointId>& queries, std::vector<PointId>& refs);

struct VerifyResult {
    std::size_t rows = 0;
    std::size_t mismatches = 0;
};

// Compares every row with an exact linear scan over the final I.
VerifyResult verifyTable(const Dataset& data, const JoinTable& table, const std::vector<PointId>& queries,
                         const std::vector<PointId>& refs);

}  // namespace dualtree
