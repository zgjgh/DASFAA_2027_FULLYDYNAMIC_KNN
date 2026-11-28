#include "workload.h"

#include <algorithm>
#include <cmath>
#include <iterator>
#include <stdexcept>

#include "scan.h"

namespace dualtree {
namespace {

// n elements of `pool` chosen uniformly at random, in random order.
std::vector<PointId> sample(const std::vector<PointId>& pool, std::size_t n, std::uint64_t seed) {
    const std::vector<PointId> order = randomPermutation(pool.size(), seed);
    std::vector<PointId> out(n);
    for (std::size_t i = 0; i < n; ++i) out[i] = pool[order[i]];
    return out;
}

std::vector<PointId> join(const std::vector<PointId>& a, const std::vector<PointId>& b) {
    std::vector<PointId> all(a);
    all.insert(all.end(), b.begin(), b.end());
    return all;
}

}  // namespace

Workload makeWorkload(std::size_t S, const WorkloadSpec& spec, int k) {
    if (spec.init <= 0 || spec.update < 0) throw std::invalid_argument("invalid workload fractions");
    const auto count = [&](double fraction) { return static_cast<std::size_t>(std::floor(fraction * S)); };
    const std::vector<PointId> perm = randomPermutation(S, spec.seed);
    std::size_t next = 0;
    const auto take = [&](std::size_t n) {
        if (next + n > S) throw std::invalid_argument("the workload needs more records than the dataset holds");
        std::vector<PointId> part(perm.begin() + static_cast<std::ptrdiff_t>(next),
                                  perm.begin() + static_cast<std::ptrdiff_t>(next + n));
        next += n;
        return part;
    };

    Workload w;
    const std::size_t x = count(spec.init);
    const std::size_t y = count(spec.update);
    if (spec.kind == "mixed") {
        w.queries = take(x);
        w.refs = take(x);
        w.insertQueries = take(y);
        w.insertRefs = take(y);
        w.deleteRefs = sample(join(w.refs, w.insertRefs), y, spec.seed + 1);
        const std::size_t z = count(spec.queryDeletions);
        if (z > 0) {
            const std::vector<PointId> pool = join(w.queries, w.insertQueries);
            if (z > pool.size()) throw std::invalid_argument("more query deletions than queries");
            w.deleteQueries = sample(pool, z, spec.seed + 2);
        }
    } else if (spec.kind == "knn-side") {
        if (x + y > S) throw std::invalid_argument("X + Y must not exceed 1");
        w.queries = take(x);
        w.insertQueries = take(y);
        w.refs = take(S - x - y);
    } else if (spec.kind == "rknn-side") {
        if (x + y > S) throw std::invalid_argument("X + Y must not exceed 1");
        w.refs = take(x);
        w.insertRefs = take(y);
        w.queries = take(S - x - y);
    } else {
        throw std::invalid_argument("unknown workload '" + spec.kind + "'");
    }
    if (w.queries.empty() || w.refs.size() + w.insertRefs.size() - w.deleteRefs.size() < static_cast<std::size_t>(k) ||
        w.refs.size() < static_cast<std::size_t>(k))
        throw std::invalid_argument("the workload leaves fewer than k references or no queries");
    return w;
}

double runWorkload(JoinMethod& method, const Workload& w) {
    Timer timer;
    for (PointId u : w.insertQueries) method.insertQuery(u);
    for (PointId q : w.insertRefs) method.insertReference(q);
    for (PointId q : w.deleteRefs) method.deleteReference(q);
    for (PointId u : w.deleteQueries) method.deleteQuery(u);
    return timer.seconds();
}

void finalSets(const Workload& w, std::vector<PointId>& queries, std::vector<PointId>& refs) {
    const auto without = [](std::vector<PointId> all, std::vector<PointId> removed) {
        std::sort(all.begin(), all.end());
        std::sort(removed.begin(), removed.end());
        std::vector<PointId> rest;
        std::set_difference(all.begin(), all.end(), removed.begin(), removed.end(), std::back_inserter(rest));
        return rest;
    };
    queries = without(join(w.queries, w.insertQueries), w.deleteQueries);
    refs = without(join(w.refs, w.insertRefs), w.deleteRefs);
}

VerifyResult verifyTable(const Dataset& data, const JoinTable& table, const std::vector<PointId>& queries,
                         const std::vector<PointId>& refs) {
    VerifyResult result;
    result.rows = queries.size();
    std::size_t mismatches = 0;
    const std::ptrdiff_t n = static_cast<std::ptrdiff_t>(queries.size());
#pragma omp parallel reduction(+ : mismatches)
    {
        std::vector<Neighbor> exact;
#pragma omp for schedule(dynamic, 64)
        for (std::ptrdiff_t i = 0; i < n; ++i) {
            const PointId u = queries[i];
            scanKnn(data, data[u], refs, table.k(), exact);
            std::sort(exact.begin(), exact.end());
            const std::vector<Neighbor> row = table.sortedRow(u);
            bool same = exact.size() == row.size();
            for (std::size_t j = 0; same && j < row.size(); ++j)
                same = std::abs(row[j].dist - exact[j].dist) <= 1e-4f * std::max(row[j].dist, exact[j].dist);
            if (!same) ++mismatches;
        }
    }
    result.mismatches = mismatches;
    return result;
}

}  // namespace dualtree
