// Experiment entry: loads a dataset, builds one method on the initial sets of a
// workload, applies the update batches, and reports the maintenance time.
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "dataset.h"
#include "methods.h"
#include "workload.h"

using namespace dualtree;

namespace {

const char* kUsage = R"(usage: dualtree --data FILE[,FILE...] --method NAME [options]

  --data FILES          dataset file(s): .fvecs, .npy, *idx3-ubyte, or text with
                        one record per line; several files are concatenated
  --max-records N       use only the first N records
  --method NAME         dual-tree, dual-tree+, delta-tree, hdr-tree, idistance,
                        spheretree, faiss, nanoflann

Workload (fractions of the dataset size S):
  --workload KIND       mixed (default), knn-side or rknn-side
  --init X              initial set size (default 0.25)
  --update Y            size of each update batch (default 0.25)
  --query-deletions Z   mixed only: delete Z S random queries at the end (default 0)
  --k K                 number of neighbors (default 10)
  --seed N              seed of the random splits and of k-means (default 1)

Indexes:
  --ref-fanout F        Delta-Tree fanout (default 10)
  --ref-leaf T          Delta-Tree leaf-size threshold (default 20)
  --query-fanout F      HDR-Tree and Spheretree fanout (default 10)
  --query-leaf T        HDR-Tree and Spheretree leaf-size threshold (default 20)
  --sample R            cost model: fraction of each leaf sampled (default 0.1)
  --fit-dims D1,D2      cost model: the two probe dimensionalities (default D/16,D/4)
  --prune-dim D         dual-tree+: fixed pruning dimensionality for both searches
  --prune-dims DK,DR    dual-tree+: fixed dimensionalities for kNN and RkNN search
  --partitions M        iDistance partitions (default 64)
  --radius-step S       iDistance radius step, relative to the largest
                        partition radius (default 0.01)

Other:
  --threads N           threads for index construction and the rows of the
                        initial queries; updates run one at a time on one thread
  --verify              compare the final table with an exact linear scan
)";

struct Options {
    std::vector<std::string> data;
    std::string method;
    std::size_t maxRecords = 0;
    WorkloadSpec workload;
    MethodParams params;
    int threads = 0;
    bool verify = false;
};

std::vector<std::string> split(const std::string& s, char sep) {
    std::vector<std::string> parts;
    std::stringstream in(s);
    std::string part;
    while (std::getline(in, part, sep))
        if (!part.empty()) parts.push_back(part);
    return parts;
}

std::pair<int, int> intPair(const std::string& s) {
    const std::vector<std::string> v = split(s, ',');
    if (v.size() != 2) throw std::invalid_argument("expected two comma-separated values: " + s);
    return {std::stoi(v[0]), std::stoi(v[1])};
}

Options parse(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        const auto value = [&]() -> std::string {
            if (i + 1 >= argc) throw std::invalid_argument("missing value for " + a);
            return argv[++i];
        };
        if (a == "--data") o.data = split(value(), ',');
        else if (a == "--max-records") o.maxRecords = std::stoull(value());
        else if (a == "--method") o.method = value();
        else if (a == "--workload") o.workload.kind = value();
        else if (a == "--init") o.workload.init = std::stod(value());
        else if (a == "--update") o.workload.update = std::stod(value());
        else if (a == "--query-deletions") o.workload.queryDeletions = std::stod(value());
        else if (a == "--k") o.params.k = std::stoi(value());
        else if (a == "--seed") o.workload.seed = o.params.seed = std::stoull(value());
        else if (a == "--ref-fanout") o.params.refFanout = std::stoi(value());
        else if (a == "--ref-leaf") o.params.refLeafSize = std::stoi(value());
        else if (a == "--query-fanout") o.params.queryFanout = std::stoi(value());
        else if (a == "--query-leaf") o.params.queryLeafSize = std::stoi(value());
        else if (a == "--sample") o.params.sampleRate = std::stod(value());
        else if (a == "--fit-dims") std::tie(o.params.fitDim1, o.params.fitDim2) = intPair(value());
        else if (a == "--prune-dim") o.params.pruneDimKnn = o.params.pruneDimRknn = std::stoi(value());
        else if (a == "--prune-dims") std::tie(o.params.pruneDimKnn, o.params.pruneDimRknn) = intPair(value());
        else if (a == "--partitions") o.params.partitions = std::stoi(value());
        else if (a == "--radius-step") o.params.radiusStep = std::stod(value());
        else if (a == "--threads") o.threads = std::stoi(value());
        else if (a == "--verify") o.verify = true;
        else if (a == "--help" || a == "-h") {
            std::cout << kUsage;
            std::exit(0);
        } else {
            throw std::invalid_argument("unknown option " + a);
        }
    }
    if (o.data.empty() || o.method.empty()) {
        std::cerr << kUsage;
        throw std::invalid_argument("--data and --method are required");
    }
    if (o.params.k < 1) throw std::invalid_argument("k must be positive");
    return o;
}

std::string baseName(const std::string& path) {
    const std::size_t p = path.find_last_of("/\\");
    return p == std::string::npos ? path : path.substr(p + 1);
}

void setThreads(int n) {
#ifdef _OPENMP
    if (n > 0) omp_set_num_threads(n);
#else
    (void)n;
#endif
}

int defaultThreads() {
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options opt = parse(argc, argv);
        const int threads = opt.threads > 0 ? opt.threads : defaultThreads();

        const Dataset data = Dataset::load(opt.data, opt.maxRecords);
        std::string dataset = baseName(opt.data.front());
        if (opt.data.size() > 1) dataset += " and " + std::to_string(opt.data.size() - 1) + " more files";
        std::printf("Dataset             : %s, %zu records x %d dimensions\n", dataset.c_str(), data.size(),
                    data.dim());

        const Workload w = makeWorkload(data.size(), opt.workload, opt.params.k);
        std::unique_ptr<JoinMethod> method = makeMethod(opt.method, data, opt.params);
        if (!method) {
            std::string names;
            for (const std::string& n : availableMethods()) names += (names.empty() ? "" : ", ") + n;
            throw std::invalid_argument("unknown or unavailable method '" + opt.method + "' (available: " + names + ")");
        }
        std::printf("Method              : %s\n", opt.method.c_str());
        std::printf("Workload            : %s, X = %g, Y = %g, k = %d\n", opt.workload.kind.c_str(), opt.workload.init,
                    opt.workload.update, opt.params.k);
        std::printf("Sets                : |U| = %zu, |I| = %zu; insert %zu queries, insert %zu references, "
                    "delete %zu references, delete %zu queries\n",
                    w.queries.size(), w.refs.size(), w.insertQueries.size(), w.insertRefs.size(), w.deleteRefs.size(),
                    w.deleteQueries.size());
        std::fflush(stdout);

        setThreads(threads);
        const BuildReport build = method->build(w.queries, w.refs);
        if (build.fitted) {
            std::printf("Variance share      : ~ %.4f ln d + %.4f\n", build.share.k, build.share.b);
            for (const auto& side : {std::make_pair("kNN", &build.knnModel), std::make_pair("RkNN", &build.rknnModel)}) {
                const CostModel& m = *side.second;
                std::printf("Cost model (%-4s)   : P(%d) = %.4f, P(%d) = %.4f, alpha = %.4f, beta = %.4f, d* = %d\n",
                            side.first, m.d1, m.p1, m.d2, m.p2, m.alpha, m.beta, m.optimalDim);
            }
        }
        if (build.pruneDimKnn > 0 || build.pruneDimRknn > 0)
            std::printf("Pruning dimensions  : kNN search %d, RkNN search %d\n", build.pruneDimKnn, build.pruneDimRknn);
        std::fflush(stdout);

        setThreads(1);
        const double maintenance = runWorkload(*method, w);
        std::printf("Maintenance time    : %.2f s\n", maintenance);
        if (build.fitted) {
            std::printf("Build               : %.2f s\n", build.buildSeconds);
            std::printf("Fit                 : %.3f s\n", build.fitSeconds);
            std::printf("Share               : %.2f %% of the maintenance time\n",
                        maintenance > 0 ? 100.0 * (build.buildSeconds + build.fitSeconds) / maintenance : 0.0);
        }
        std::fflush(stdout);

        if (opt.verify) {
            setThreads(threads);
            std::vector<PointId> queries, refs;
            finalSets(w, queries, refs);
            const VerifyResult v = verifyTable(data, method->table(), queries, refs);
            std::printf("Verification        : %zu of %zu rows differ from an exact linear scan\n", v.mismatches,
                        v.rows);
        }
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
