// Flat-scan baselines on library kernels: FAISS (flat index) and nanoflann
// (brute-force mode). Both scan all references for kNN search and all
// queries for RkNN search.
#pragma once

#include <memory>

#include "methods.h"

namespace dualtree {

// Null if the program was built without the library.
std::unique_ptr<JoinMethod> makeFaissJoin(const Dataset& data, const MethodParams& params);
std::unique_ptr<JoinMethod> makeNanoflannJoin(const Dataset& data, const MethodParams& params);

bool haveFaiss();
bool haveNanoflann();

}  // namespace dualtree
