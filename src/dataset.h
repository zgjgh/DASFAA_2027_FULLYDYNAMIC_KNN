// Dataset loading. All records are kept in one row-major float matrix; a
// record is identified by its row index.
#pragma once

#include <string>
#include <vector>

#include "common.h"

namespace dualtree {

class Dataset {
public:
    // Reads one or more files with the same dimensionality and concatenates
    // their records in the given order. The format follows the file name:
    //   *.fvecs       each record is an int32 dimension and that many float32 values
    //   *.npy         2-D array of float32, float16 or float64 in C order
    //   *idx3-ubyte*  IDX image file (one image per record, pixels as values)
    //   otherwise     text, one record per line, values separated by spaces,
    //                 tabs, commas or semicolons
    // At most maxRecords records are kept (0 keeps all of them).
    static Dataset load(const std::vector<std::string>& paths, std::size_t maxRecords = 0);

    std::size_t size() const { return n_; }
    int dim() const { return dim_; }
    const float* operator[](PointId id) const {
        return values_.data() + static_cast<std::size_t>(id) * static_cast<std::size_t>(dim_);
    }

private:
    void readFvecs(const std::string& path, std::size_t limit);
    void readNpy(const std::string& path, std::size_t limit);
    void readIdx(const std::string& path, std::size_t limit);
    void readText(const std::string& path, std::size_t limit);
    void setDim(int dim, const std::string& path);

    std::vector<float> values_;
    std::size_t n_ = 0;
    int dim_ = 0;
};

}  // namespace dualtree
