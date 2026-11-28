#include "dataset.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace dualtree {
namespace {

bool endsWith(const std::string& s, const std::string& suffix) {
    return s.size() >= suffix.size() && s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

std::ifstream openBinary(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open " + path);
    return in;
}

std::uint64_t fileSize(std::ifstream& in) {
    in.seekg(0, std::ios::end);
    const std::uint64_t size = static_cast<std::uint64_t>(in.tellg());
    in.seekg(0, std::ios::beg);
    return size;
}

std::uint32_t readBigEndian32(std::ifstream& in) {
    unsigned char b[4];
    in.read(reinterpret_cast<char*>(b), 4);
    return (std::uint32_t(b[0]) << 24) | (std::uint32_t(b[1]) << 16) | (std::uint32_t(b[2]) << 8) |
           std::uint32_t(b[3]);
}

float halfToFloat(std::uint16_t h) {
    const std::uint32_t sign = std::uint32_t(h & 0x8000u) << 16;
    std::uint32_t exponent = (h >> 10) & 0x1Fu;
    std::uint32_t mantissa = h & 0x3FFu;
    std::uint32_t bits;
    if (exponent == 0) {
        if (mantissa == 0) {
            bits = sign;
        } else {  // subnormal half: normalize
            exponent = 127 - 15 + 1;
            while ((mantissa & 0x400u) == 0) {
                mantissa <<= 1;
                --exponent;
            }
            bits = sign | (exponent << 23) | ((mantissa & 0x3FFu) << 13);
        }
    } else if (exponent == 0x1F) {
        bits = sign | 0x7F800000u | (mantissa << 13);
    } else {
        bits = sign | ((exponent + 127 - 15) << 23) | (mantissa << 13);
    }
    float f;
    std::memcpy(&f, &bits, sizeof f);
    return f;
}

// Value of `key` in the header dictionary of an .npy file.
std::string npyField(const std::string& header, const std::string& key) {
    const std::size_t k = header.find("'" + key + "'");
    if (k == std::string::npos) return {};
    std::size_t p = header.find(':', k);
    if (p == std::string::npos) return {};
    ++p;
    while (p < header.size() && header[p] == ' ') ++p;
    if (p >= header.size()) return {};
    if (header[p] == '\'') {
        const std::size_t e = header.find('\'', p + 1);
        return header.substr(p + 1, e - p - 1);
    }
    if (header[p] == '(') {
        const std::size_t e = header.find(')', p);
        return header.substr(p + 1, e - p - 1);
    }
    const std::size_t e = header.find_first_of(",}", p);
    return header.substr(p, e - p);
}

}  // namespace

Dataset Dataset::load(const std::vector<std::string>& paths, std::size_t maxRecords) {
    if (paths.empty()) throw std::runtime_error("no dataset file given");
    Dataset ds;
    for (const std::string& path : paths) {
        if (maxRecords && ds.n_ >= maxRecords) break;
        const std::size_t limit = maxRecords ? maxRecords - ds.n_ : std::numeric_limits<std::size_t>::max();
        if (endsWith(path, ".fvecs")) {
            ds.readFvecs(path, limit);
        } else if (endsWith(path, ".npy")) {
            ds.readNpy(path, limit);
        } else if (path.find("idx3-ubyte") != std::string::npos) {
            ds.readIdx(path, limit);
        } else {
            ds.readText(path, limit);
        }
    }
    if (ds.n_ == 0) throw std::runtime_error("the dataset is empty");
    if (ds.n_ > std::numeric_limits<PointId>::max()) throw std::runtime_error("too many records");
    return ds;
}

void Dataset::setDim(int dim, const std::string& path) {
    if (dim <= 0) throw std::runtime_error(path + ": invalid dimensionality");
    if (dim_ == 0) dim_ = dim;
    if (dim_ != dim) throw std::runtime_error(path + ": dimensionality differs from the previous files");
}

void Dataset::readFvecs(const std::string& path, std::size_t limit) {
    std::ifstream in = openBinary(path);
    const std::uint64_t bytes = fileSize(in);
    std::int32_t d = 0;
    in.read(reinterpret_cast<char*>(&d), sizeof d);
    if (!in || d <= 0) throw std::runtime_error(path + ": not an fvecs file");
    const std::uint64_t recordBytes = 4 + 4 * std::uint64_t(d);
    if (bytes % recordBytes != 0) throw std::runtime_error(path + ": truncated fvecs file");
    setDim(d, path);
    const std::size_t count = static_cast<std::size_t>(std::min<std::uint64_t>(bytes / recordBytes, limit));
    in.seekg(0, std::ios::beg);
    const std::size_t start = values_.size();
    values_.resize(start + count * std::size_t(d));
    std::vector<char> record(recordBytes);
    for (std::size_t i = 0; i < count; ++i) {
        in.read(record.data(), static_cast<std::streamsize>(recordBytes));
        std::int32_t di = 0;
        std::memcpy(&di, record.data(), sizeof di);
        if (!in || di != d) throw std::runtime_error(path + ": inconsistent fvecs record");
        std::memcpy(&values_[start + i * std::size_t(d)], record.data() + 4, 4 * std::size_t(d));
    }
    n_ += count;
}

void Dataset::readNpy(const std::string& path, std::size_t limit) {
    std::ifstream in = openBinary(path);
    char magic[6];
    in.read(magic, 6);
    if (!in || std::memcmp(magic, "\x93NUMPY", 6) != 0) throw std::runtime_error(path + ": not an npy file");
    unsigned char version[2];
    in.read(reinterpret_cast<char*>(version), 2);
    std::uint32_t headerLen = 0;
    if (version[0] == 1) {
        std::uint16_t h = 0;
        in.read(reinterpret_cast<char*>(&h), 2);
        headerLen = h;
    } else {
        in.read(reinterpret_cast<char*>(&headerLen), 4);
    }
    std::string header(headerLen, ' ');
    in.read(&header[0], headerLen);
    if (!in) throw std::runtime_error(path + ": truncated npy header");

    const std::string descr = npyField(header, "descr");
    if (npyField(header, "fortran_order").find("True") != std::string::npos)
        throw std::runtime_error(path + ": Fortran-ordered arrays are not supported");
    std::vector<std::size_t> shape;
    std::stringstream shapeText(npyField(header, "shape"));
    std::string item;
    while (std::getline(shapeText, item, ',')) {
        if (item.find_first_not_of(' ') != std::string::npos) shape.push_back(std::stoull(item));
    }
    if (shape.size() != 2) throw std::runtime_error(path + ": expected a 2-D array");

    int width = 0;
    if (descr == "<f4") width = 4;
    else if (descr == "<f2") width = 2;
    else if (descr == "<f8") width = 8;
    else if (descr == "|u1") width = 1;
    else throw std::runtime_error(path + ": unsupported dtype " + descr);

    setDim(static_cast<int>(shape[1]), path);
    const std::size_t count = std::min(shape[0], limit);
    const std::size_t total = count * shape[1];
    const std::size_t start = values_.size();
    values_.resize(start + total);
    const std::size_t chunk = std::size_t(1) << 20;
    std::vector<char> buffer(chunk * width);
    for (std::size_t done = 0; done < total; done += chunk) {
        const std::size_t m = std::min(chunk, total - done);
        in.read(buffer.data(), static_cast<std::streamsize>(m * width));
        if (!in) throw std::runtime_error(path + ": truncated npy data");
        float* out = &values_[start + done];
        for (std::size_t i = 0; i < m; ++i) {
            const char* src = buffer.data() + i * width;
            if (width == 4) {
                std::memcpy(out + i, src, 4);
            } else if (width == 2) {
                std::uint16_t h;
                std::memcpy(&h, src, 2);
                out[i] = halfToFloat(h);
            } else if (width == 8) {
                double v;
                std::memcpy(&v, src, 8);
                out[i] = static_cast<float>(v);
            } else {
                out[i] = static_cast<float>(static_cast<unsigned char>(*src));
            }
        }
    }
    n_ += count;
}

void Dataset::readIdx(const std::string& path, std::size_t limit) {
    std::ifstream in = openBinary(path);
    if (readBigEndian32(in) != 0x00000803u) throw std::runtime_error(path + ": not an idx3-ubyte file");
    const std::size_t images = readBigEndian32(in);
    const std::size_t rows = readBigEndian32(in);
    const std::size_t cols = readBigEndian32(in);
    if (!in) throw std::runtime_error(path + ": truncated idx header");
    const std::size_t d = rows * cols;
    setDim(static_cast<int>(d), path);
    const std::size_t count = std::min(images, limit);
    const std::size_t start = values_.size();
    values_.resize(start + count * d);
    std::vector<unsigned char> image(d);
    for (std::size_t i = 0; i < count; ++i) {
        in.read(reinterpret_cast<char*>(image.data()), static_cast<std::streamsize>(d));
        if (!in) throw std::runtime_error(path + ": truncated idx data");
        for (std::size_t j = 0; j < d; ++j) values_[start + i * d + j] = static_cast<float>(image[j]);
    }
    n_ += count;
}

void Dataset::readText(const std::string& path, std::size_t limit) {
    std::ifstream in = openBinary(path);
    const std::uint64_t bytes = fileSize(in);
    std::string text(static_cast<std::size_t>(bytes), '\0');
    in.read(&text[0], static_cast<std::streamsize>(bytes));
    if (!in) throw std::runtime_error("cannot read " + path);

    const char* s = text.c_str();
    const std::size_t len = text.size();
    std::size_t pos = 0;
    std::size_t lineValues = 0;
    std::size_t count = 0;
    std::size_t line = 1;
    int dim = dim_;
    auto endLine = [&]() {
        if (lineValues == 0) return;
        if (dim == 0) dim = static_cast<int>(lineValues);
        if (lineValues != static_cast<std::size_t>(dim))
            throw std::runtime_error(path + ": line " + std::to_string(line) + " has " +
                                     std::to_string(lineValues) + " values, expected " + std::to_string(dim));
        lineValues = 0;
        ++count;
    };
    while (pos < len && count < limit) {
        const char c = s[pos];
        if (c == '\n') {
            endLine();
            ++line;
            ++pos;
            continue;
        }
        if (c == ' ' || c == '\t' || c == '\r' || c == ',' || c == ';') {
            ++pos;
            continue;
        }
        char* end = nullptr;
        const float v = std::strtof(s + pos, &end);
        if (end == s + pos)
            throw std::runtime_error(path + ": unexpected character on line " + std::to_string(line));
        values_.push_back(v);
        ++lineValues;
        pos = static_cast<std::size_t>(end - s);
    }
    if (count < limit) endLine();
    if (count == 0) return;
    setDim(dim, path);
    values_.resize((n_ + count) * std::size_t(dim));
    n_ += count;
}

}  // namespace dualtree
