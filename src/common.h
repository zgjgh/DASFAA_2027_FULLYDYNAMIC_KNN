// Shared types and small utilities.
#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace dualtree {

// Points are identified by their record index in the dataset. U and I are
// disjoint sets of identifiers.
using PointId = std::uint32_t;

constexpr float kInf = std::numeric_limits<float>::infinity();

// Squared Euclidean distance over the first n coordinates of a and b.
inline float squaredDistance(const float* a, const float* b, std::size_t n) {
    float s = 0.0f;
#pragma omp simd reduction(+ : s)
    for (std::size_t i = 0; i < n; ++i) {
        const float t = a[i] - b[i];
        s += t * t;
    }
    return s;
}

// One entry of a kNN list. Entries are ordered by distance and then by
// identifier: this is the fixed total order that resolves ties in kNN(u, I).
struct Neighbor {
    float dist = kInf;
    PointId id = 0;
};

inline bool operator<(const Neighbor& a, const Neighbor& b) {
    return a.dist < b.dist || (a.dist == b.dist && a.id < b.id);
}

// Distances and bounds are computed in single precision. Every pruning test
// compares a lower bound with a threshold through this class. It widens the
// threshold by a small margin, so rounding never prunes a point or a cluster
// whose true distance lies within the threshold. Results stay exact.
class Tolerance {
public:
    Tolerance() = default;
    // scale: magnitude of the data vectors.
    explicit Tolerance(float scale) : scale_(scale) {}

    float widen(float threshold) const { return threshold + kRelative * (threshold + scale_); }

    // True if the lower bound shows that the true value exceeds the threshold.
    bool exceeds(float lowerBound, float threshold) const { return lowerBound > widen(threshold); }

private:
    static constexpr float kRelative = 1e-4f;
    float scale_ = 0.0f;
};

// Wall-clock time since construction.
class Timer {
public:
    Timer() : start_(std::chrono::steady_clock::now()) {}
    double seconds() const {
        return std::chrono::duration<double>(std::chrono::steady_clock::now() - start_).count();
    }

private:
    std::chrono::steady_clock::time_point start_;
};

// Small deterministic generator (splitmix64), so that a seed gives the same
// splits and clusterings on every platform.
class Random {
public:
    explicit Random(std::uint64_t seed) : state_(seed) {}

    std::uint64_t next() {
        std::uint64_t z = (state_ += 0x9E3779B97F4A7C15ull);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        return z ^ (z >> 31);
    }

    // Uniform integer in [0, n), n > 0.
    std::uint64_t below(std::uint64_t n) {
        const std::uint64_t max = std::numeric_limits<std::uint64_t>::max();
        const std::uint64_t limit = max - max % n;
        std::uint64_t x;
        do {
            x = next();
        } while (x >= limit);
        return x % n;
    }

private:
    std::uint64_t state_;
};

// Uniformly random permutation of 0 .. n-1 (Fisher-Yates).
inline std::vector<PointId> randomPermutation(std::size_t n, std::uint64_t seed) {
    std::vector<PointId> p(n);
    for (std::size_t i = 0; i < n; ++i) p[i] = static_cast<PointId>(i);
    Random rng(seed);
    for (std::size_t i = n; i > 1; --i) {
        const std::size_t j = static_cast<std::size_t>(rng.below(i));
        const PointId t = p[i - 1];
        p[i - 1] = p[j];
        p[j] = t;
    }
    return p;
}

}  // namespace dualtree
