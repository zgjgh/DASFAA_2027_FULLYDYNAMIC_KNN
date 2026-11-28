#include "idistance.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "kmeans.h"

namespace dualtree {

IDistance::IDistance(const Dataset& data, int partitions, double radiusStep, Tolerance tol, std::uint64_t seed)
    : data_(data), numPartitions_(partitions), radiusStep_(radiusStep), tol_(tol), seed_(seed) {
    if (partitions < 1 || radiusStep <= 0) throw std::invalid_argument("iDistance needs partitions >= 1 and a positive step");
}

void IDistance::build(const std::vector<PointId>& refs) {
    const std::size_t D = static_cast<std::size_t>(data_.dim());
    partOf_.assign(data_.size(), 0);
    keyOf_.assign(data_.size(), 0.0f);
    KMeansResult km = kmeans(refs, PointVectors{data_[0], D}, data_.dim(), numPartitions_, seed_);
    parts_.assign(km.centers.size(), Partition{});
    float maxRadius = 0.0f;
    for (std::size_t i = 0; i < parts_.size(); ++i) {
        parts_[i].center = std::move(km.centers[i]);
        for (PointId p : km.members[i]) {
            const float key = std::sqrt(squaredDistance(data_[p], parts_[i].center.data(), D));
            parts_[i].keys.emplace_back(key, p);
            partOf_[p] = static_cast<std::uint32_t>(i);
            keyOf_[p] = key;
            maxRadius = std::max(maxRadius, key);
        }
        std::sort(parts_[i].keys.begin(), parts_[i].keys.end());
    }
    step_ = static_cast<float>(radiusStep_) * maxRadius;
    if (!(step_ > 0.0f)) step_ = 1.0f;
}

std::size_t IDistance::nearest(const float* x, float* dist) const {
    const std::size_t D = static_cast<std::size_t>(data_.dim());
    std::size_t best = 0;
    float bestDist = kInf;
    for (std::size_t i = 0; i < parts_.size(); ++i) {
        const float d = squaredDistance(x, parts_[i].center.data(), D);
        if (d < bestDist) {
            bestDist = d;
            best = i;
        }
    }
    *dist = std::sqrt(bestDist);
    return best;
}

void IDistance::add(std::size_t part, float key, PointId p) {
    auto& keys = parts_[part].keys;
    const std::pair<float, PointId> entry(key, p);
    keys.insert(std::lower_bound(keys.begin(), keys.end(), entry), entry);
    partOf_[p] = static_cast<std::uint32_t>(part);
    keyOf_[p] = key;
}

void IDistance::insert(PointId q) {
    float key = 0.0f;
    const std::size_t part = nearest(data_[q], &key);
    add(part, key, q);
}

void IDistance::remove(PointId q) {
    auto& keys = parts_[partOf_[q]].keys;
    const auto it = std::lower_bound(keys.begin(), keys.end(), std::make_pair(keyOf_[q], q));
    if (it != keys.end() && it->second == q) keys.erase(it);
}

void IDistance::knn(const float* x, int k, std::vector<Neighbor>& out) const {
    const std::size_t D = static_cast<std::size_t>(data_.dim());
    const std::size_t m = parts_.size();
    thread_local std::vector<float> dq;
    thread_local std::vector<std::size_t> lo, hi;   // scanned key range [lo, hi) per partition
    dq.resize(m);
    lo.resize(m);
    hi.resize(m);
    for (std::size_t i = 0; i < m; ++i) {
        dq[i] = std::sqrt(squaredDistance(x, parts_[i].center.data(), D));
        const auto& keys = parts_[i].keys;
        lo[i] = hi[i] = static_cast<std::size_t>(
            std::lower_bound(keys.begin(), keys.end(), std::make_pair(dq[i], PointId(0))) - keys.begin());
    }

    out.clear();
    float kth = kInf;
    const auto consider = [&](PointId v) {
        const Neighbor n{std::sqrt(squaredDistance(x, data_[v], D)), v};
        if (static_cast<int>(out.size()) < k) {
            out.push_back(n);
            std::push_heap(out.begin(), out.end());
        } else if (n < out.front()) {
            std::pop_heap(out.begin(), out.end());
            out.back() = n;
            std::push_heap(out.begin(), out.end());
        }
        if (static_cast<int>(out.size()) == k) kth = out.front().dist;
    };

    for (float r = step_;; r += step_) {
        // Points within distance r of x have keys within dq[i] +- r in their partition.
        const float reach = tol_.widen(r);
        bool exhausted = true;
        for (std::size_t i = 0; i < m; ++i) {
            const auto& keys = parts_[i].keys;
            while (lo[i] > 0 && keys[lo[i] - 1].first >= dq[i] - reach) consider(keys[--lo[i]].second);
            while (hi[i] < keys.size() && keys[hi[i]].first <= dq[i] + reach) consider(keys[hi[i]++].second);
            if (lo[i] > 0 || hi[i] < keys.size()) exhausted = false;
        }
        if (exhausted || (static_cast<int>(out.size()) == k && kth <= r)) break;
    }
}

}  // namespace dualtree
