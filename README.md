# Fully Dynamic High-Dimensional kNN Join with a Dual Index

This repository implements **Dual-Tree**, which keeps an exact kNN join table
kNN_join(U, I) up to date while both the query set U and the reference set I
receive insertions and deletions. It also implements **Dual-Tree+**, which adds
low-dimensional point-to-point pruning at the leaves, with a cost model that
chooses the pruning dimensionality, and the baselines of the evaluation.

## Where things are

| Paper | Code |
|---|---|
| PCA and the lower bound dist_PC(a, b) <= dist(a, b) (Sec. 3.1) | `src/pca.*` |
| Structure shared by Delta-Tree and HDR-Tree (Sec. 3.2) | `src/cluster_tree.*`, `src/kmeans.*` |
| kNN search on Delta-Tree | `src/delta_tree.*` |
| RkNN search on HDR-Tree | `src/hdr_tree.*` |
| Direct update links, Algorithm 1 (query insertion), Algorithm 2 (query deletion), Algorithm 3 (refresh of dist_MkNN) (Sec. 4.1) | `ClusterTree::insert`, `ClusterTree::remove`, `ClusterTree::refresh` in `src/cluster_tree.cpp` |
| Join table; Algorithms 4, 5, 6 and Case 2 (Sec. 4.2) | `src/join_table.h`; `DualTreeJoin` in `src/methods.cpp` |
| Leaf-level pruning, the cost model, the variance-share model and d* = D alpha (Sec. 5) | leaf scans in `src/delta_tree.cpp` and `src/hdr_tree.cpp`; `src/cost_model.*` |
| Baselines (Sec. 6.1) | `src/methods.cpp`, `src/idistance.*`, `src/scan.*`, `src/library_baselines.*` |
| Data loading, workloads and the experiment entry | `src/dataset.*`, `src/workload.*`, `src/main.cpp` |

## Methods

| Name | Description |
|---|---|
| `dual-tree` | Delta-Tree on I for kNN search and HDR-Tree on U for RkNN search, maintained together with the join table |
| `dual-tree+` | `dual-tree` with the leaf-level check in both searches; each search direction gets its pruning dimensionality from its own cost model, fitted on a sample of every leaf before the updates |
| `delta-tree` | Delta-Tree on I only; RkNN search scans U |
| `hdr-tree` | HDR-Tree (with the extension of Sec. 4.1) on U only; kNN search scans I |
| `idistance` | iDistance on I for kNN search; RkNN search scans U |
| `spheretree` | a cluster tree on U in the full space, without dimensionality reduction, for RkNN search; kNN search scans I |
| `faiss` | exact flat search of FAISS (`knn_L2sqr`, the kernel of `IndexFlatL2`) for kNN search, and the FAISS distance kernel for the RkNN scan over U |
| `nanoflann` | nanoflann in brute-force mode: its L2 metric and k-NN result set over all points |

All methods return the exact join. Lower-bound tests compare single-precision
values with thresholds widened by a small relative margin, so that rounding
never discards a true neighbor.

## Build

Requirements: a C++17 compiler, CMake 3.16 or newer, and Eigen 3.3 or newer.
OpenMP is optional; it is used for index construction and the rows of the
initial queries.

```
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

CMake options:

- `-DDUALTREE_WITH_FAISS=ON` builds the FAISS baseline (FAISS must be installed; set `faiss_DIR` if CMake does not find it).
- `-DDUALTREE_WITH_NANOFLANN=ON` builds the nanoflann baseline (set `NANOFLANN_INCLUDE_DIR` to the folder of `nanoflann.hpp` if it is not installed).
- `-DEIGEN3_INCLUDE_DIR=...` points to Eigen if it is not found automatically.
- `-DDUALTREE_NATIVE=OFF` builds without `-march=native`.

## Data

The datasets are publicly available from the sources cited in the paper.
The program reads `.fvecs`, `.npy` (float32, float16 or float64, C order),
uncompressed IDX image files (`*idx3-ubyte`) and text files with one record per
line. A comma-separated list of files is read as one dataset.

## Running

```
./build/dualtree --data FILE[,FILE...] --method NAME [options]
```

One run builds a method on the initial sets of a workload and applies its
update batches; `./build/dualtree --help` lists the options. Queries and
references are disjoint random subsets of the records, and all sizes are
fractions of the dataset size S:

- `mixed` (default): starts from |U| = |I| = X S, inserts Y S queries, then
  inserts Y S references, then deletes Y S references chosen at random from the
  enlarged I.
- `knn-side`: inserts Y S queries into an initial U of X S against a fixed I
  that holds the remaining records.
- `rknn-side`: inserts Y S references into an initial I of X S against a fixed
  U that holds the remaining records.

`--init X`, `--update Y` and `--k K` set the sizes and the number of
neighbors; the fanout and the leaf-size threshold of the trees, the sample
fraction of the cost model and the iDistance parameters are options as well.
`--prune-dim D` runs `dual-tree+` with a fixed pruning dimensionality instead
of the one chosen by the cost model.

The program reports the maintenance time of the update batches. For
`dual-tree+` with the cost model it also reports the fitted models, the chosen
dimensionalities, the time to build both trees and to fit the model, and their
share of the maintenance time. The rows of the initial queries are computed
before the updates and are not part of any reported time. Updates are applied
one at a time on a single thread. `--verify` compares the final join table with
an exact linear scan.
