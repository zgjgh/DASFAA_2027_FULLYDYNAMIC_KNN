cmake_minimum_required(VERSION 3.16)
project(dualtree LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(DUALTREE_NATIVE "Optimize for the build machine (-march=native)" ON)
option(DUALTREE_WITH_FAISS "Build the FAISS baseline" OFF)
option(DUALTREE_WITH_NANOFLANN "Build the nanoflann baseline" OFF)

# Eigen 3.3 or newer (header only): an installed package, or EIGEN3_INCLUDE_DIR.
find_package(Eigen3 3.3 QUIET NO_MODULE)
if(NOT TARGET Eigen3::Eigen)
  find_path(EIGEN3_INCLUDE_DIR NAMES Eigen/Dense PATH_SUFFIXES eigen3)
  if(NOT EIGEN3_INCLUDE_DIR)
    message(FATAL_ERROR "Eigen 3.3 or newer is required; set Eigen3_DIR or EIGEN3_INCLUDE_DIR")
  endif()
  add_library(Eigen3::Eigen INTERFACE IMPORTED)
  set_target_properties(Eigen3::Eigen PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${EIGEN3_INCLUDE_DIR}")
endif()

find_package(OpenMP)

add_executable(dualtree
  src/main.cpp
  src/dataset.cpp
  src/pca.cpp
  src/kmeans.cpp
  src/cluster_tree.cpp
  src/delta_tree.cpp
  src/hdr_tree.cpp
  src/cost_model.cpp
  src/scan.cpp
  src/idistance.cpp
  src/methods.cpp
  src/library_baselines.cpp
  src/workload.cpp)

target_link_libraries(dualtree PRIVATE Eigen3::Eigen)
if(OpenMP_CXX_FOUND)
  target_link_libraries(dualtree PRIVATE OpenMP::OpenMP_CXX)
elseif(NOT MSVC)
  target_compile_options(dualtree PRIVATE -fopenmp-simd)
endif()
if(NOT MSVC)
  target_compile_options(dualtree PRIVATE -Wall -Wextra)
  if(DUALTREE_NATIVE)
    target_compile_options(dualtree PRIVATE -march=native)
  endif()
endif()

if(DUALTREE_WITH_FAISS)
  find_package(faiss REQUIRED)
  target_link_libraries(dualtree PRIVATE faiss)
  target_compile_definitions(dualtree PRIVATE DUALTREE_WITH_FAISS)
endif()

if(DUALTREE_WITH_NANOFLANN)
  find_package(nanoflann QUIET)
  if(TARGET nanoflann::nanoflann)
    target_link_libraries(dualtree PRIVATE nanoflann::nanoflann)
  else()
    find_path(NANOFLANN_INCLUDE_DIR nanoflann.hpp)
    if(NOT NANOFLANN_INCLUDE_DIR)
      message(FATAL_ERROR "nanoflann.hpp not found; set NANOFLANN_INCLUDE_DIR")
    endif()
    target_include_directories(dualtree PRIVATE "${NANOFLANN_INCLUDE_DIR}")
  endif()
  target_compile_definitions(dualtree PRIVATE DUALTREE_WITH_NANOFLANN)
endif()
