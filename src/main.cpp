//==============================================================
// Copyright © 2020 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

// oneDPL headers should be included before standard headers
#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/execution>
#include <oneapi/dpl/iterator>

#include <sycl/sycl.hpp>
#include <iostream>

#include<mpi.h>

using namespace sycl;
using namespace std;

template <typename T>
// void generate_random(T *v, std::size_t n, std::size_t bound = 100) {
void generate_random(std::vector<T> &v, std::size_t n, std::size_t bound = 100) {
  for (std::size_t i = 0; i < n; i++) {
    v[i] = lrand48() % bound;
  }
}

template <typename T>
void generate_random(span<T> &v, std::size_t n, std::size_t bound = 100) {
  for (std::size_t i = 0; i < n; i++) {
    v[i] = lrand48() % bound;
  }
}

int main(int argc, char **argv) {

  using T = int;
  std::vector<std::size_t> sizes = { 4, 7, 23 };

  // auto policy = oneapi::dpl::execution::dpcpp_default;
  auto policy = oneapi::dpl::execution::make_device_policy(sycl::queue());

  int rank = 0, size = 0;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_rank(MPI_COMM_WORLD, &size);

  for (std::size_t n : sizes) {
    std::cout << rank << ": >>> Test size " << n << std::endl;

    std::vector<T> sv(n);

    // T *s = std::allocator<T>().allocate(n);
    // T *s = sycl::usm_allocator<T, sycl::usm::alloc::shared, 0>(sycl::queue()).allocate(n); /* aka: mhp::default_allocator */
    // auto sv = span<T>(s, n);

    generate_random(sv, n, 100);

    std::cout << rank << ":   Unsorted ";
    for (int i = 0; i < n; i++) {
      std::cout << sv[i] << ", ";
    }
    std::cout << "\n";

    oneapi::dpl::sort(policy, sv.begin(), sv.end());

    bool sorted = true;
    for (int i = 0; i < n; i++) {
      if(sv[i-1] > sv[i]) {
        sorted = false;
        break;
      }
    }

    if (sorted)
      std::cout << rank << ":     Sorted ";
    else
      std::cout << rank << ": NOT SORTED ";

    for (int i = 0; i < n; i++) {
      std::cout << sv[i] << ", ";
    }
    std::cout << "\n";

    MPI_Barrier(MPI_COMM_WORLD);

    // std::allocator<T>().deallocate(s, n);
    // sycl::usm_allocator<T, sycl::usm::alloc::shared, 0>(q).deallocate(s, n);
  }
}
