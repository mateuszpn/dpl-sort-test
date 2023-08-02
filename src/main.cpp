//==============================================================
// Copyright © 2020 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

// oneDPL headers should be included before standard headers
#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/execution>
#include <oneapi/dpl/iterator>

// #include "/opt/hpc_software/tools/compilers/gnu/12.1.0/include/c++/12.1.0/ranges"
#include <iostream>

#include <sycl/sycl.hpp>
#include <mpi.h>

using namespace __nanorange::nano ;

    template <typename T>
    void generate_random(std::vector<T> &v, std::size_t n,
                         std::size_t bound = 100) {
  for (std::size_t i = 0; i < n; i++) {
    v[i] = lrand48() % bound;
  }
}

int main(int argc, char **argv) {

  using T = int;
  std::vector<std::size_t> sizes = {4, 7, 11, 17, 23, 121};

  // auto policy = oneapi::dpl::execution::dpcpp_default;
  auto policy = oneapi::dpl::execution::make_device_policy(sycl::queue());

  int rank = 0, size = 0;
  MPI_Init(&argc, &argv);

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_rank(MPI_COMM_WORLD, &size);

  for (std::size_t n : sizes) {
    std::cout << rank << ": >>> Test size " << n << std::endl;

    std::vector<T> sv(n);

    generate_random(sv, n, 100);

    std::cout << rank << ":  Input vec ";
    for (int i = 0; i < n; i++) {
      std::cout << sv[i] << ", ";
    }
    std::cout << "\n";

    ranges::subrange<int *, int *, ranges::subrange_kind::sized> segment =
        ranges::subrange(sv.data(), sv.data() + sv.size(), n);

    oneapi::dpl::sort(policy, oneapi::dpl::begin(segment), oneapi::dpl::end(segment));

    std::cout << rank << ": Sorted ";

    for (int i = 0; i < n; i++) {
      std::cout << sv[i] << ", ";
    }
    std::cout << "\n";

    MPI_Barrier(MPI_COMM_WORLD);

  }
}
