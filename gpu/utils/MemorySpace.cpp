/**
 * Copyright (c) 2015-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD+Patents license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Copyright 2004-present Facebook. All Rights Reserved.

#include "MemorySpace.h"
#include <cuda_runtime.h>

namespace faiss { namespace gpu {

/// Allocates CUDA memory for a given memory space
bool allocMemorySpace(MemorySpace space, void** p, size_t size) {
  if (space == MemorySpace::Device) {
    auto ret = cudaMalloc(p, size);
    if (ret != cudaSuccess)
    {
        fprintf(stderr, "Failed to cudaMalloc %zu bytes\n", size);
        return false;
    }
//    FAISS_ASSERT_FMT(ret == cudaSuccess,
//                     "Failed to cudaMalloc %zu bytes", size);
  }
#ifdef FAISS_UNIFIED_MEM
  else if (space == MemorySpace::Unified) {
    auto ret = cudaMallocManaged(p, size);
    if (ret != cudaSuccess)
    {
      fprintf(stderr, "Failed to cudaMallocManaged %zu bytes\n", size);
    }
//    FAISS_ASSERT_FMT(cudaMallocManaged(p, size) == cudaSuccess,
//                     "Failed to cudaMallocManaged %zu bytes", size);
  }
#endif
  else {
    fprintf(stderr, "Unknown MemorySpace %d\n", (int) space);
    return false;
//    FAISS_ASSERT_FMT(false, "Unknown MemorySpace %d", (int) space);
  }
  return true;
}

} }
