/**
 * Copyright (c) 2015-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD+Patents license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Copyright 2004-present Facebook. All Rights Reserved.

#pragma once

#include "../utils/DeviceTensor.cuh"
#include "../utils/Float16.cuh"

namespace faiss { namespace gpu {

class GpuResources;

/// Calculates brute-force L2 distance between `vectors` and
/// `queries`, returning the k closest results seen
void runL2Distance(GpuResources* resources,
                   Tensor<float, 2, true>& vectors,
                   Tensor<float, 2, true>* vectorsTransposed,
                   // can be optionally pre-computed; nullptr if we
                   // have to compute it upon the call
                   Tensor<float, 1, true>* vectorNorms,
                   Tensor<float, 2, true>& queries,
                   int k,
                   Tensor<float, 2, true>& outDistances,
                   Tensor<int, 2, true>& outIndices,
                   // Do we care about `outDistances`? If not, we can
                   // take shortcuts.
                   bool ignoreOutDistances = false);

void runL2Distance(GpuResources* resources,
                   Tensor<half, 2, true>& vectors,
                   Tensor<half, 2, true>* vectorsTransposed,
                   Tensor<half, 1, true>* vectorNorms,
                   Tensor<half, 2, true>& queries,
                   int k,
                   Tensor<half, 2, true>& outDistances,
                   Tensor<int, 2, true>& outIndices,
                   bool useHgemm,
                   bool ignoreOutDistances = false);

/// Calculates brute-force inner product distance between `vectors`
/// and `queries`, returning the k closest results seen
void runIPDistance(GpuResources* resources,
                   Tensor<float, 2, true>& vectors,
                   Tensor<float, 2, true>* vectorsTransposed,
                   Tensor<float, 2, true>& queries,
                   int k,
                   Tensor<float, 2, true>& outDistances,
                   Tensor<int, 2, true>& outIndices);

#ifdef FAISS_USE_FLOAT16
void runIPDistance(GpuResources* resources,
                   Tensor<half, 2, true>& vectors,
                   Tensor<half, 2, true>* vectorsTransposed,
                   Tensor<half, 2, true>& queries,
                   int k,
                   Tensor<half, 2, true>& outDistances,
                   Tensor<int, 2, true>& outIndices,
                   bool useHgemm);

void runIPDistance(GpuResources* resources,
                   Tensor<int8_t , 2, true>& vectors,
                   Tensor<int8_t, 2, true>* vectorsTransposed,
                   Tensor<int8_t, 2, true>& queries,
                   int k,
                   Tensor<float, 2, true>& outDistances,
                   Tensor<int, 2, true>& outIndices,
                   Tensor<float, 1, true>& normsInt8,
                   Tensor<float, 1, true>& queryNorms,
                   bool use_int8_norms,
                   bool useHgemm);

#endif

} } // namespace
