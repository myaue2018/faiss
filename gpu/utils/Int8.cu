/**
 * Copyright (c) 2015-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD+Patents license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Copyright 2004-present Facebook. All Rights Reserved.

#include "Int8.cuh"
#include "nvidia/fp16_emu.cuh"
#include <thrust/execution_policy.h>
#include <thrust/transform.h>

#ifdef FAISS_USE_FLOAT16

namespace faiss { namespace gpu {


        //TODO:int8
bool getDeviceSupportsFloat16Math(int device) {
  const auto& prop = getDeviceProperties(device);

  return (prop.major >= 6 ||
          (prop.major == 5 && prop.minor >= 3));
}

struct FloatToInt8 {
  __device__ int8_t operator()(float v) const { return (int8_t)(v*KINT8); }
};

struct Int8ToFloat {
  __device__ float operator()(int8_t v) const { return ((float)(v/KINT8)); }
};

struct Int32ToFloat {
    __device__ float operator()(int32_t v) const { return ((float)(v*IVKINT8)); }
};

void runConvertToInt8(int8_t * out,
                         const float* in,
                         size_t num,
                         cudaStream_t stream) {
//    float* host = new float[num];
//    cudaMemcpy(host,in,num*4,cudaMemcpyDeviceToHost);
//    printf("\n");
//    for (int j = 0; j < num; ++j) {
//        printf("%d:%f ",j,*(host+j));
//    }
//    printf("\n");
  thrust::transform(thrust::cuda::par.on(stream),
                    in, in + num, out, FloatToInt8());
}

void runConvertToFloat32(float* out,
                         const int8_t* in,
                         size_t num,
                         cudaStream_t stream) {
  thrust::transform(thrust::cuda::par.on(stream),
                    in, in + num, out, Int8ToFloat());
}

void runConvertInt32ToFloat32(float* out,
                         const int32_t * in,
                         size_t num,
                         cudaStream_t stream) {
    thrust::transform(thrust::cuda::par.on(stream),
                      in, in + num, out, Int32ToFloat());
}

int8_t hostFloat2Int8(float a) {
    return a*KINT8;
}

} } // namespace

#endif // FAISS_USE_FLOAT16
