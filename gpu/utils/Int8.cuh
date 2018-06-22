/**
 * Copyright (c) 2015-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD+Patents license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Copyright 2004-present Facebook. All Rights Reserved.

#pragma once

#include <cuda.h>
#include "../GpuResources.h"
#include "DeviceTensor.cuh"

// For float16, We use the half datatype, expecting it to be a struct
// as in CUDA 7.5.
#if CUDA_VERSION >= 7050
#define FAISS_USE_FLOAT16 1

// Some compute capabilities have full float16 ALUs.
#if __CUDA_ARCH__ >= 530
#define FAISS_USE_FULL_FLOAT16 1
#endif // __CUDA_ARCH__ types

#endif // CUDA_VERSION

#ifdef FAISS_USE_FLOAT16
#include <cuda_fp16.h>
#endif

namespace faiss { namespace gpu {

#ifdef FAISS_USE_FLOAT16
/// Returns true if the given device supports native float16 math
        bool getDeviceSupportsInt8Math(int device);
// 64 bytes containing 4 half (float16) values v//TODO:int8 wtf
// 64 bytes containing 8 int8
//struct Int8_4 {
//  half2 a;
//  half2 b;
//};
//
//inline __device__ float4 half4ToFloat4(Half4 v) {
//  float2 a = __half22float2(v.a);
//  float2 b = __half22float2(v.b);
//
//  float4 out;
//  out.x = a.x;
//  out.y = a.y;
//  out.z = b.x;
//  out.w = b.y;
//
//  return out;
//}
//
//inline __device__ Half4 float4ToHalf4(float4 v) {
//  float2 a;
//  a.x = v.x;
//  a.y = v.y;
//
//  float2 b;
//  b.x = v.z;
//  b.y = v.w;
//
//  Half4 out;
//  out.a = __float22half2_rn(a);
//  out.b = __float22half2_rn(b);
//
//  return out;
//}
//
//// 128 bytes containing 8 half (float16) values
//struct Half8 {
//  Half4 a;
//  Half4 b;
//};

/// Returns true if the given device supports native float16 math
//bool getDeviceSupportsFloat16Math(int device);
const float KINT8 = 256;
const float IVKINT8 = 1.0f/KINT8/KINT8;
/// Copies `in` to `out` while performing a float32 -> int8 conversion
void runConvertToInt8(int8_t * out,
                         const float* in,
                         size_t num,
                         cudaStream_t stream);

/// Copies `in` to `out` while performing a float16 -> float32
/// conversion
void runConvertToFloat32(float* out,
                         const int8_t* in,
                         size_t num,
                         cudaStream_t stream);
void runConvertInt32ToFloat32(float* out,
                 const int32_t * in,
                 size_t num,
                 cudaStream_t stream);

void runConvertInt32ToFloat32WithoutNorms(float* out,
                              const int32_t * in,
                              size_t num,
                              cudaStream_t stream);

template <int Dim>
void toInt8(cudaStream_t stream,
            Tensor<float, Dim, true>& in,
            Tensor<int8_t, Dim, true>& out) {
  FAISS_ASSERT(in.numElements() == out.numElements());

  // The memory is contiguous (the `true`), so apply a pointwise
  // kernel to convert
  runConvertToInt8(out.data(), in.data(), in.numElements(), stream);
}

template <int Dim>
DeviceTensor<int8_t, Dim, true> toInt8(GpuResources* resources,
                                     cudaStream_t stream,
                                     Tensor<float, Dim, true>& in) {
  DeviceTensor<int8_t, Dim, true> out;
  if (resources) {
    out = std::move(DeviceTensor<int8_t, Dim, true>(
                      resources->getMemoryManagerCurrentDevice(),
                      in.sizes(),
                      stream));
  } else {
    out = std::move(DeviceTensor<int8_t, Dim, true>(in.sizes()));
  }

  toInt8<Dim>(stream, in, out);

  return out;
}

template <int Dim>
void fromInt8(cudaStream_t stream,
            Tensor<int8_t, Dim, true>& in,
            Tensor<float, Dim, true>& out) {
  FAISS_ASSERT(in.numElements() == out.numElements());

  // The memory is contiguous (the `true`), so apply a pointwise
  // kernel to convert
  runConvertToFloat32(out.data(), in.data(), in.numElements(), stream);
}

template <int Dim>
DeviceTensor<float, Dim, true> fromInt8(GpuResources* resources,
                                        cudaStream_t stream,
                                        Tensor<int8_t, Dim, true>& in) {
  DeviceTensor<float, Dim, true> out;
  if (resources) {
    out = std::move(DeviceTensor<float, Dim, true>(
                      resources->getMemoryManagerCurrentDevice(),
                      in.sizes(),
                      stream));
  } else {
    out = std::move(DeviceTensor<float, Dim, true>(in.sizes()));
  }

    fromInt8<Dim>(stream, in, out);
  return out;
}

int8_t hostFloat2Int8(float v);

#endif // FAISS_USE_FLOAT16

} } // namespace
