/**
 * Copyright (c) 2015-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD+Patents license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Copyright 2004-present Facebook. All Rights Reserved.

#pragma once

#include "../../FaissAssert.h"
#include "DeviceUtils.h"
#include "MemorySpace.h"
#include "StaticUtils.h"
#include <algorithm>
#include <cuda.h>
#include <vector>
#include <math_functions.h>
#include "../../Index.h"

namespace faiss { namespace gpu {

/// A simple version of thrust::device_vector<T>, but has more control
/// over whether resize() initializes new space with T() (which we
/// don't want), and control on how much the reserved space grows by
/// upon resize/reserve. It is also meant for POD types only.
template <typename T>
class DeviceVector {
 public:
  DeviceVector(MemorySpace space = MemorySpace::Device)
      : data_(nullptr),
        num_(0),
        capacity_(0),
        max_size_(SIZE_MAX),
        space_(space),
        error_(Faiss_Error_OK) {
  }

  ~DeviceVector() {
    clear();
  }

  // Clear all allocated memory; reset to zero size
  void clear() {
    CUDA_VERIFY(cudaFree(data_));
    data_ = nullptr;
    num_ = 0;
    capacity_ = 0;
  }

  void set_max_size(size_t ms){max_size_ = ms;}
  size_t get_max_size() const {return max_size_;}
  size_t size() const { return num_; }
  size_t capacity() const { return capacity_; }
  T* data() { return data_; }
  const T* data() const { return data_; }

  template <typename OutT>
  std::vector<OutT> copyToHost(cudaStream_t stream) const {
    FAISS_ASSERT(num_ * sizeof(T) % sizeof(OutT) == 0);

    std::vector<OutT> out((num_ * sizeof(T)) / sizeof(OutT));
    CUDA_VERIFY(cudaMemcpyAsync(out.data(), data_, num_ * sizeof(T),
                                cudaMemcpyDeviceToHost, stream));
    CUDA_VERIFY(cudaDeviceSynchronize());

    return out;
  }

  // Returns true if we actually reallocated memory
  // If `reserveExact` is true, then we reserve only the memory that
  // we need for what we're appending
  bool append(const T* d,
              size_t n,
              cudaStream_t stream,
              bool reserveExact = false) {
    bool mem = false;

    if (n > 0) {
      size_t reserveSize = num_ + n;
      if (!reserveExact) {
        mem = getNewCapacitySmartAndReserve_(reserveSize,stream);
      }else{
        mem = reserve(reserveSize, stream);
      }

      if (error_ != Faiss_Error_OK)
      {
          return false;
      }

      int dev = getDeviceForAddress(d);
      if (dev == -1) {
        CUDA_VERIFY(cudaMemcpyAsync(data_ + num_, d, n * sizeof(T),
                                    cudaMemcpyHostToDevice, stream));
      } else {
        CUDA_VERIFY(cudaMemcpyAsync(data_ + num_, d, n * sizeof(T),
                                    cudaMemcpyDeviceToDevice, stream));
      }
      num_ += n;
    }

    return mem;
  }


  // Returns true if we actually reallocated memory
  bool resize(size_t newSize, cudaStream_t stream) {
    bool mem = false;

    if (num_ < newSize) {
      mem = reserve(getNewCapacity_(newSize), stream);
    } else if (num_ > newSize)
    {
        num_ = newSize;
        deallocateMemory(stream);
    }

    // Don't bother zero initializing the newly accessible memory
    // (unlike thrust::device_vector)
    num_ = newSize;

    return mem;
  }

  // Clean up after oversized allocations, while leaving some space to
  // remain for subsequent allocations (if `exact` false) or to
  // exactly the space we need (if `exact` true); returns space
  // reclaimed in bytes
  size_t reclaim(bool exact, cudaStream_t stream) {
    size_t free = capacity_ - num_;

    if (exact) {
      realloc_(num_, stream);
      return free * sizeof(T);
    }

    // If more than 1/4th of the space is free, then we want to
    // truncate to only having 1/8th of the space free; this still
    // preserves some space for new elements, but won't force us to
    // double our size right away
    if (free > (capacity_ / 4)) {
      size_t newFree = capacity_ / 8;
      size_t newCapacity = num_ + newFree;

      size_t oldCapacity = capacity_;
      FAISS_ASSERT(newCapacity < oldCapacity);

      realloc_(newCapacity, stream);

      return (oldCapacity - newCapacity) * sizeof(T);
    }

    return 0;
  }

  // Returns true if we actually reallocated memory
  bool reserve(size_t newCapacity, cudaStream_t stream) {
    if (newCapacity <= capacity_) {
      return false;
    }

    if (newCapacity > max_size_)
    {
        error_ = Faiss_Error_Exceed_Max_Size;
        return false;
    }

    // Otherwise, we need new space.
    // realloc_(newCapacity, stream);
    allocateMemory(newCapacity, stream);
    return true;
  }

  ErrorTypes error() {
      ErrorTypes ret = error_;
      error_ = Faiss_Error_OK;
      return ret;
  }

 private:
    //      在扩张过程中，如果 (cap - size) < left &&  size > left ,数据会先拷贝到内存中，析构显存，再拷贝回内存；
//      如果(cap - size) > left &&  size > left ;则该库不能再被add
    //    {
//      uint64_t  left_mem_t = 0;
//      uint64_t  total_mem_t = 0;
////      int device_id = getDeviceForAddress(data_);
//      CUDA_VERIFY(cudaMemGetInfo(&left_mem_t,&total_mem_t));
//      if(newCapacity){
//
//      }
//    }
  bool realloc_(size_t newCapacity, cudaStream_t stream) {
    FAISS_ASSERT(num_ <= newCapacity);
    T* newData = nullptr;
    if (!allocMemorySpace(space_, (void**) &newData, newCapacity * sizeof(T)))
    {
      error_ = Faiss_Error_Alloc_Fail;
      return false;
    }
    CUDA_VERIFY(cudaMemcpyAsync(newData, data_, num_ * sizeof(T),
                                cudaMemcpyDeviceToDevice, stream));
    // FIXME: keep on reclamation queue to avoid hammering cudaFree?
    CUDA_VERIFY(cudaFree(data_));

    data_ = newData;
    capacity_ = newCapacity;
    return true;
  }

  void allocateMemory(size_t requiredSpace, cudaStream_t stream)
  {
      size_t needAllocSpace = calculateSpace(requiredSpace);
      if (needAllocSpace > max_size_)
      {
          needAllocSpace = max_size_;
      }

      size_t freeSpace = 0;
      size_t totalSpace = 0;
      CUDA_VERIFY(cudaMemGetInfo(&freeSpace, &totalSpace));

      if (freeSpace * 95 / 100 >= needAllocSpace)
      {
          realloc_(needAllocSpace, stream);
      } else if (capacity_ + freeSpace >= needAllocSpace)
      {
          std::vector<char> buffer = copyToHost<char>(stream);
          clear();
          if (realloc_(needAllocSpace, stream))
          {
              cudaMemcpyAsync(data_, buffer.data(), buffer.size(), cudaMemcpyHostToDevice, stream);
              CUDA_VERIFY(cudaDeviceSynchronize());
          } else
          {
              if (realloc_(buffer.size(), stream))
              {
                  cudaMemcpyAsync(data_, buffer.data(), buffer.size(), cudaMemcpyHostToDevice, stream);
                  CUDA_VERIFY(cudaDeviceSynchronize());
              } else
              {
                  error_ = Faiss_Error_Lost_Index_Data;
              }
          }
      } else
      {
          error_ = Faiss_Error_Run_Out_Of_Mem;
      }
  }

  size_t calculateSpace(size_t memSpace)
  {
      const size_t VECTOR_DIM = 384;
      const size_t K_VECTORS = (1 << 10) * VECTOR_DIM;
      const size_t M_VECTORS = (1 << 20) * VECTOR_DIM;
      const size_t G_VECTORS = (1 << 30) * VECTOR_DIM;
      size_t space = 0;
      if (memSpace < K_VECTORS)
      {
          return (size_t) K_VECTORS;
      } else if (memSpace < M_VECTORS)
      {
          space = K_VECTORS;
          while (space <= memSpace)
          {
              space <<= 1;
          }
      } else if (memSpace < G_VECTORS)
      {
          space = M_VECTORS;
          while (space <= memSpace)
          {
              space += M_VECTORS;
          }
      } else
      {
          space = G_VECTORS;
          while (space <= memSpace)
          {
              space += 5 * M_VECTORS;
          }
      }
      return space;
  }

  void deallocateMemory(cudaStream_t stream)
  {
      size_t newCapacity = 0;
      if (!deallocateSpace(newCapacity))
      {
          return;
      }

      if (newCapacity == 0)
      {
          clear();
          return;
      }

      size_t freeSpace = 0;
      size_t totalSpace = 0;
      CUDA_VERIFY(cudaMemGetInfo(&freeSpace, &totalSpace));

      if (freeSpace * 95 / 100 >= newCapacity)
      {
          realloc_(newCapacity, stream);
          return;
      } else if (freeSpace + capacity_ >= newCapacity)
      {
          std::vector<char> buffer = copyToHost<char>(stream);
          clear();
          if (realloc_(newCapacity, stream))
          {
              cudaMemcpy((char*) data_, buffer.data(), buffer.size(), cudaMemcpyHostToDevice);
              CUDA_VERIFY(cudaDeviceSynchronize());
          } else
          {
              error_ = Faiss_Error_Lost_Index_Data;
          }
          return;
      }
      error_ = Faiss_Error_Unknown;
  }

  bool deallocateSpace(size_t &newCapacity)
  {
      const size_t VECTOR_DIM = 384;
      const size_t K_VECTORS = (1 << 10) * VECTOR_DIM;
      const size_t M_VECTORS = (1 << 20) * VECTOR_DIM;
      const size_t G_VECTORS = (1 << 30) * VECTOR_DIM;

      if (num_ >= G_VECTORS && capacity_ - num_ > 10 * M_VECTORS)
      {
          newCapacity = capacity_ - 5 * M_VECTORS;
          return true;
      }
      else if (num_ >= M_VECTORS && num_ < G_VECTORS && capacity_ - num_ > 2 * M_VECTORS)
      {
          newCapacity = capacity_ - M_VECTORS;
          return true;
      } else if (num_ >= K_VECTORS && num_ < M_VECTORS && num_ < capacity_ / 4)
      {
          newCapacity = capacity_ / 2;
          return true;
      } else if (num_ == 0)
      {
          newCapacity = 0;
          return true;
      }
      return false;
  }

  void reallocTemoInHost_(size_t newCapacity, cudaStream_t stream) {
    FAISS_ASSERT(num_ <= newCapacity);
    T* tmpHostData = nullptr;
    CUDA_VERIFY(cudaMallocHost((void**) &tmpHostData, num_ * sizeof(T)));
    CUDA_VERIFY(cudaMemcpyAsync(tmpHostData, data_, num_ * sizeof(T),
                                cudaMemcpyDeviceToHost, stream));
    CUDA_VERIFY(cudaFree(data_));

    T* newData = nullptr;
    allocMemorySpace(space_, (void**) &newData, newCapacity * sizeof(T));
    CUDA_VERIFY(cudaMemcpyAsync(newData, tmpHostData, num_ * sizeof(T),
                                cudaMemcpyHostToDevice, stream));

    data_ = newData;
    capacity_ = newCapacity;
  }

  size_t getNewCapacity_(size_t preferredSize) {
    return utils::nextHighestPowerOf2(preferredSize);
  }
//      -  (size<2^10) 第一次一次性增加到最大
//      - （size<2^20）当库中特征数量较少时，按照2的次幂增加
//      -  (size>2^20) 每次增加2^20;any add time: size > left 扩充到cap
//
//      在扩张过程中，如果 (cap - size) < left &&  size > left ,数据会先拷贝到内存中，析构显存，再拷贝回内存；
//      如果(cap - size) > left &&  size > left ;则该库不能再被add
        //TODO:not impl
  bool getNewCapacitySmartAndReserve_(size_t preferredSize, cudaStream_t stream) {
    //num_ + n;
        FAISS_ASSERT("not impl");
    const int small_size = (1<<10);
    const int mid_size = (1<<20);
    auto delta = preferredSize - num_;
    int new_size = preferredSize;

    if(delta<0){
      new_size =  getNewCapacity_(preferredSize);
    }else{
      if(preferredSize<= small_size ){
        new_size =  small_size;
      }else if(preferredSize <= mid_size ){
        new_size =  utils::nextHighestPowerOf2(preferredSize);
      }else{
        int s = num_ + mid_size;
        while(s<preferredSize){
          s+=mid_size;
        }
        new_size = s;
      }
    }
    preferredSize =  min((unsigned long) max_size_,(unsigned long)new_size);

  }

  T* data_;
  size_t num_;
  size_t capacity_;
  size_t max_size_;
  MemorySpace space_;
  ErrorTypes error_;
};

} } // namespace
