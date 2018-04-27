/**
 * Copyright (c) 2015-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD+Patents license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Copyright 2004-present Facebook. All Rights Reserved.

#pragma once


#include <iostream>

#include <thrust/inner_product.h>
#include <thrust/execution_policy.h>

#include "../utils/Float16.cuh"
#include "../utils/Tensor.cuh"

namespace faiss { namespace gpu {

void runL2Norm(Tensor<float, 2, true>& input,
               Tensor<float, 1, true>& output,
               bool normSquared,
               cudaStream_t stream);

#ifdef FAISS_USE_FLOAT16
void runL2Norm(Tensor<half, 2, true>& input,
               Tensor<half, 1, true>& output,
               bool normSquared,
               cudaStream_t stream);


template <typename T>
void runL2Norm(Tensor<T, 2, true>& input, Tensor<float, 1, true>& output, bool normSquard)
{
    int rows = output.getSize(0);
    std::vector<float> output_buff(rows, 0.0f);
    for (int i = 0; i < rows; i++)
    {
        output_buff[i] = thrust::inner_product(thrust::device, &input[i][0], &input[i + 1][0], &input[i][0], 0.0f);
        output_buff[i] = 1.0f / output_buff[i];
    }
    if (normSquard)
    {
        for (auto &item : output_buff)
        {
            item = sqrt(item);
        }
    }
    cudaMemcpy(output.data(), output_buff.data(), sizeof(float) * rows, cudaMemcpyDefault);

//    std::cout << "\nNorms: ";
//    for (auto item : output_buff)
//    {
//        std::cout << item << " ";
//    }
//    std::cout << "\n";
}
#endif

} } // namespace
