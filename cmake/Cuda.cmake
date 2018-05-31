# configure cuda

find_package(CUDA QUIET REQUIRED)
if(CUDA_FOUND)
    include_directories(SYSTEM ${CUDA_INCLUDE_DIRS})
    list(APPEND CUDA_LINKER_LIBS ${CUDA_CUDART_LIBRARY} ${CUDA_curand_LIBRARY} ${CUDA_CUBLAS_LIBRARIES})
else(CUDA_FOUND)
    message(STATUS "Could not locate cuda, disabling cuda support.")
    set(BUILD_WITH_GPU OFF)
    return()
endif(CUDA_FOUND)

# set cuda flags
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    list(APPEND CUDA_NVCC_FLAGS "-arch=sm_61 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_32,code=sm_32 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_37,code=sm_37 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_53,code=sm_53 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_61,code=compute_61
;-std=c++11;-DVERBOSE;-g;-lineinfo;-Xcompiler;-ggdb")
else()
    list(APPEND CUDA_NVCC_FLAGS "-arch=sm_61 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_32,code=sm_32 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_37,code=sm_37 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_53,code=sm_53 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_61,code=compute_61
;-std=c++11;-DVERBOSE;-O3;-DNDEBUG;-Xcompiler;-DNDEBU")
endif()
set(CUDA_PROPAGATE_HOST_FLAGS OFF)
