#pragma once
#include <cstdint>
#include <iostream>

#define RandLAPACK_CUDA_STATUS(_msg)                                                \
    std::cout << "STATUS: " _msg << std::endl;

#define RandLAPACK_CUDA_ERROR(_msg)                                                \
    std::cerr << "ERROR: " << __FILE__ << ":" << __LINE__ << std::endl  \
        << "" _msg << std::endl;

#define RandLAPACK_CUDA_FATAL_ERROR(_msg)                                          \
    std::cerr << "ERROR: " << __FILE__ << ":" << __LINE__ << std::endl  \
        << "" _msg << std::endl;                                        \
    abort();

#define check_cuda_error                                                \
{                                                                       \
    cudaDeviceSynchronize();                                            \
    cudaError_t ierr = cudaGetLastError();                              \
    if (ierr != cudaSuccess)                                            \
    {                                                                   \
        RandLAPACK_CUDA_ERROR("CUDA error : " << cudaGetErrorString( ierr ));      \
        abort();                                                        \
    }                                                                   \
}
