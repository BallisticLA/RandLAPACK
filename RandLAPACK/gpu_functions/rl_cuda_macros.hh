#pragma once
#include <cstdint>
#include <iostream>

// Status message (for debugging/logging)
#define RandLAPACK_CUDA_STATUS(_msg)                                                \
    std::cout << "STATUS: " _msg << std::endl;

// Error message (non-fatal)
#define RandLAPACK_CUDA_ERROR(_msg)                                                \
    std::cerr << "ERROR: " << __FILE__ << ":" << __LINE__ << std::endl  \
        << "" _msg << std::endl;

// Fatal error message (aborts)
#define RandLAPACK_CUDA_FATAL_ERROR(_msg)                                          \
    std::cerr << "ERROR: " << __FILE__ << ":" << __LINE__ << std::endl  \
        << "" _msg << std::endl;                                        \
    abort();

// Check for CUDA errors with device synchronization (use sparingly - expensive!)
#define check_cuda_error                                                \
{                                                                       \
    cudaDeviceSynchronize();                                            \
    cudaError_t ierr = cudaGetLastError();                              \
    if (ierr != cudaSuccess)                                            \
    {                                                                   \
        RandLAPACK_CUDA_ERROR("CUDA error: " << cudaGetErrorString( ierr ));      \
        abort();                                                        \
    }                                                                   \
}

// Check for CUDA errors WITHOUT synchronization (fast check after kernel launches)
// Use this after kernel launches instead of check_cuda_error for better performance
#define check_cuda_error_nosync(_context_msg)                           \
{                                                                       \
    cudaError_t ierr = cudaGetLastError();                              \
    if (ierr != cudaSuccess)                                            \
    {                                                                   \
        RandLAPACK_CUDA_ERROR(_context_msg << " " << cudaGetErrorString(ierr));   \
        abort();                                                        \
    }                                                                   \
}
