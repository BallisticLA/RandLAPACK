#ifndef MACROS_HH
#define MACROS_HH
#include <cstdint>
#include <iostream>

#define BPCG_STATUS(_msg)                                                \
    std::cout << "STATUS: " _msg << std::endl;

#define BPCG_ERROR(_msg)                                                \
    std::cerr << "ERROR: " << __FILE__ << ":" << __LINE__ << std::endl  \
        << "" _msg << std::endl;

#define BPCG_FATAL_ERROR(_msg)                                          \
    std::cerr << "ERROR: " << __FILE__ << ":" << __LINE__ << std::endl  \
        << "" _msg << std::endl;                                        \
    abort();

#define check_cuda_error                                                \
{                                                                       \
    cudaDeviceSynchronize();                                            \
    cudaError_t ierr = cudaGetLastError();                              \
    if (ierr != cudaSuccess)                                            \
    {                                                                   \
        BPCG_ERROR("CUDA error : " << cudaGetErrorString( ierr ));      \
        abort();                                                        \
    }                                                                   \
}
#endif