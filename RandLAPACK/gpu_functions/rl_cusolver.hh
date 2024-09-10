#pragma once

#if defined(ENABLE_CUDA)
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include "cublas_traits.h"
#else
//
// We don't have access to cuBLAS / cuSOLVER.
//
// We need to create dummy aliases for CUDA types.
//
using cublasHandle_t = void *;
using cusolverDnHandle_t = void *;
using cublasFillMode_t = char;
//
// We have code that calls certain CUDA-defined functions
// regardless of whether USE_CUBLAS was defined.
//
// We need to define dummy versions of these functions.
// 
// Naturally, these dummy versions should raise an error
// if called. Here is some context that informs how we handle this
//
//     It so happens that the CUDA functions in question all return
//     values from a CUDA-defined enum to indicate if the operation
//     succeeded or failed.
// 
//     Our code that calls these CUDA functions is supposed to raise
//     an error if the CUDA function returned anything other than zero.
//
// Therefore our dummy versions of these functions only need to
// return a value other than zero in order to raise errors where 
// they're invoked.
// 
// We have them return 1, since this corresponds to the value of
// CUSOLVER_STATUS_NOT_INITIALIZED in the cusolverStatus_t enum.
// 
int cusolverDnCreate(cusolverDnHandle_t *x) {
    return 1;
}
int cusolverDnDestroy(cusolverDnHandle_t x) {
    return 1;
}
int cublasCreate(cublasHandle_t *x) {
    return 1;
}
int cublasDestroy(cublasHandle_t x) {
    return 1;
}
// 
// For good measure, we define macros that map to the success condition
// for the CUDA functions we might call.
//
#define CUSOLVER_STATUS_SUCCESS 0
#define CUBLAS_STATUS_SUCCESS 0
#endif
