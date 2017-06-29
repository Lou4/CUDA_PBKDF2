#include <sys/time.h>
#include <stdio.h>
#include <stdlib.h>

#ifndef _COMMON_H
#define _COMMON_H

#define CHECK(call)                                                            \
{                                                                              \
    const cudaError_t error = call;                                            \
    if (error != cudaSuccess)                                                  \
    {                                                                          \
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);                 \
        fprintf(stderr, "code: %d, reason: %s\n", error,                       \
                cudaGetErrorString(error));                                    \
    }                                                                          \
}

#define CHECK_CUBLAS(call)                                                     \
{                                                                              \
    cublasStatus_t err;                                                        \
    if ((err = (call)) != CUBLAS_STATUS_SUCCESS)                               \
    {                                                                          \
        fprintf(stderr, "Got CUBLAS error %d at %s:%d\n", err, __FILE__,       \
                __LINE__);                                                     \
        exit(1);                                                               \
    }                                                                          \
}

#define CHECK_CURAND(call)                                                     \
{                                                                              \
    curandStatus_t err;                                                        \
    if ((err = (call)) != CURAND_STATUS_SUCCESS)                               \
    {                                                                          \
        fprintf(stderr, "Got CURAND error %d at %s:%d\n", err, __FILE__,       \
                __LINE__);                                                     \
        exit(1);                                                               \
    }                                                                          \
}

#define CHECK_CUFFT(call)                                                      \
{                                                                              \
    cufftResult err;                                                           \
    if ( (err = (call)) != CUFFT_SUCCESS)                                      \
    {                                                                          \
        fprintf(stderr, "Got CUFFT error %d at %s:%d\n", err, __FILE__,        \
                __LINE__);                                                     \
        exit(1);                                                               \
    }                                                                          \
}

#define CHECK_CUSPARSE(call)                                                   \
{                                                                              \
    cusparseStatus_t err;                                                      \
    if ((err = (call)) != CUSPARSE_STATUS_SUCCESS)                             \
    {                                                                          \
        fprintf(stderr, "Got error %d at %s:%d\n", err, __FILE__, __LINE__);   \
        cudaError_t cuda_err = cudaGetLastError();                             \
        if (cuda_err != cudaSuccess)                                           \
        {                                                                      \
            fprintf(stderr, "  CUDA error \"%s\" also detected\n",             \
                    cudaGetErrorString(cuda_err));                             \
        }                                                                      \
        exit(1);                                                               \
    }                                                                          \
}

inline void device_name() {
	int dev = 0;
	cudaDeviceProp deviceProp;
	CHECK(cudaGetDeviceProperties(&deviceProp, dev));
	printf("device %d: %s\n", dev, deviceProp.name);
	printf("device %d: warp size in thread %d\n", dev, deviceProp.warpSize);
	printf("device %d: total global mem %f GB\n", dev, ((float)deviceProp.totalGlobalMem/1024.0/1024.0/1024.0));
	printf("device %d: max thread per block %d\n", dev, deviceProp.maxThreadsPerBlock);
	printf("device %d: max size dim block x %d, max size dim block y %d, max size dim block z %d\n", dev, deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2]);
	printf("device %d: max size dim grid x %d, max size dim grid y %d, max size dim grid z %d\n", dev, deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);
	printf("\n");
	CHECK(cudaSetDevice(dev));
}

inline void device_name(int dev) {
	cudaDeviceProp deviceProp;
	CHECK(cudaGetDeviceProperties(&deviceProp, dev));
	printf("device %d: %s\n", dev, deviceProp.name);
	printf("device %d: warp size in thread %d\n", dev, deviceProp.warpSize);
	printf("device %d: total global mem %f GB\n", dev, ((float)deviceProp.totalGlobalMem/1024.0/1024.0/1024.0));
	printf("device %d: max thread per block %d\n", dev, deviceProp.maxThreadsPerBlock);
	printf("device %d: max size dim block x %d, max size dim block y %d, max size dim block z %d\n", dev, deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2]);
	printf("device %d: max size dim grid x %d, max size dim grid y %d, max size dim grid z %d\n", dev, deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);
	printf("\n");
	CHECK(cudaSetDevice(dev));
}

#endif // _COMMON_H
