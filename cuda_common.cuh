#ifndef CUDA_CMN_H
#define CUDA_CMN_H

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

#include <stdlib.h>
#include <iostream>

#define gpuErrchk(ans){gpuAssert(ans,__FILE__,__LINE__);}

inline void gpuAssert(cudaError_t code,
const char *file ,int line, bool abort = true)
{
    if(code!=cudaSuccess)
    {
        fprintf(stderr,"GPUassert: %s %d %s %d\n",
        cudaGetErrorString(code),static_cast<unsigned int>(code),file,line);
        if(abort) exit(code);
    }
}

#endif

