#ifndef CUDA_TOOLS_DATAAUG
#define CUDA_TOOLS_DATAAUG

#include "mix-memory.cuh"
#include "cuda-tools.cuh"
#include <vector>

namespace IMGPRrocess{
    

    __global__ void decode_kernel(float *in, float *out, float *d2i, const int numbox, 
                const int numprob, const int num_classes, const float confidence_threshold);

    std::vector<std::vector<float>> decode_kernel_invoker(float *d_in, float *d2i, 
                    const int numbox, const int numprob, const int num_classes, 
                    const float confidence_threshold, cudaStream_t stream);

}

#endif