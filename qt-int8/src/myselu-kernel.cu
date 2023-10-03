#include <cuda_runtime.h>
#include <math.h>

static __device__ float sigmoid(float x)
{
    return 1 / (1 + expf(-x));
}

static __global__ void myselu_kernel(const float *x, float *output, int n)
{
    int position = threadIdx.x + blockDim.x*blockIdx.x;

    if(position >= n)
        return;

    output[position] = x[position]*sigmoid(x[position]);
}

void myselu_inference(const float *x, float *output, int n, cudaStream_t stream)
{
    const int nthreads = 512;
    int grid_size = (n+nthreads-1)/nthreads;
    myselu_kernel<<<grid_size, nthreads, 0, stream>>>(x, output, n);
}


