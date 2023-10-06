#include "MYLNODE1.cuh"
#include "MYLNODE2.cuh"
#include "MYLNODE3.cuh"


__forceinline__ __device__ float my_sigmoid(const float x)
{
    return 1.0f/(1+exp(-x));
}

__global__ void mylnode1_kernel(const float *x, const float *addNum, const float *mutNum, float *output, int n)
{
   int idx = threadIdx.x + blockDim.x*blockIdx.x;
   if(idx < n)
   {
        output[idx] = my_sigmoid(x[idx]);
        int select_id =  idx%85;
        if(select_id < 2)
        {
            output[idx] = output[idx]*2 - 0.5 + addNum[(idx/85)*2+select_id];
            output[idx] = output[idx]*8;
        }
        if(select_id >= 2 && select_id < 4)
        {
            output[idx] = output[idx]*2;
            output[idx] = output[idx]*output[idx];
            output[idx] = output[idx]*mutNum[(idx/(85*80*80))*2+select_id%2];
        }
   }
}

__global__ void mylnode2_kernel(const float *x, const float *addNum, const float *mutNum, float *output, int n)
{
   int idx = threadIdx.x + blockDim.x*blockIdx.x;
   if(idx < n)
   {
        output[idx] = my_sigmoid(x[idx]);
        int select_id =  idx%85;
        if(select_id < 2)
        {
            output[idx] = output[idx]*2 - 0.5 + addNum[(idx/85)*2+select_id];
            output[idx] = output[idx]*16;
        }
        if(select_id >= 2 && select_id < 4)
        {
            output[idx] = output[idx]*2;
            output[idx] = output[idx]*output[idx];
            output[idx] = output[idx]*mutNum[(idx/(85*40*40))*2+select_id%2];
        }
   }
}

__global__ void mylnode3_kernel(const float *x, const float *addNum, const float *mutNum, float *output, int n)
{
   int idx = threadIdx.x + blockDim.x*blockIdx.x;
   if(idx < n)
   {
        output[idx] = my_sigmoid(x[idx]);
        int select_id =  idx%85;
        if(select_id < 2)
        {
            output[idx] = output[idx]*2 - 0.5 + addNum[(idx/85)*2+select_id];
            output[idx] = output[idx]*32;
        }
        if(select_id >= 2 && select_id < 4)
        {
            output[idx] = output[idx]*2;
            output[idx] = output[idx]*output[idx];
            output[idx] = output[idx]*mutNum[(idx/(85*20*20))*2+select_id%2];
        }
   }
}


static __global__ void printer_val(const float *x, int n)
{
    int idx = threadIdx.x + blockDim.x*blockIdx.x;
    if(idx == 0)
    {
        for (size_t i = 0; i < n; i++)
        {
            printf("%f ", x[i]);
        }
        
    }
}


void MYLNODE1_inference(const float *x, const float *addNum, const float *mutNum, float *output, int n, cudaStream_t stream)
{
    const int nthreads = 512;
    int grid_size = (n+nthreads-1)/nthreads;
    mylnode1_kernel<<<grid_size, nthreads>>>(x, addNum, mutNum, output, n);
}

void MYLNODE2_inference(const float *x, const float *addNum, const float *mutNum, float *output, int n, cudaStream_t stream)
{
    const int nthreads = 512;
    int grid_size = (n+nthreads-1)/nthreads;
    mylnode2_kernel<<<grid_size, nthreads>>>(x, addNum, mutNum, output, n);
}

void MYLNODE3_inference(const float *x, const float *addNum, const float *mutNum, float *output, int n, cudaStream_t stream)
{
    const int nthreads = 512;
    int grid_size = (n+nthreads-1)/nthreads;
    mylnode3_kernel<<<grid_size, nthreads>>>(x, addNum, mutNum, output, n);
}

void myprint_val(const float *x, int n, cudaStream_t stream)
{
    printer_val<<<1, 1, 0, stream>>>(x, n);
}