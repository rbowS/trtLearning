#include "img-process.cuh"

namespace IMGPRrocess{
    

    __global__ void decode_kernel(float *in, float *out, float *d2i, const int numbox, 
                const int numprob, const int num_classes, const float confidence_threshold)
    {
        const int idx = threadIdx.x + blockDim.x*blockIdx.x;
        const int tid = threadIdx.x;
        __shared__ float3 smem;
        if(tid == 0)
        {
            smem.x = d2i[0];
            smem.y = d2i[2];
            smem.z = d2i[5];
        }
        __syncthreads();
        if(idx < numbox)
        {
            float* ptr = in + idx * numprob;
            float objness = ptr[4];
            //bad writting, may cause warp diverge
            if(objness > confidence_threshold)
            {
                float* pclass = ptr + 5;
                float prob = pclass[0];
                float label = 0;
                for (int i = 1; i < num_classes; i++)
                {
                    if(prob < pclass[i])
                    {
                        prob = pclass[i];
                        label = i;
                    }
                }
                
                float confidence = prob * objness;
                if(confidence > confidence_threshold)
                {
                    // 中心点、宽、高
                    float cx     = ptr[0];
                    float cy     = ptr[1];
                    float width  = ptr[2];
                    float height = ptr[3];

                    // 预测框
                    float left   = cx - width * 0.5;
                    float top    = cy - height * 0.5;
                    float right  = cx + width * 0.5;
                    float bottom = cy + height * 0.5;

                    // 对应图上的位置
                    float image_base_left   = smem.x * left   + smem.y;
                    float image_base_right  = smem.x * right  + smem.y;
                    float image_base_top    = smem.x * top    + smem.z;
                    float image_base_bottom = smem.x * bottom + smem.z;

                    out[idx*6]   = image_base_left;
                    out[idx*6+1] = image_base_top;
                    out[idx*6+2] = image_base_right;
                    out[idx*6+3] = image_base_bottom;
                    out[idx*6+4] = label;
                    out[idx*6+5] = confidence;
                }
            }
        }
    }

    std::vector<std::vector<float>> decode_kernel_invoker(float *d_in, float *d2i, 
                    const int numbox, const int numprob, const int num_classes, 
                    const float confidence_threshold, cudaStream_t stream){
        
        std::vector<std::vector<float>> bboxes;
        const int blockSize = 256;
        const int gridSize = (numbox + blockSize -1)/blockSize;

        float *d_out;
        float *d_d2i;
        float *h_out = new float[numbox*6];
        checkRuntime( cudaMalloc((void**)&d_out, numbox*6*sizeof(float)) );
        checkRuntime( cudaMalloc((void**)&d_d2i, 6*sizeof(float)) );
        checkRuntime( cudaMemcpy(d_d2i, d2i, sizeof(float)*6, cudaMemcpyHostToDevice) );
        decode_kernel<<<gridSize, blockSize, 0, stream>>>(d_in, d_out, d_d2i,numbox, 
                        numprob, num_classes, confidence_threshold);
        cudaDeviceSynchronize();
        checkRuntime( cudaMemcpy(h_out, d_out, numbox*6*sizeof(float), cudaMemcpyDeviceToHost) );
        for (int i = 0; i < numbox; i++)
        {
            if(h_out[i*6+5] != 0.0f)
            {
                bboxes.push_back({h_out[i*6], h_out[i*6+1], h_out[i*6+2], 
                                  h_out[i*6+3],h_out[i*6+4], h_out[i*6+5]});
            }
        }
        delete[] h_out;
        cudaFree(d_out);
        cudaFree(d_d2i);
        return bboxes;
    }

}