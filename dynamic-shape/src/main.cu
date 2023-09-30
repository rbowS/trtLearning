#include <NvInfer.h>
#include <NvInferRuntime.h>

#include <cuda_runtime.h>

#include <stdio.h>
#include <math.h>

#include <iostream>
#include <fstream>
#include <vector>
#include <unistd.h>

using std::vector;
using std::string;
using std::ifstream;
using std::ios;

class TRTLogger : public nvinfer1::ILogger{

public:
    virtual void log(Severity severity, nvinfer1::AsciiChar const *msg)
    noexcept override
    {
        if(severity <= Severity::kINFO)
        {
            printf("%d: %s\n", severity, msg);
        }
    }

};


nvinfer1::Weights make_weights(float *ptr, int n)
{
    nvinfer1::Weights w;
    w.count = n;
    w.type = nvinfer1::DataType::kFLOAT;
    w.values = ptr;
    return w;
}

bool build_model()
{
    TRTLogger logger;

    // 这是基本需要的组件
    nvinfer1::IBuilder *builder = nvinfer1::createInferBuilder(logger);
    nvinfer1::IBuilderConfig *config = builder->createBuilderConfig();
    nvinfer1::INetworkDefinition *network = builder->createNetworkV2(1);

    // 构建一个模型
    /*
        Network definition:

        image
          |
        conv(3x3, pad=1)  input = 1, output = 1, bias = True     w=[[1.0, 2.0, 0.5], [0.1, 0.2, 0.5], [0.2, 0.2, 0.1]], b=0.0
          |
        relu
          |
        prob
    */
    const int num_input = 1;
    const int num_output = 1;
    float layer1_weight_values[] = {
        1.0, 2.0, 3.1, 
        0.1, 0.1, 0.1,
        0.2, 0.2, 0.2};
    float layer1_bias_values[] = {0.0};

    nvinfer1::ITensor *input = network->addInput("image", nvinfer1::DataType::kFLOAT, nvinfer1::Dims4(-1, num_input, -1, -1));
    nvinfer1::Weights layer1_weight = make_weights(layer1_weight_values, 9);
    nvinfer1::Weights layer1_bias = make_weights(layer1_bias_values, 1);

    auto layer1 = network->addConvolution(*input, num_output, nvinfer1::DimsHW(3, 3), layer1_weight, layer1_bias);
    layer1->setPadding(nvinfer1::DimsHW(1,1));
    
    auto prob = network->addActivation(*layer1->getOutput(0), nvinfer1::ActivationType::kRELU);

    // 将我们需要的prob标记为输出
    network->markOutput(*prob->getOutput(0));
    
    int maxBatchSize = 10;

    printf("Workspace Size = %.2f MB\n", (1 << 28) / 1024.0f / 1024.0f);

    config->setMaxWorkspaceSize(1<<28);

    // --------------------------------- 2.1 关于profile ----------------------------------
    auto profile = builder->createOptimizationProfile();
    profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims4(1, num_input, 3, 3));
    profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims4(1, num_input, 3, 3));

    profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims4(maxBatchSize, num_input, 5, 5));
    config->addOptimizationProfile(profile);

    nvinfer1::ICudaEngine *engine = builder->buildEngineWithConfig(*network, *config);
    if(engine == nullptr)
    {
        printf("Build engine failed.\n");
        return false;
    }

    // 将模型序列化，并储存为文件
    nvinfer1::IHostMemory *model_data = engine->serialize();
    FILE *f = fopen("engine.trtmodel", "wb");
    fwrite(model_data->data(), 1, model_data->size(), f);
    fclose(f);

    delete model_data;
    delete engine;
    delete network;
    delete config;
    delete builder;

    printf("Done\n");
    return true;
}


vector<uint8_t> load_file(const string &file)
{
    ifstream in(file, ios::in | ios::binary);
    if(!in.is_open())
        return {};
    in.seekg(0, ios::end);
    size_t length = in.tellg();
    vector<uint8_t> data;
    if(length > 0)
    {
        in.seekg(0, ios::beg);
        data.resize(length);
        in.read(reinterpret_cast<char*>(&data[0]), length);
    }
    in.close();
    return data;
}



void inference(const string &file)
{
    TRTLogger logger;
    auto engine_data = load_file(file);
    // 执行推理前，需要创建一个推理的runtime接口实例。与builer一样，runtime需要logger：
    nvinfer1::IRuntime *runtime = nvinfer1::createInferRuntime(logger);
    // 将模型从读取到engine_data中，则可以对其进行反序列化以获得engine
    nvinfer1::ICudaEngine *engine = runtime->deserializeCudaEngine(engine_data.data(), engine_data.size());
    if(engine == nullptr){
        printf("Deserialize cuda engine failed.\n");
        delete runtime;
        return;
    }
    nvinfer1::IExecutionContext *execution_context = engine->createExecutionContext();
    // 创建CUDA流，以确定这个batch的推理是独立的
    cudaStream_t stream = nullptr;
    cudaStreamCreate(&stream);

    /*
        Network definition:

        image
          |
        conv(3x3, pad=1)  input = 1, output = 1, bias = True     w=[[1.0, 2.0, 0.5], [0.1, 0.2, 0.5], [0.2, 0.2, 0.1]], b=0.0
          |
        relu
          |
        prob
    */
    float input_data_host[] = {
        // batch 0
        1,   1,   1,
        1,   1,   1,
        1,   1,   1,

        // batch 1
        -1,   1,   1,
        1,   0,   1,
        1,   1,   -1
    };
    float *input_data_device = nullptr;

    int ib = 2, iw = 3, ih = 3;
    float output_data_host[ib*iw*ih];
    float *output_data_device = nullptr;

    cudaMalloc(&input_data_device, sizeof(input_data_host));
    cudaMalloc(&output_data_device, sizeof(output_data_host));

    cudaMemcpyAsync(input_data_device, input_data_host, sizeof(input_data_host), cudaMemcpyHostToDevice, stream);

    // 用一个指针数组指定input和output在gpu中的指针。
    execution_context->setBindingDimensions(0, nvinfer1::Dims4(ib, 1, ih, iw));
    float *bindings[] = {input_data_device, output_data_device};

    // ------------------------------ 3. 推理并将结果搬运回CPU   ---------------------------
    bool success = execution_context->enqueueV2((void**)bindings, stream, nullptr);
    if (!success)
    {
        printf("inference fail\n");
        delete execution_context;
        delete engine;
        delete runtime;
        return ;
    }

    cudaMemcpyAsync(output_data_host, output_data_device, sizeof(output_data_host), cudaMemcpyDeviceToHost, stream);

    cudaStreamSynchronize(stream);
    for(int b = 0; b < ib; ++b){
        printf("batch %d. output_data_host = \n", b);
        for(int i = 0; i < iw * ih; ++i){
            printf("%f, ", output_data_host[b * iw * ih + i]);
            if((i + 1) % iw == 0)
                printf("\n");
        }
    }

    // ------------------------------ 4. 释放内存 ----------------------------
    printf("Clean memory\n");
    cudaStreamDestroy(stream);
    delete execution_context;
    delete engine;
    delete runtime;

}

int main()
{
    const char *file = "/home/srb/trtLearning/dynamic-shape/engine.trtmodel";
    if (access(file, F_OK) != 0)
    {
        build_model();
    }
    string str = file;
    inference(str);
    return 0;
}