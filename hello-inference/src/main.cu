#include <NvInfer.h>
#include <NvInferRuntime.h>

#include <cuda_runtime.h>

#include <stdio.h>
#include <math.h>

#include <iostream>
#include <fstream>
#include <vector>

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
        linear (fully connected)  input = 3, output = 2, bias = True     w=[[1.0, 2.0, 0.5], [0.1, 0.2, 0.5]], b=[0.3, 0.8]
          |
        sigmoid
          |
        prob
    */
    const int num_input = 3;
    const int num_output = 2;
    float layer1_weight_values[] = {1.0, 2.0, 0.5, 0.1, 0.2, 0.5};
    float layer1_bias_values[] = {0.3, 0.8};

    nvinfer1::ITensor *input = network->addInput("image", nvinfer1::DataType::kFLOAT, nvinfer1::Dims4(1, num_input, 1, 1));
    nvinfer1::Weights layer1_weight = make_weights(layer1_weight_values, 6);
    nvinfer1::Weights layer1_bias = make_weights(layer1_bias_values, 2);

    auto layer1 = network->addFullyConnected(*input, num_output, layer1_weight, layer1_bias);
    auto prob = network->addActivation(*layer1->getOutput(0), nvinfer1::ActivationType::kSIGMOID);

    // 将我们需要的prob标记为输出
    network->markOutput(*prob->getOutput(0));

    printf("Workspace Size = %.2f MB\n", (1 << 28) / 1024.0f / 1024.0f);

    config->setMaxWorkspaceSize(1<<28);
    builder->setMaxBatchSize(1);

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
        linear (fully connected)  input = 3, output = 2, bias = True     w=[[1.0, 2.0, 0.5], [0.1, 0.2, 0.5]], b=[0.3, 0.8]
          |
        sigmoid
          |
        prob
    */
    float input_data_host[] = {1,2,3};
    float *input_data_device = nullptr;

    float output_data_host[2];
    float *output_data_device = nullptr;

    cudaMalloc(&input_data_device, sizeof(input_data_host));
    cudaMalloc(&output_data_device, sizeof(output_data_host));

    cudaMemcpyAsync(input_data_device, input_data_host, sizeof(input_data_host), cudaMemcpyHostToDevice, stream);

    // 用一个指针数组指定input和output在gpu中的指针。
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
    printf("output_data_host = %f, %f\n", output_data_host[0], output_data_host[1]);

    // ------------------------------ 4. 释放内存 ----------------------------
    printf("Clean memory\n");
    cudaStreamDestroy(stream);
    delete execution_context;
    delete engine;
    delete runtime;

    // ------------------------------ 5. 手动推理进行验证 ----------------------------
    const int num_input = 3;
    const int num_output = 2;
    float layer1_weight_values[] = {1.0, 2.0, 0.5, 0.1, 0.2, 0.5};
    float layer1_bias_values[]   = {0.3, 0.8};
    printf("手动验证计算结果：\n");
    for (int i = 0; i < num_output; i++)
    {
        float output_host = layer1_bias_values[1];
        for (int j = 0; j < num_input; j++)
        {
            output_host += layer1_weight_values[i*num_input+j]*input_data_host[j];
        }
        float prob = 1/(1+exp(-output_host));
        printf("output_prob[%d] = %f\n", i, prob);
    }
}

int main()
{
    const string file = "/home/srb/trtLearning/hello/engine.trtmodel";
    inference(file);
    return 0;
}