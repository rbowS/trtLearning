#include <NvInfer.h>
#include <NvInferRuntime.h>
#include "../include/NvOnnxParser.h"

#include <cuda_runtime.h>

#include <stdio.h>
#include <math.h>

#include <unistd.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <unistd.h>

using std::vector;
using std::string;
using std::ifstream;
using std::ios;


inline const char* severity_string(nvinfer1::ILogger::Severity t)
{
    switch (t)
    {
        case nvinfer1::ILogger::Severity::kINTERNAL_ERROR: return "internal_error";
        case nvinfer1::ILogger::Severity::kERROR: return "error";
        case nvinfer1::ILogger::Severity::kWARNING: return "warning";
        case nvinfer1::ILogger::Severity::kINFO: return "info";
        case nvinfer1::ILogger::Severity::kVERBOSE: return "verbose";
        default: return "unkonw";
    }
}

class TRTLogger : public nvinfer1::ILogger{

public:
    virtual void log(Severity severity, nvinfer1::AsciiChar const *msg)
    noexcept override
    {
        if(severity <= Severity::kINFO)
        {
            if(severity == Severity::kWARNING)
            {
                printf("\033[33m%s: %s\033[0m\n", severity_string(severity), msg);
            }
            else if(severity < Severity::kERROR)
            {
                printf("\033[31m%s: %s\033[0m\n", severity_string(severity), msg);
            }
            else
            {
                printf("%s: %s\n", severity_string(severity), msg);
            }
        }
    }

};

vector<unsigned char> load_file(const string& file){
    ifstream in(file, ios::in | ios::binary);
    if (!in.is_open())
        return {};

    in.seekg(0, ios::end);
    size_t length = in.tellg();

    std::vector<uint8_t> data;
    if (length > 0){
        in.seekg(0, ios::beg);
        data.resize(length);

        in.read((char*)&data[0], length);
    }
    in.close();
    return data;
}

bool build_model(const char *path)
{   
    TRTLogger logger;
    // ----------------------------- 1. 定义 builder, config 和network -----------------------------
    nvinfer1::IBuilder *builder = nvinfer1::createInferBuilder(logger);
    nvinfer1::IBuilderConfig *config = builder->createBuilderConfig();
    nvinfer1::INetworkDefinition *network = builder->createNetworkV2(1);

    // ----------------------------- 2. 输入，模型结构和输出的基本信息 -----------------------------
    // 通过onnxparser解析的结果会填充到network中，类似addConv的方式添加进去
    nvonnxparser::IParser *parser = nvonnxparser::createParser(*network, logger);
    if(!parser->parseFromFile(path, 1))
    {
        printf("Failed to parser demo.onnx\n");
        delete builder;
        delete config;
        delete parser;
        return false;
    }
    int maxBatchSize = 10;
    printf("Workspace Size = %.2f MB\n", (1 << 28) / 1024.0f / 1024.0f);
    config->setMaxWorkspaceSize(1<<28);

    // --------------------------------- 2.1 关于profile ----------------------------------
    // 如果模型有多个输入，则必须多个profile
    auto profile = builder->createOptimizationProfile();
    auto input_tensor = network->getInput(0);
    int input_channel = input_tensor->getDimensions().d[1];

    // 配置输入的最小、最优、最大的范围
    profile->setDimensions(input_tensor->getName(), nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims4(1, input_channel, 3, 3));
    profile->setDimensions(input_tensor->getName(), nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims4(1, input_channel, 3, 3));
    profile->setDimensions(input_tensor->getName(), nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims4(maxBatchSize, input_channel, 5, 5));

    // 添加到配置
    config->addOptimizationProfile(profile);

    nvinfer1::ICudaEngine *engine = builder->buildEngineWithConfig(*network, *config);
    if(engine == nullptr){
        printf("Build engine failed.\n");
        return false;
    }
    // -------------------------- 3. 序列化 ----------------------------------
    // 将模型序列化，并储存为文件
    nvinfer1::IHostMemory* model_data = engine->serialize();
    FILE* f = fopen("engine.trtmodel", "wb");
    fwrite(model_data->data(), 1, model_data->size(), f);
    fclose(f);

    // 卸载顺序按照构建顺序倒序
    delete model_data;
    delete parser;
    delete engine;
    delete network;
    delete config;
    delete builder;
    printf("Done.\n");
    return true;
}

void inference(const char *path){

    TRTLogger logger;
    auto engine_data = load_file(path);
    nvinfer1::IRuntime* runtime   = nvinfer1::createInferRuntime(logger);
    nvinfer1::ICudaEngine* engine = runtime->deserializeCudaEngine(engine_data.data(), engine_data.size());
    if(engine == nullptr){
        printf("Deserialize cuda engine failed.\n");
        runtime->destroy();
        return;
    }

    nvinfer1::IExecutionContext* execution_context = engine->createExecutionContext();
    cudaStream_t stream = nullptr;
    cudaStreamCreate(&stream);

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
    float* input_data_device = nullptr;

    // 3x3输入，对应3x3输出
    int ib = 2;
    int iw = 3;
    int ih = 3;
    float output_data_host[ib * iw * ih];
    float* output_data_device = nullptr;
    cudaMalloc(&input_data_device, sizeof(input_data_host));
    cudaMalloc(&output_data_device, sizeof(output_data_host));
    cudaMemcpyAsync(input_data_device, input_data_host, sizeof(input_data_host), cudaMemcpyHostToDevice, stream);

    // 明确当前推理时，使用的数据输入大小
    execution_context->setBindingDimensions(0, nvinfer1::Dims4(ib, 1, ih, iw));
    float* bindings[] = {input_data_device, output_data_device};
    bool success      = execution_context->enqueueV2((void**)bindings, stream, nullptr);
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

    printf("Clean memory\n");
    cudaStreamDestroy(stream);
    cudaFree(input_data_device);
    cudaFree(output_data_device);
    execution_context->destroy();
    engine->destroy();
    runtime->destroy();
}

int main(){
    const char *path = "/home/srb/trtLearning/hello-plugin/demo.onnx";
    const char *engine_path = "/home/srb/trtLearning/hello-plugin/engine.trtmodel";
    if (!access(engine_path, F_OK) == 0)
    {
        if(!build_model(path)){
            return -1;
        }
    }
    
    inference(engine_path);
    return 0;
}
