#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <NvOnnxParser.h>

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

int main(){
    const char *path = "/home/srb/trtLearning/onnx-parser/demo.onnx";
    build_model(path);
    return 0;
}