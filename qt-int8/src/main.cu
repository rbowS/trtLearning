#include <NvInfer.h>
#include <NvInferRuntime.h>
#include "../include/onnx-tensorrt/NvOnnxParser.h"

#include <cuda_runtime.h>

#include <stdio.h>
#include <math.h>

#include <unistd.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <unistd.h>
#include <functional>
#include <memory>
#include <assert.h>
#include <algorithm>
#include <opencv2/opencv.hpp>


using namespace std;

#define checkRuntime(op) __check_cuda_runtime((op), #op, __FILE__, __LINE__)


bool __check_cuda_runtime(cudaError_t code, const char *op, const char *file, int line)
{
    if(code != cudaSuccess){    
        const char* err_name = cudaGetErrorName(code);    
        const char* err_message = cudaGetErrorString(code);  
        printf("runtime error %s:%d  %s failed. \n  code = %s, message = %s\n", file, line, op, err_name, err_message);   
        return false;
    }
    return true;
}

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


typedef std::function<void(
    int current, int count, const std::vector<std::string> &files,
    nvinfer1::Dims dims, float *ptensor
)> Int8Process;

// int8熵校准器：用于评估量化前后的分布改变
class Int8EntropyCalibrator : public nvinfer1::IInt8EntropyCalibrator2
{

private:
    //一个函数指针，用于对输入的图像数据进行预处理，如裁剪、缩放、归一化等。
    Int8Process preprocess_;
    //一个字符串向量，存储了所有用于标定的图像文件的路径。
    vector<string> allimgs_;
    //一个整数，表示每个标定批次在CUDA内存中占用的字节数。
    size_t batchCudaSize_ = 0;
    //一个整数，表示当前标定批次的起始位置在allimgs_中的索引。
    int cursor_ = 0;
    //一个整数，表示每个标定批次在主机内存中占用的字节数。
    size_t bytes_ = 0;
    //一个nvinfer1::Dims类型的对象，表示标定输入张量的维度信息。
    nvinfer1::Dims dims_;
    //一个字符串向量，存储了当前标定批次中的图像文件的路径。
    vector<string> files_;
    //一个浮点数指针，指向主机内存中分配的用于存储标定输入张量的空间。
    float *tensor_host_ = nullptr;
    //一个浮点数指针，指向CUDA内存中分配的用于存储标定输入张量的空间。
    float *tensor_device_ = nullptr;
    //一个无符号字符向量，存储了标定缓存数据，即计算出的量化因子等信息
    vector<uint8_t> entropyCalibratorData_;
    //一个布尔值，表示是否从已有的标定缓存数据中加载标定结果，而不是重新进行标定。
    bool fromCalibratorData_ = false;

public:
    //第一个构造函数接受三个参数：imagefiles、dims和preprocess。这个构造函数是用于从头开始进行标定的，
    //它会将imagefiles赋值给allimgs_，将dims赋值给dims_，将preprocess赋值给preprocess_，
    //并将fromCalibratorData_设为false。它还会根据dims.d[0]（即标定批次大小）来调整files_的大小
    Int8EntropyCalibrator(const vector<string> &imagefiles, nvinfer1::Dims dims, const Int8Process preprocess)
    {
        assert(preprocess != nullptr);
        this->dims_ = dims;
        this->allimgs_ = imagefiles;
        this->preprocess_ = preprocess;
        this->fromCalibratorData_ = false;
        files_.resize(dims.d[0]);
    }

    // 第二个构造函数接受三个参数：entropyCalibratorData、dims和preprocess。
    // 这个构造函数是用于从已有的标定缓存数据中加载标定结果的，
    // 它会将entropyCalibratorData赋值给entropyCalibratorData_，将dims赋值给dims_，
    // 将preprocess赋值给preprocess_，并将fromCalibratorData_设为true。它也会根据dims.d[0]来调整files_的大小
    Int8EntropyCalibrator(const vector<uint8_t> &entropyCalibratorData, nvinfer1::Dims dims, const Int8Process &preprocess)
    {
        assert(preprocess != nullptr);
        this->dims_ = dims;
        this->entropyCalibratorData_ = entropyCalibratorData;
        this->preprocess_ = preprocess;
        this->fromCalibratorData_ = true;
        files_.resize(dims.d[0]);
    }

    virtual ~Int8EntropyCalibrator()
    {
        if(tensor_host_ != nullptr)
        {
            checkRuntime(cudaFreeHost(tensor_host_));
            checkRuntime(cudaFree(tensor_device_));
            tensor_host_ = nullptr;
            tensor_device_ = nullptr;
        }
    }

    // 想要按照多少的batch进行标定, 返回标定批次大小，即dims_.d[0]。
    int getBatchSize() const noexcept
    {
        return dims_.d[0];
    }

    // 根据cursor_和allimgs_来更新files_中的内容，并调用preprocess_对files_中的图像进行预处理，
    // 并将结果拷贝到tensor_host_和tensor_device_中。如果cursor_加上batch_size超过了allimgs_.size()，
    // 则返回false；否则返回true。如果tensor_host_为空，则根据dims_计算bytes_并分配主机内存和CUDA内存空间。
    bool next()
    {
        int batch_size = dims_.d[0];
        if(cursor_ + batch_size > allimgs_.size())
            return false;
        
        for (int i = 0; i < batch_size; i++)
        {
            files_[i] = allimgs_[cursor_++];
        }

        if(tensor_host_ == nullptr)
        {
            size_t volumn = 1;
            for (int i = 0; i < dims_.nbDims; i++)
            {
                volumn *= dims_.d[i];
            }
            bytes_ = volumn*sizeof(float);
            checkRuntime(cudaMallocHost(&tensor_host_, bytes_));
            checkRuntime(cudaMalloc(&tensor_device_, bytes_));
        }

        preprocess_(cursor_, allimgs_.size(), files_, dims_, tensor_host_);

        checkRuntime(cudaMemcpy(tensor_device_, tensor_host_, bytes_, cudaMemcpyHostToDevice));

        return true;
    }

    // 调用next()来获取下一个标定批次，并将tensor_device_赋值给bindings[0]。
    // 如果next()返回false，则表示没有更多的标定数据了，返回false；否则返回true。
    bool getBatch(void *bindings[], const char *names[], int nbBindings) noexcept
    {
        if(!next())
            return false;
        bindings[0] = tensor_device_;
        return true;
    }

    
    const vector<uint8_t> getEntropyCalibratorData()
    {
        return entropyCalibratorData_;
    }

    // 如果fromCalibratorData_为true，则返回entropyCalibratorData_.data()作为缓存数据，
    // 并将entropyCalibratorData_.size()赋值给length；否则返回nullptr，并将length设为0。
    const void *readCalibrationCache(size_t &length) noexcept
    {
        if(fromCalibratorData_)
        {
            length = this->entropyCalibratorData_.size();
            return this->entropyCalibratorData_.data();
        }
        length = 0;
        return nullptr;
    }

    // 将cache指向的缓存数据拷贝到entropyCalibratorData_中，并将length赋值给entropyCalibratorData_.size()。
    virtual void writeCalibrationCache(const void *cache, size_t length) noexcept
    {
        entropyCalibratorData_.assign((uint8_t*)cache, (uint8_t*)cache+length);
    }

};


// 通过智能指针管理nv返回的指针参数
// 内存自动释放，避免泄漏
template<typename _T>
static shared_ptr<_T> make_nvshared(_T *ptr)
{
    return shared_ptr<_T>(ptr, [](_T *p){p->destroy();});
}

static bool exist(const string &path)
{
    #ifdef _WIN32
    return ::PathFileExistsA(path.c_str());
    #else
        return access(path.c_str(), R_OK) == 0;
    #endif
}

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

vector<string> load_labels(const char* file)
{
    vector<string> lines;

    ifstream in(file, ios::in | ios::binary);
    if(!in.is_open())
    {
        printf("open %d failed.\n", file);
        return lines;
    }

    string line;
    while (getline(in, line))
    {
        lines.push_back(line);
    }
    in.close();
    return lines;
}



bool build_model(const char *path)
{   
    TRTLogger logger;
    // ----------------------------- 1. 定义 builder, config 和network -----------------------------
    auto builder = make_nvshared( nvinfer1::createInferBuilder(logger) );
    auto config = make_nvshared( builder->createBuilderConfig() );
    auto network = make_nvshared( builder->createNetworkV2(1) );

    // ----------------------------- 2. 输入，模型结构和输出的基本信息 -----------------------------
    // 通过onnxparser解析的结果会填充到network中，类似addConv的方式添加进去
    auto parser = make_nvshared( nvonnxparser::createParser(*network, logger) );
    if(!parser->parseFromFile(path, 1))
    {
        printf("Failed to parser demo.onnx\n");
        return false;
    }
    int maxBatchSize = 10;
    printf("Workspace Size = %.2f MB\n", (1 << 28) / 1024.0f / 1024.0f);
    config->setMaxWorkspaceSize(1<<28);

    // --------------------------------- 2.1 关于profile ----------------------------------
    // 如果模型有多个输入，则必须多个profile
    auto profile = builder->createOptimizationProfile();
    auto input_tensor = network->getInput(0);
    auto input_dims = input_tensor->getDimensions();
    input_dims.d[0] = 1;

    config->setFlag(nvinfer1::BuilderFlag::kINT8);

    auto preprocess = [](
        int current, int count, const::std::vector<std::string> &files,
        nvinfer1::Dims dims, float *ptensor
    ){
        printf("Preprocess %d / %d\n", count, current);

        // 标定所采用的数据预处理必须与推理时一样
        int width = dims.d[3];
        int height = dims.d[2];
        float mean[] = {0.406, 0.456, 0.485};
        float stds[]  = {0.225, 0.224, 0.229};
        for (int i = 0; i < files.size(); i++)
        {
            auto image = cv::imread(files[i]);
            cv::resize(image, image, cv::Size(width, height));
            int image_area = width * height;
            unsigned char *pimage = image.data;
            float *phost_b = ptensor + image_area*0;
            float *phost_g = ptensor + image_area*1;
            float *phost_r = ptensor + image_area*2;
            for (int i = 0; i < image_area; i++, pimage+=3)
            {
                *phost_r++ = (pimage[0]/255.0 - mean[0]) / stds[0];
                *phost_g++ = (pimage[1]/255.0 - mean[1]) / stds[1];
                *phost_r++ = (pimage[2]/255.0 - mean[2]) / stds[2];
            }
            ptensor += image_area*3;
        }
    };

    shared_ptr<Int8EntropyCalibrator> calib(new Int8EntropyCalibrator(
        {"workspace/kej.jpg"}, input_dims, preprocess
    ));

    config->setInt8Calibrator(calib.get());

    // 配置最小允许batch
    input_dims.d[0] = 1;
    // 配置输入的最小、最优、最大的范围
    profile->setDimensions(input_tensor->getName(), nvinfer1::OptProfileSelector::kMIN, input_dims);
    profile->setDimensions(input_tensor->getName(), nvinfer1::OptProfileSelector::kOPT, input_dims);
    
    input_dims.d[0] = maxBatchSize;
    profile->setDimensions(input_tensor->getName(), nvinfer1::OptProfileSelector::kMAX, input_dims);

    // 添加到配置
    config->addOptimizationProfile(profile);

    auto engine = make_nvshared( builder->buildEngineWithConfig(*network, *config) );
    if(engine == nullptr){
        printf("Build engine failed.\n");
        return false;
    }
    // -------------------------- 3. 序列化 ----------------------------------
    // 将模型序列化，并储存为文件
    auto model_data = make_nvshared( engine->serialize() );
    FILE* f = fopen("workspace/engine.trtmodel", "wb");
    fwrite(model_data->data(), 1, model_data->size(), f);
    fclose(f);

    // 卸载顺序按照构建顺序倒序
    printf("Done.\n");
    return true;
}

void inference(const char *path){

    TRTLogger logger;
    auto engine_data = load_file(path);
    auto runtime   = make_nvshared( nvinfer1::createInferRuntime(logger) );
    auto engine = make_nvshared( runtime->deserializeCudaEngine(engine_data.data(), engine_data.size()) );
    if(engine == nullptr){
        printf("Deserialize cuda engine failed.\n");
        return;
    }

    auto execution_context = make_nvshared( engine->createExecutionContext() );
    cudaStream_t stream = nullptr;
    checkRuntime(cudaStreamCreate(&stream));

    int input_batch   = 1;
    int input_channel = 3;
    int input_height  = 224;
    int input_width   = 224;
    int input_numel   = input_batch * input_channel * input_height * input_width;
    float* input_data_host   = nullptr;
    float* input_data_device = nullptr;

    checkRuntime(cudaMallocHost(&input_data_host, input_numel * sizeof(float)));
    checkRuntime(cudaMalloc(&input_data_device, input_numel * sizeof(float)));

    ///////////////////////////////////////////////////
    // image to float
    auto image = cv::imread("workspace/kej.jpg");
    float mean[] = {0.406, 0.456, 0.485};
    float stds[]  = {0.225, 0.224, 0.229};

    // 对应于pytorch的代码部分
    cv::resize(image, image, cv::Size(input_width, input_height));
    int image_area = image.cols * image.rows;
    unsigned char* pimage = image.data;
    float* phost_b = input_data_host + image_area * 0;
    float* phost_g = input_data_host + image_area * 1;
    float* phost_r = input_data_host + image_area * 2;
    for(int i = 0; i < image_area; ++i, pimage += 3){
        // 注意这里的顺序rgb调换了
        *phost_r++ = (pimage[0] / 255.0f - mean[0]) / stds[0];
        *phost_g++ = (pimage[1] / 255.0f - mean[1]) / stds[1];
        *phost_b++ = (pimage[2] / 255.0f - mean[2]) / stds[2];
    }
    ///////////////////////////////////////////////////
    checkRuntime(cudaMemcpyAsync(input_data_device, input_data_host, input_numel * sizeof(float), cudaMemcpyHostToDevice, stream));

    // 设置输出
    const int num_classes = 1000;
    float output_data_host[num_classes];
    float* output_data_device = nullptr;
    checkRuntime(cudaMalloc(&output_data_device, sizeof(output_data_host)));
    
    // 明确当前推理时，使用的数据输入大小
    auto input_dims = execution_context->getBindingDimensions(0);
    input_dims.d[0] = input_batch;

    execution_context->setBindingDimensions(0, input_dims);
    float* bindings[] = {input_data_device, output_data_device};
    bool success      = execution_context->enqueueV2((void**)bindings, stream, nullptr);
    checkRuntime(cudaMemcpyAsync(output_data_host, output_data_device, sizeof(output_data_host), cudaMemcpyDeviceToHost, stream));
    checkRuntime(cudaStreamSynchronize(stream));

    float *prob = output_data_host;
    int pred_label = std::max_element(prob, prob+num_classes)-prob;
    auto labels = load_labels("workspace/labels.imagenet.txt");
    auto predict_name = labels[pred_label];
    float confidence = prob[pred_label];
    
    printf("Predict: %s, confidence = %f, label = %d\n", predict_name.c_str(), confidence, pred_label);

    checkRuntime(cudaStreamDestroy(stream));
    checkRuntime(cudaFreeHost(input_data_host));
    checkRuntime(cudaFree(input_data_device));
    checkRuntime(cudaFree(output_data_device));
  
}

int main(){
    const char *path = "/home/srb/trtLearning/qt-int8/workspace/classifier.onnx";
    const char *engine_path = "/home/srb/trtLearning/qt-int8/workspace/engine.trtmodel";
    if (!access(engine_path, F_OK) == 0)
    {
        if(!build_model(path)){
            return -1;
        }
    }
    
    inference(engine_path);
    return 0;
}
