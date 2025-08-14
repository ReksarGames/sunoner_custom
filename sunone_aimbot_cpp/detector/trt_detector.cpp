#ifdef USE_CUDA
#define WIN32_LEAN_AND_MEAN
#define _WINSOCKAPI_
#include <winsock2.h>
#include <Windows.h>

#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <algorithm>
#include <atomic>
#include <numeric>
#include <vector>
#include <queue>
#include <mutex>

#include "trt_detector.h"
#include "nvinf.h"
#include "sunone_aimbot_cpp.h"
#include "other_tools.h"
#include "postProcess.h"

#include "fused_preprocess.h"
#include <opencv2/core/cuda_stream_accessor.hpp>

extern std::atomic<bool> detectionPaused;
int model_quant;
std::vector<float> outputData;

extern std::atomic<bool> detector_model_changed;
extern std::atomic<bool> detection_resolution_changed;

static bool error_logged = false;

TrtDetector::TrtDetector()
    : frameReady(false),
    shouldExit(false),
    inputBufferDevice(nullptr),
    img_scale(1.0f),
    numClasses(0)
{
    cudaStreamCreate(&stream);
}

TrtDetector::~TrtDetector()
{
    for (auto& buffer : pinnedOutputBuffers)
    {
        if (buffer.second) cudaFreeHost(buffer.second);
    }
    for (auto& binding : inputBindings)
    {
        if (binding.second) cudaFree(binding.second);
    }
    for (auto& binding : outputBindings)
    {
        if (binding.second) cudaFree(binding.second);
    }
    if (inputBufferDevice)
    {
        cudaFree(inputBufferDevice);
    }
}

void TrtDetector::destroyCudaGraph()
{
    cudaStreamDestroy(stream);

    if (cudaGraphCaptured)
    {
        cudaGraphExecDestroy(cudaGraphExec);
        cudaGraphDestroy(cudaGraph);
        cudaGraphCaptured = false;
    }
}

void TrtDetector::captureCudaGraph()
{
    if (!useCudaGraph || cudaGraphCaptured) return;

    // На всякий случай — убедимся, что предыдущие работы завершены
    cudaStreamSynchronize(stream);

    cudaError_t st = cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
    if (st != cudaSuccess) {
        std::cerr << "[Detector] BeginCapture failed: "
            << cudaGetErrorString(st) << std::endl;
        return;
    }

    // 1) Инференс
    context->enqueueV3(stream);

    // 2) Асинхронные копии всех выходов в pinned host
    for (const auto& name : outputNames)
    {
        auto itD = outputBindings.find(name);
        auto itH = pinnedOutputBuffers.find(name);
        if (itD == outputBindings.end() || itH == pinnedOutputBuffers.end()) continue;

        cudaMemcpyAsync(
            itH->second,
            itD->second,
            outputSizes[name],
            cudaMemcpyDeviceToHost,
            stream);
    }

    // Завершаем захват
    st = cudaStreamEndCapture(stream, &cudaGraph);
    if (st != cudaSuccess) {
        std::cerr << "[Detector] EndCapture failed: "
            << cudaGetErrorString(st) << std::endl;
        return;
    }

    st = cudaGraphInstantiate(&cudaGraphExec, cudaGraph, 0);
    if (st != cudaSuccess) {
        std::cerr << "[Detector] GraphInstantiate failed: "
            << cudaGetErrorString(st) << std::endl;
        return;
    }

    cudaGraphCaptured = true;
}

inline void TrtDetector::launchCudaGraph()
{
    auto err = cudaGraphLaunch(cudaGraphExec, stream);
    if (err != cudaSuccess)
    {
        std::cerr << "[Detector] GraphLaunch failed: " << cudaGetErrorString(err) << std::endl;
    }
}

void TrtDetector::getInputNames()
{
    inputNames.clear();
    inputSizes.clear();

    for (int i = 0; i < engine->getNbIOTensors(); ++i)
    {
        const char* name = engine->getIOTensorName(i);
        if (engine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT)
        {
            inputNames.emplace_back(name);
            if (config.verbose)
            {
                std::cout << "[Detector] Detected input: " << name << std::endl;
            }
        }
    }
}

void TrtDetector::getOutputNames()
{
    outputNames.clear();
    outputSizes.clear();
    outputTypes.clear();
    outputShapes.clear();

    for (int i = 0; i < engine->getNbIOTensors(); ++i)
    {
        const char* name = engine->getIOTensorName(i);
        if (engine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kOUTPUT)
        {
            outputNames.emplace_back(name);
            outputTypes[name] = engine->getTensorDataType(name);
            
            if (config.verbose)
            {
                std::cout << "[Detector] Detected output: " << name << std::endl;
            }
        }
    }
}

void TrtDetector::getBindings()
{
    // Освобождаем старые буферы
    for (auto& binding : inputBindings)
        if (binding.second) cudaFree(binding.second);
    inputBindings.clear();

    for (auto& binding : outputBindings)
        if (binding.second) cudaFree(binding.second);
    outputBindings.clear();

    // Освобождаем pinned host буферы под выходы (если были)
    for (auto& kv : pinnedOutputBuffers)
        if (kv.second) cudaFreeHost(kv.second);
    pinnedOutputBuffers.clear();

    // Входы — как и раньше: device память
    for (const auto& name : inputNames)
    {
        size_t size = inputSizes[name];
        if (size == 0) continue;

        void* dptr = nullptr;
        cudaError_t err = cudaMalloc(&dptr, size);
        if (err == cudaSuccess)
        {
            inputBindings[name] = dptr;
            if (config.verbose)
                std::cout << "[Detector] Allocated " << size
                << " bytes for input " << name << std::endl;
        }
        else
        {
            std::cerr << "[Detector] Failed to allocate input memory for '"
                << name << "': " << cudaGetErrorString(err) << std::endl;
        }
    }

    // Выходы — device + pinned host
    for (const auto& name : outputNames)
    {
        size_t size = outputSizes[name];
        if (size == 0) continue;

        // device
        void* dptr = nullptr;
        cudaError_t err = cudaMalloc(&dptr, size);
        if (err != cudaSuccess)
        {
            std::cerr << "[Detector] Failed to allocate DEVICE output '"
                << name << "': " << cudaGetErrorString(err)
                << " (" << size << " bytes)" << std::endl;
            continue;
        }
        outputBindings[name] = dptr;

        // pinned host
        void* hptr = nullptr;
        err = cudaMallocHost(&hptr, size);
        if (err != cudaSuccess)
        {
            std::cerr << "[Detector] Failed to allocate PINNED HOST for '"
                << name << "': " << cudaGetErrorString(err)
                << " (" << size << " bytes)" << std::endl;
            cudaFree(dptr);
            outputBindings.erase(name);
            continue;
        }
        pinnedOutputBuffers[name] = hptr;

        if (config.verbose)
        {
            std::cout << "[Detector] Allocated " << size << " bytes for output "
                << name << " (device) and pinned host buffer." << std::endl;
        }
    }
}

void TrtDetector::initialize(const std::string& modelFile)
{
    runtime.reset(nvinfer1::createInferRuntime(gLogger));
    loadEngine(modelFile);

    if (!engine)
    {
        std::cerr << "[Detector] Engine loading failed" << std::endl;
        return;
    }

    context.reset(engine->createExecutionContext());
    if (!context)
    {
        std::cerr << "[Detector] Context creation failed" << std::endl;
        return;
    }

    getInputNames();
    getOutputNames();

    if (inputNames.empty())
    {
        std::cerr << "[Detector] No input tensors found" << std::endl;
        return;
    }

    inputName = inputNames[0];

    context->setInputShape(inputName.c_str(), nvinfer1::Dims4{ 1, 3, 640, 640 });
    if (!context->allInputDimensionsSpecified())
    {
        std::cerr << "[Detector] Failed to set input dimensions" << std::endl;
        return;
    }

    for (const auto& inName : inputNames)
    {
        nvinfer1::Dims dims = context->getTensorShape(inName.c_str());
        nvinfer1::DataType dtype = engine->getTensorDataType(inName.c_str());
        inputSizes[inName] = getSizeByDim(dims) * getElementSize(dtype);
    }
    for (const auto& outName : outputNames)
    {
        nvinfer1::Dims dims = context->getTensorShape(outName.c_str());
        nvinfer1::DataType dtype = engine->getTensorDataType(outName.c_str());
        outputSizes[outName] = getSizeByDim(dims) * getElementSize(dtype);

        std::vector<int64_t> shape;
        for (int j = 0; j < dims.nbDims; j++)
        {
            shape.push_back(dims.d[j]);
        }
        outputShapes[outName] = shape;
    }

    getBindings();

    if (!outputNames.empty())
    {
        const std::string& mainOut = outputNames[0];
        nvinfer1::Dims outDims = context->getTensorShape(mainOut.c_str());

        if (config.postprocess == "yolo10")
        {
            numClasses = 11;
        }
        else
        {
            numClasses = outDims.d[1] - 4;
        }
    }

    img_scale = static_cast<float>(config.detection_resolution) / 640;
    nvinfer1::Dims dims = context->getTensorShape(inputName.c_str());
    int c = dims.d[1];
    int h = dims.d[2];
    int w = dims.d[3];
    resizedBuffer.create(h, w, CV_8UC3);
    floatBuffer.create(h, w, CV_32FC3);
    channelBuffers.resize(c);
    for (int i = 0; i < c; ++i)
    {
        channelBuffers[i].create(h, w, CV_32F);
    }
    for (const auto& name : inputNames)
    {
        context->setTensorAddress(name.c_str(), inputBindings[name]);
    }
    for (const auto& name : outputNames)
    {
        context->setTensorAddress(name.c_str(), outputBindings[name]);
    }
}

size_t TrtDetector::getSizeByDim(const nvinfer1::Dims& dims)
{
    size_t size = 1;
    for (int i = 0; i < dims.nbDims; ++i)
    {
        if (dims.d[i] < 0) return 0;
        size *= dims.d[i];
    }
    return size;
}

size_t TrtDetector::getElementSize(nvinfer1::DataType dtype)
{
    switch (dtype)
    {
        case nvinfer1::DataType::kFLOAT: return 4;
        case nvinfer1::DataType::kHALF: return 2;
        case nvinfer1::DataType::kINT32: return 4;
        case nvinfer1::DataType::kINT8: return 1;
        default: return 0;
    }
}

void TrtDetector::loadEngine(const std::string& modelFile)
{
    std::string engineFilePath;
    std::filesystem::path modelPath(modelFile);
    std::string extension = modelPath.extension().string();

    if (extension == ".engine")
    {
        engineFilePath = modelFile;
    }
    else if (extension == ".onnx")
    {
        engineFilePath = modelPath.replace_extension(".engine").string();

        if (!fileExists(engineFilePath))
        {
            std::cout << "[Detector] Building engine from ONNX model" << std::endl;

            nvinfer1::ICudaEngine* builtEngine = buildEngineFromOnnx(modelFile, gLogger);
            if (builtEngine)
            {
                nvinfer1::IHostMemory* serializedEngine = builtEngine->serialize();

                if (serializedEngine)
                {
                    std::ofstream engineFile(engineFilePath, std::ios::binary);
                    if (engineFile)
                    {
                        engineFile.write(reinterpret_cast<const char*>(serializedEngine->data()), serializedEngine->size());
                        engineFile.close();
                        
                        config.ai_model = std::filesystem::path(engineFilePath).filename().string();
                        config.saveConfig("config.ini");
                        
                        std::cout << "[Detector] Engine saved to: " << engineFilePath << std::endl;
                    }
                    delete serializedEngine;
                }
                delete builtEngine;
            }
        }
    }
    else
    {
        std::cerr << "[Detector] Unsupported model format: " << extension << std::endl;
        return;
    }

    std::cout << "[Detector] Loading engine: " << engineFilePath << std::endl;
    engine.reset(loadEngineFromFile(engineFilePath, runtime.get()));
}

void TrtDetector::processFrame(const cv::Mat& frame)
{
    if (config.backend == "DML") return;

    if (detectionPaused)
    {
        std::lock_guard<std::mutex> lock(detectionBuffer.mutex);
        detectionBuffer.boxes.clear();
        detectionBuffer.classes.clear();
        return;
    }

    std::unique_lock<std::mutex> lock(inferenceMutex);
    currentFrame = frame.clone();
    frameReady = true;
    inferenceCV.notify_one();
}

void TrtDetector::inferenceThread()
{
    while (!shouldExit)
    {
        if (detector_model_changed.load())
        {
            {
                std::unique_lock<std::mutex> lock(inferenceMutex);
                context.reset();
                engine.reset();
                for (auto& b : inputBindings)  if (b.second) cudaFree(b.second);
                inputBindings.clear();
                for (auto& b : outputBindings) if (b.second) cudaFree(b.second);
                outputBindings.clear();
                for (auto& kv : pinnedOutputBuffers) if (kv.second) cudaFreeHost(kv.second);
                pinnedOutputBuffers.clear();
            }
            initialize("models/" + config.ai_model);
            detection_resolution_changed.store(true);
            detector_model_changed.store(false);
        }

        cv::Mat frame;
        bool hasNewFrame = false;

        {
            std::unique_lock<std::mutex> lock(inferenceMutex);
            if (!frameReady && !shouldExit)
                inferenceCV.wait(lock, [this] { return frameReady || shouldExit; });

            if (shouldExit) break;

            if (frameReady)
            {
                frame = std::move(currentFrame);
                frameReady = false;
                hasNewFrame = true;
            }
        }

        if (!context)
        {
            if (!error_logged)
            {
                std::cerr << "[Detector] Context not initialized" << std::endl;
                error_logged = true;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            continue;
        }
        else
        {
            error_logged = false;
        }

        if (hasNewFrame && !frame.empty())
        {
            try
            {
                // PRE
                auto t0 = std::chrono::steady_clock::now();
                preProcess(frame);
                auto t1 = std::chrono::steady_clock::now();

                // INFER
                context->enqueueV3(stream);

                // Событие окончания инференса (до копий)
                cudaEvent_t inferDone;
                cudaEventCreateWithFlags(&inferDone, cudaEventDisableTiming);
                cudaEventRecord(inferDone, stream);

                // D2H копии (асинхронно) — в pinned буферы
                auto t_copy_start = std::chrono::steady_clock::now();
                for (const auto& name : outputNames)
                {
                    auto itD = outputBindings.find(name);
                    auto itH = pinnedOutputBuffers.find(name);
                    if (itD == outputBindings.end() || itH == pinnedOutputBuffers.end()) continue;

                    size_t size = outputSizes[name];
                    cudaMemcpyAsync(
                        itH->second,             // host pinned
                        itD->second,             // device
                        size,
                        cudaMemcpyDeviceToHost,
                        stream);
                }

                // Событие окончания копий
                cudaEvent_t copiesDone;
                cudaEventCreateWithFlags(&copiesDone, cudaEventDisableTiming);
                cudaEventRecord(copiesDone, stream);

                // Ждём завершения инференса (метрика inferenceTime без учета копий)
                cudaEventSynchronize(inferDone);
                auto t2 = std::chrono::steady_clock::now();
                lastInferenceTime = t2 - t1;
                cudaEventDestroy(inferDone);

                // Ждём завершения копий
                cudaEventSynchronize(copiesDone);
                auto t_copy_end = std::chrono::steady_clock::now();
                lastCopyTime = t_copy_end - t_copy_start;
                cudaEventDestroy(copiesDone);

                // POST — читаем из pinnedOutputBuffers
                for (const auto& name : outputNames)
                {
                    size_t size = outputSizes[name];
                    nvinfer1::DataType dtype = outputTypes[name];

                    auto t_post_start = std::chrono::steady_clock::now();

                    if (dtype == nvinfer1::DataType::kHALF)
                    {
                        // конверт kHALF -> float на CPU (просто, можно ускорить позже)
                        size_t num = size / sizeof(__half);
                        const __half* hptr = static_cast<const __half*>(pinnedOutputBuffers[name]);
                        std::vector<float> tmp(num);
                        for (size_t i = 0; i < num; ++i) tmp[i] = __half2float(hptr[i]);

                        postProcess(tmp.data(), name, &lastNmsTime);
                    }
                    else // kFLOAT
                    {
                        const float* fptr = static_cast<const float*>(pinnedOutputBuffers[name]);
                        postProcess(fptr, name, &lastNmsTime);
                    }

                    auto t_post_end = std::chrono::steady_clock::now();
                    lastPostprocessTime = t_post_end - t_post_start;
                }

                lastPreprocessTime = t1 - t0;
            }
            catch (const std::exception& e)
            {
                std::cerr << "[Detector] Error during inference: " << e.what() << std::endl;
            }
        }
    }
}

std::vector<std::vector<Detection>> TrtDetector::detectBatch(const std::vector<cv::Mat>& frames)
{
    std::vector<std::vector<Detection>> batchDetections;
    if (frames.empty() || !context) return batchDetections;

    int batch_size = static_cast<int>(frames.size());

    nvinfer1::Dims dims = context->getTensorShape(inputName.c_str());
    int c = dims.d[1];
    int h = dims.d[2];
    int w = dims.d[3];

    if (dims.d[0] != batch_size)
    {
        context->setInputShape(inputName.c_str(), nvinfer1::Dims4{ batch_size, c, h, w });
    }

    std::vector<float> batchInput(batch_size * c * h * w);

    for (int b = 0; b < batch_size; ++b)
    {
        cv::Mat resized;
        cv::resize(frames[b], resized, cv::Size(w, h));
        resized.convertTo(resized, CV_32FC3, 1.0 / 255.0);
        std::vector<cv::Mat> channels;
        cv::split(resized, channels);

        for (int ch = 0; ch < c; ++ch)
        {
            float* dst = batchInput.data() + b * c * h * w + ch * h * w;
            memcpy(dst, channels[ch].ptr<float>(), h * w * sizeof(float));
        }
    }

    cudaMemcpy(inputBindings[inputName], batchInput.data(), batchInput.size() * sizeof(float), cudaMemcpyHostToDevice);

    context->enqueueV3(stream);
    cudaStreamSynchronize(stream);

    std::vector<float> output;
    const auto& outName = outputNames[0];
    size_t outputElements = outputSizes[outName] / sizeof(float);
    output.resize(outputElements);
    cudaMemcpy(output.data(), outputBindings[outName], outputSizes[outName], cudaMemcpyDeviceToHost);

    const std::vector<int64_t>& shape = outputShapes[outName];
    int batch_out = static_cast<int>(shape[0]);
    int rows = static_cast<int>(shape[1]);
    int cols = static_cast<int>(shape[2]);

    for (int b = 0; b < batch_out; ++b)
    {
        const float* out_ptr = output.data() + b * rows * cols;
        std::vector<Detection> detections;

        if (config.postprocess == "yolo10")
        {
            std::vector<int64_t> shape = { batch_out, rows, cols };
            detections = postProcessYolo10(
                out_ptr,
                shape,
                numClasses,
                config.confidence_threshold,
                config.nms_threshold,
                &lastNmsTime
            );
        }
        else if (
            config.postprocess == "yolo8" ||
            config.postprocess == "yolo9" ||
            config.postprocess == "yolo11" ||
            config.postprocess == "yolo12"
        )
        {
            std::vector<int64_t> shape = { rows, cols };
            detections = postProcessYolo11(
                out_ptr,
                shape,
                numClasses,
                config.confidence_threshold,
                config.nms_threshold,
                &lastNmsTime
            );
        }

        batchDetections.push_back(std::move(detections));
    }

    return batchDetections;
}

//void TrtDetector::preProcess(const cv::Mat& frame)
//{
//    if (frame.empty()) return;
//
//    void* inputBuffer = inputBindings[inputName];
//    if (!inputBuffer) return;
//
//    nvinfer1::Dims dims = context->getTensorShape(inputName.c_str());
//    int c = dims.d[1];
//    int h = dims.d[2];
//    int w = dims.d[3];
//
//    cv::cuda::GpuMat gpuFrame, gpuResized, gpuFloat;
//    gpuFrame.upload(frame);
//
//    if (frame.channels() == 4)
//        cv::cuda::cvtColor(gpuFrame, gpuFrame, cv::COLOR_BGRA2BGR);
//    else if (frame.channels() == 1)
//        cv::cuda::cvtColor(gpuFrame, gpuFrame, cv::COLOR_GRAY2BGR);
//
//    cv::cuda::resize(gpuFrame, gpuResized, cv::Size(w, h));
//    gpuResized.convertTo(gpuFloat, CV_32FC3, 1.0 / 255.0);
//
//    std::vector<cv::cuda::GpuMat> gpuChannels;
//    cv::cuda::split(gpuFloat, gpuChannels);
//
//    for (int i = 0; i < c; ++i)
//        cudaMemcpy((float*)inputBuffer + i * h * w, gpuChannels[i].ptr<float>(), h * w * sizeof(float), cudaMemcpyDeviceToDevice);
//}

void TrtDetector::preProcess(const cv::Mat& frame)
{
    if (frame.empty()) return;

    void* inputBuffer = inputBindings[inputName];
    if (!inputBuffer) return;

    // Запрашиваем форму входа у контекста
    nvinfer1::Dims dims = context->getTensorShape(inputName.c_str());
    const int c = dims.d[1];
    const int h = dims.d[2];
    const int w = dims.d[3];
    (void)c; // на случай, если компилятор ворчит

    // Тот же CUDA stream, что и у TensorRT
    cv::cuda::Stream cvStream = cv::cuda::StreamAccessor::wrapStream(stream);

    // Готовим RGB8 на GPU
    cv::cuda::GpuMat gpuRGB;

    if (frame.channels() == 4)
    {
        cv::cuda::GpuMat gpuBGRA(frame.size(), CV_8UC4);
        gpuBGRA.upload(frame, cvStream);
        cv::cuda::cvtColor(gpuBGRA, gpuRGB, cv::COLOR_BGRA2RGB, 0, cvStream);
    }
    else if (frame.channels() == 3)
    {
        cv::cuda::GpuMat gpuBGR(frame.size(), CV_8UC3);
        gpuBGR.upload(frame, cvStream);
        cv::cuda::cvtColor(gpuBGR, gpuRGB, cv::COLOR_BGR2RGB, 0, cvStream);
    }
    else if (frame.channels() == 1)
    {
        cv::cuda::GpuMat gpuGray(frame.size(), CV_8UC1);
        gpuGray.upload(frame, cvStream);
        cv::cuda::cvtColor(gpuGray, gpuRGB, cv::COLOR_GRAY2RGB, 0, cvStream);
    }
    else
    {
        // Неизвестное число каналов
        return;
    }

    // Шаг в "пикселях" для uchar3
    const int in_pitch_pixels = static_cast<int>(gpuRGB.step) / static_cast<int>(sizeof(uchar3));

    // Нормализация — по умолчанию просто /255
    float mean[3] = { 0.f, 0.f, 0.f };
    float std_[3] = { 1.f, 1.f, 1.f };
    // Если у тебя есть значения mean/std в конфиге — подставь здесь
    // mean[0] = config.mean[0]; ...  std_[0] = config.std[0]; ...

    // Фьюзим: letterbox/resize + to float + normalize + CHW
    launch_fused_preprocess(
        reinterpret_cast<const uchar3*>(gpuRGB.ptr<uchar3>()),
        gpuRGB.cols, gpuRGB.rows, in_pitch_pixels,
        static_cast<float*>(inputBuffer),  // CHW float на device
        w, h,
        mean, std_,
        stream);

    // Больше никаких resize/convertTo/split/memcpy не нужно
}


void TrtDetector::postProcess(const float* output, const std::string& outputName, std::chrono::duration<double, std::milli>* nmsTime)
{
    if (numClasses <= 0) return;

    std::vector<Detection> detections;

    if (config.postprocess == "yolo10")
    {
        const std::vector<int64_t>& shape = outputShapes[outputName];
        detections = postProcessYolo10(
            output,
            shape,
            numClasses,
            config.confidence_threshold,
            config.nms_threshold,
            nmsTime
        );
    }
    else if(
        config.postprocess == "yolo8" ||
        config.postprocess == "yolo9" ||
        config.postprocess == "yolo11" ||
        config.postprocess == "yolo12"
        )
    {
        auto shape = context->getTensorShape(outputName.c_str());
        std::vector<int64_t> engineShape;
        for (int i = 0; i < shape.nbDims; ++i)
        {
            engineShape.push_back(shape.d[i]);
        }

        detections = postProcessYolo11(
            output,
            engineShape,
            numClasses,
            config.confidence_threshold,
            config.nms_threshold,
            nmsTime
        );
    }

    {
        std::lock_guard<std::mutex> lock(detectionBuffer.mutex);
        detectionBuffer.boxes.clear();
        detectionBuffer.classes.clear();

        for (const auto& det : detections)
        {
            detectionBuffer.boxes.push_back(det.box);
            detectionBuffer.classes.push_back(det.classId);
        }

        detectionBuffer.version++;
        detectionBuffer.cv.notify_all();
    }
}
#endif