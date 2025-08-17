#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <atomic>
#include <mutex>
#include <condition_variable>

#include "config/config.h"

struct ColorRange {
    std::string name;
    cv::Scalar lower;
    cv::Scalar upper;
};

class ColorDetector {
public:
    ColorDetector();
    ~ColorDetector();

    void initializeFromConfig(const Config& cfg);

    void processFrame(const cv::Mat& frame);
    void inferenceThread();

    void stop() {
        shouldExit.store(true);
        inferenceCV.notify_all();
    }

    std::atomic<bool> shouldExit{ false };
    std::condition_variable inferenceCV;

private:
    void detectColors(const cv::Mat& frame);
    void postProcess(const std::vector<cv::Rect>& boxes);

    std::vector<ColorRange> colorRanges;
    int erodeIter = 1;
    int dilateIter = 2;
    int minArea = 50;

    std::atomic<bool> frameReady{ false };
    cv::Mat currentFrame;
    std::mutex inferenceMutex;
};
