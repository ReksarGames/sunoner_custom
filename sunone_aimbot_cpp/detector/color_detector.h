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
    void setFrame(const cv::Mat& frameBGRorBGRA);

    void stop() {
        shouldExit.store(true);
        inferenceCV.notify_all();
    }

    std::atomic<bool> shouldExit{ false };
    std::condition_variable inferenceCV;

    float scanError = 0.1f;
private:
    void detectColors(const cv::Mat& frame);
    void postProcess(const std::vector<cv::Rect>& boxes);

    // === Управление ===
    std::vector<ColorRange> colorRanges;
    int erodeIter = 1;
    int dilateIter = 2;
    int minArea = 50;
    int tinyArea = 10;
    bool isOnlyTop = true;

    std::atomic<bool> frameReady{ false };
    cv::Mat currentFrame;
    std::mutex inferenceMutex;

    bool        debug_show_window{ true };
    bool        debug_show_fps{ true };
    std::string debug_window_name{ "ColorDetection Debug" };
    cv::Scalar  debug_bgr{ 0, 255, 0 }; // цвет отрисовки (B, G, R)

    // для расчёта FPS
    std::chrono::steady_clock::time_point dbg_prev_ts{};
    double      dbg_fps{ 0.0 };
};
