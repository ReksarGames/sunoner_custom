#include "color_detector.h"
#include "sunone_aimbot_cpp.h"

ColorDetector::ColorDetector()
    : erodeIter(1), dilateIter(2), minArea(50),
    shouldExit(false), frameReady(false) {
}

ColorDetector::~ColorDetector() {
    stop();
}

void ColorDetector::initializeFromConfig(const Config& cfg) {
    colorRanges.clear();

    // Если выбран конкретный цвет — берём его
    for (const auto& cr : cfg.color_ranges) {
        if (cr.name == cfg.color_target) {
            colorRanges.push_back({
                cr.name,
                cv::Scalar(cr.h_low, cr.s_low, cr.v_low),
                cv::Scalar(cr.h_high, cr.s_high, cr.v_high)
                });
            break;
        }
    }

    // Если не нашли — fallback: берём все
    if (colorRanges.empty()) {
        for (const auto& cr : cfg.color_ranges) {
            colorRanges.push_back({
                cr.name,
                cv::Scalar(cr.h_low, cr.s_low, cr.v_low),
                cv::Scalar(cr.h_high, cr.s_high, cr.v_high)
                });
        }
    }

    erodeIter = cfg.color_erode_iter;
    dilateIter = cfg.color_dilate_iter;
    minArea = cfg.color_min_area;
}

void ColorDetector::processFrame(const cv::Mat& frame) {
    std::unique_lock<std::mutex> lock(inferenceMutex);
    frame.copyTo(currentFrame);
    frameReady = true;
    inferenceCV.notify_one();
}

void ColorDetector::inferenceThread() {
    while (!shouldExit) {
        cv::Mat frame;

        {
            std::unique_lock<std::mutex> lock(inferenceMutex);
            inferenceCV.wait(lock, [this] { return frameReady || shouldExit; });

            if (shouldExit) break;

            if (frameReady) {
                currentFrame.copyTo(frame);
                frameReady = false;
            }
        }

        if (!frame.empty()) {
            detectColors(frame);
        }
    }
}

void ColorDetector::detectColors(const cv::Mat& frame) {
    cv::Mat hsv;
    cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV);

    cv::Mat combinedMask = cv::Mat::zeros(frame.size(), CV_8UC1);

    // Собираем все выбранные диапазоны
    for (const auto& range : colorRanges) {
        cv::Mat mask;
        cv::inRange(hsv, range.lower, range.upper, mask);
        combinedMask |= mask;
    }

    // Морфология (шумоподавление)
    cv::erode(combinedMask, combinedMask, cv::Mat(), cv::Point(-1, -1), erodeIter);
    cv::dilate(combinedMask, combinedMask, cv::Mat(), cv::Point(-1, -1), dilateIter);

    // Контуры
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(combinedMask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    std::vector<cv::Rect> boxes;
    for (auto& c : contours) {
        double area = cv::contourArea(c);
        if (area < minArea) continue;
        boxes.push_back(cv::boundingRect(c));
    }

    postProcess(boxes);
}

void ColorDetector::postProcess(const std::vector<cv::Rect>& boxes) {
    std::lock_guard<std::mutex> lock(detectionBuffer.mutex);
    detectionBuffer.boxes.clear();
    detectionBuffer.classes.clear();

    for (const auto& box : boxes) {
        detectionBuffer.boxes.push_back(box);
        detectionBuffer.classes.push_back(0);
    }

    detectionBuffer.version++;
    detectionBuffer.cv.notify_all();
}
