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

#include <algorithm> // std::clamp

void ColorDetector::detectColors(const cv::Mat& frame) {
    // Преобразуем изображение в формат BGR (если оно в формате BGRA)
    cv::Mat bgr;
    if (frame.channels() == 4) {
        cv::cvtColor(frame, bgr, cv::COLOR_BGRA2BGR);
    }
    else {
        bgr = frame;
    }

    // Преобразуем изображение в цветовую модель HSV
    cv::Mat hsv;
    cv::cvtColor(bgr, hsv, cv::COLOR_BGR2HSV);

    // Создаем пустую маску для всех выбранных цветовых диапазонов
    cv::Mat combinedMask = cv::Mat::zeros(bgr.size(), CV_8UC1);

    // Собираем все выбранные диапазоны
    for (const auto& range : colorRanges) {
        cv::Mat mask;
        cv::inRange(hsv, range.lower, range.upper, mask); // Находим пиксели, попадающие в диапазон
        combinedMask |= mask; // Объединяем маски (сейчас все диапазоны объединены в один)
    }

    // Немного расширим пятна, чтобы мелкие объекты не пропали
    cv::dilate(combinedMask, combinedMask, cv::Mat(), cv::Point(-1, -1), 1);

    // Находим контуры на маске
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(combinedMask.clone(), contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // Вектор для хранения найденных прямоугольников
    std::vector<cv::Rect> boxes;

    // Проходим по всем контурам
    for (auto& c : contours) {
        double area = cv::contourArea(c); // Вычисляем площадь контура

        // Если площадь достаточна и объект не слишком маленький
        if (area >= minArea) {
            cv::Rect boundingBox = cv::boundingRect(c); // Получаем ограничивающий прямоугольник

            // Пропускаем слишком маленькие объекты
            if (area < tinyArea) {
                continue;
            }

            boxes.push_back(boundingBox); // Добавляем прямоугольник в список
        }
    }

    // Обрабатываем результаты
    postProcess(boxes);

    // === DEBUG WINDOW ===
    if (debug_show_window) {
        cv::Mat dbg = bgr.clone();

        // Рисуем контуры на изображении для отладки
        cv::drawContours(dbg, contours, -1, { 0, 0, 255 }, 1);

        // Рисуем прямоугольники для каждого найденного объекта
        for (auto& b : boxes) {
            cv::rectangle(dbg, b, debug_bgr, 2);
        }

        // Показываем FPS
        if (debug_show_fps) {
            auto now = std::chrono::steady_clock::now();
            double dt = std::chrono::duration<double>(now - dbg_prev_ts).count();
            dbg_prev_ts = now;
            if (dt > 0.0) dbg_fps = 1.0 / dt;
            char buf[64];
            snprintf(buf, sizeof(buf), "FPS: %.1f", dbg_fps);
            cv::putText(dbg, buf, { 10, 30 }, cv::FONT_HERSHEY_SIMPLEX, 0.8, debug_bgr, 2);
        }

        // Показываем изображение с отрисованными результатами
        cv::imshow(debug_window_name, dbg);
        cv::waitKey(1);
    }
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
