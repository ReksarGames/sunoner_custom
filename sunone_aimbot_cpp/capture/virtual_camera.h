// virtual_camera.h
#ifndef VIRTUAL_CAMERA_H
#define VIRTUAL_CAMERA_H

#include <memory>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include "capture.h"   // <-- здесь объявлен IScreenCapture

class VirtualCameraCapture final : public IScreenCapture {
public:
    VirtualCameraCapture(int width, int height);
    ~VirtualCameraCapture() override;

    // возвращает следующий BGR-кадр
    cv::Mat GetNextFrameCpu() override;

    // Список доступных «виртуалок» вида "DSHOW|0", "MSMF|1" и т.п.
    static std::vector<std::string> GetAvailableVirtualCameras(bool forceRescan = false);

    // Очистить кэш-файл
    static void ClearCachedCameraList();

private:
    std::unique_ptr<cv::VideoCapture> cap_;
    int roiW_{ 0 }, roiH_{ 0 };
    cv::Mat frameCpu_;

    static std::vector<std::string>& CamCache();
};

#endif // VIRTUAL_CAMERA_H
