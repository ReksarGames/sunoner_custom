// virtual_camera.cpp
#include "virtual_camera.h"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <algorithm>

#ifdef _WIN32
#include <windows.h>
#endif

namespace {
    // Вспомогательные функции и константы:
    inline int even(int v) { return (v % 2 == 0) ? v : v + 1; }

    std::filesystem::path cacheFilePath() {
#ifdef _WIN32
        char buf[MAX_PATH] = {};
        GetModuleFileNameA(nullptr, buf, MAX_PATH);
        return std::filesystem::path(buf).parent_path() / "virtual_cameras_cache.txt";
#else
        return std::filesystem::current_path() / "virtual_cameras_cache.txt";
#endif
    }

    void ensureCacheDir() {
        auto p = cacheFilePath().parent_path();
        std::error_code ec;
        std::filesystem::create_directories(p, ec);
    }

    std::vector<std::string> LoadCamList() {
        ensureCacheDir();
        std::vector<std::string> out;
        std::ifstream ifs(cacheFilePath());
        std::string line;
        while (std::getline(ifs, line)) {
            auto l = line.find_first_not_of(" \t\r\n");
            auto r = line.find_last_not_of(" \t\r\n");
            if (l != std::string::npos && r != std::string::npos)
                out.emplace_back(line.substr(l, r - l + 1));
        }
        return out;
    }

    void SaveCamList(const std::vector<std::string>& cams) {
        ensureCacheDir();
        std::ofstream ofs(cacheFilePath(), std::ios::trunc);
        for (auto& c : cams) ofs << c << "\n";
    }
} // namespace

// === Вот именно здесь нужны реализации методов класса ===

// Реализация статического метода CamCache()
std::vector<std::string>& VirtualCameraCapture::CamCache() {
    static std::vector<std::string> cache = LoadCamList();
    return cache;
}

VirtualCameraCapture::VirtualCameraCapture(int w, int h) {
    auto cams = GetAvailableVirtualCameras();
    if (cams.empty())
        throw std::runtime_error("[VirtualCamera] No devices");

    auto sel = cams[0];
    auto sep = sel.find('|');
    auto api = sel.substr(0, sep);
    auto idx = std::stoi(sel.substr(sep + 1));
    int backend = (api == "DSHOW") ? cv::CAP_DSHOW : cv::CAP_MSMF;

    cap_ = std::make_unique<cv::VideoCapture>(idx, backend);
    if (!cap_->isOpened())
        throw std::runtime_error("[VirtualCamera] Can't open " + sel);

    if (w > 0 && h > 0) {
        cap_->set(cv::CAP_PROP_FRAME_WIDTH, even(w));
        cap_->set(cv::CAP_PROP_FRAME_HEIGHT, even(h));
    }
    roiW_ = even((int)cap_->get(cv::CAP_PROP_FRAME_WIDTH));
    roiH_ = even((int)cap_->get(cv::CAP_PROP_FRAME_HEIGHT));
    cap_->set(cv::CAP_PROP_BUFFERSIZE, 1);
}

VirtualCameraCapture::~VirtualCameraCapture() {
    if (cap_ && cap_->isOpened()) cap_->release();
}

cv::Mat VirtualCameraCapture::GetNextFrameCpu() {
    if (!cap_ || !cap_->isOpened()) return {};
    cv::Mat f;
    if (!cap_->read(f) || f.empty()) return {};
    switch (f.channels()) {
    case 1:  cv::cvtColor(f, f, cv::COLOR_GRAY2BGR); break;
    case 4:  cv::cvtColor(f, f, cv::COLOR_BGRA2BGR); break;
    case 3:  break;
    default: std::cerr << "[VC] ch=" << f.channels() << "\n"; return {};
    }
    return f.clone();
}

std::vector<std::string> VirtualCameraCapture::GetAvailableVirtualCameras(bool forceRescan) {
    auto& cache = CamCache();
    if (!forceRescan && !cache.empty()) return cache;

    cache.clear();
    const std::vector<std::pair<int, std::string>> backends = {
        {cv::CAP_DSHOW,"DSHOW"},{cv::CAP_MSMF,"MSMF"}
    };
    for (auto& [api, name] : backends) {
        for (int i = 0; i < 10; ++i) {
            cv::VideoCapture t(i, api);
            if (t.isOpened()) cache.emplace_back(name + "|" + std::to_string(i));
        }
    }
    std::sort(cache.begin(), cache.end());
    cache.erase(std::unique(cache.begin(), cache.end()), cache.end());
    SaveCamList(cache);
    return cache;
}

void VirtualCameraCapture::ClearCachedCameraList() {
    CamCache().clear();
    std::error_code ec;
    std::filesystem::remove(cacheFilePath(), ec);
}
