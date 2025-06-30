// mouse.h
#ifndef MOUSE_H
#define MOUSE_H

#define WIN32_LEAN_AND_MEAN
#define _WINSOCKAPI_
#include <winsock2.h>
#include <Windows.h>

#include <mutex>
#include <atomic>
#include <chrono>
#include <vector>
#include <utility>
#include <queue>
#include <thread>
#include <condition_variable>
#include <cmath> // std::modf, std::cos, M_PI

#include "AimbotTarget.h"
#include "SerialConnection.h"
#include "Kmbox_b.h"
#include "KmboxNetConnection.h"
#include "ghub.h"
#include "MakcuConnection.h"
#include "HID/HIDConnection.h"
#include "HID/HIDConnectionV2.h"

class MouseThread
{
private:
    // Основные параметры
    double screen_width, screen_height;
    double prediction_interval;
    double fov_x, fov_y;
    double max_distance;
    double min_speed_multiplier, max_speed_multiplier;
    double center_x, center_y;
    bool   auto_shoot;
    float  bScope_multiplier;

    // Для предсказания
    double prev_x, prev_y;
    double prev_velocity_x, prev_velocity_y;
    std::chrono::time_point<std::chrono::steady_clock> prev_time;
    std::chrono::steady_clock::time_point last_target_time;
    std::atomic<bool> target_detected{ false };
    std::atomic<bool> mouse_pressed{ false };

    // Интерфейсы ввода
    SerialConnection* serial;
    KmboxConnection* kmbox;
    KmboxNetConnection* kmbox_net;
    GhubMouse* gHub;
    MakcuConnection* makcu;
    HIDConnection* hid;
    HidConnectionV2* arduinoHid;

    // Отправка движений
    void sendMovementToDriver(int dx, int dy);
    struct Move { int dx, dy; };
    std::queue<Move>            moveQueue;
    std::mutex                  queueMtx;
    std::condition_variable     queueCv;
    const size_t                queueLimit = 5;
    std::thread                 moveWorker;
    std::atomic<bool>           workerStop{ false };

    // Флаг выбора логики
    bool use_smoothing = { false };
    bool use_kalman{ false };

    // Для «ветра»
    bool   wind_mouse_enabled = true;
    double wind_G, wind_W, wind_M, wind_D;
    void   windMouseMoveRelative(int dx, int dy);

    // Расчёт базового движения и скорости
    std::pair<double, double> calc_movement(double tx, double ty);
    double calculate_speed_multiplier(double distance);

    // Хранение прогнозов
    std::vector<std::pair<double, double>> futurePositions;
    std::mutex                            futurePositionsMutex;

    void moveWorkerLoop();
    void queueMove(int dx, int dy);

    // Параметры для сглаживания
    int    smoothness{ 100 };
    double move_overflow_x{ 0.0 }, move_overflow_y{ 0.0 };

    double easeInOut(double t);
    std::pair<double, double> addOverflow(double dx, double dy,
        double& overflow_x, double& overflow_y);
    void moveMouseWithSmoothing(double targetX, double targetY);

    struct Kalman1D {
        double x{ 0 }, v{ 0 }, P{ 1 }, Q, R;
        Kalman1D(double processNoise, double measurementNoise)
            : Q(processNoise), R(measurementNoise) {
        }
        double update(double z, double dt) {
            x += v * dt;
            P += Q * dt;
            double K = P / (P + R);
            x += K * (z - x);
            P *= (1 - K);
            v = (1 - K) * v + K * ((z - x) / std::max(dt, 1e-8));
            return x;
        }
    };

    Kalman1D    kfX{ 0.01, 0.1 }, kfY{ 0.01, 0.1 };
    std::chrono::steady_clock::time_point prevKalmanTime{};
    bool        smoothingActive = false;
    int         smoothingSteps = 0;
    double      smoothingStartX = 0, smoothingStartY = 0;
    double      smoothingTargetX = 0, smoothingTargetY = 0;
    double      smoothingDurationMs = 0;
    std::chrono::steady_clock::time_point smoothingStartTime{};
    void moveMouseWithSmoothingKalma(double smoothX, double smoothY);

public:
    std::mutex input_method_mutex;

    MouseThread(
        int resolution,
        int fovX,
        int fovY,
        double minSpeedMultiplier,
        double maxSpeedMultiplier,
        double predictionInterval,
        bool auto_shoot,
        float bScope_multiplier,
        SerialConnection* serialConnection = nullptr,
        GhubMouse* gHubMouse = nullptr,
        KmboxConnection* kmboxConnection = nullptr,
        KmboxNetConnection* kmboxNetConnection = nullptr,
        MakcuConnection* makcu = nullptr,
        HIDConnection* hid = nullptr,
        HidConnectionV2* arduinoHid = nullptr
    );
    ~MouseThread();

    void updateConfig(
        int resolution,
        int fovX,
        int fovY,
        double minSpeedMultiplier,
        double maxSpeedMultiplier,
        double predictionInterval,
        bool auto_shoot,
        float bScope_multiplier
    );

    // Переключатель логики
    void setUseSmoothing(bool v) { use_smoothing = v; }
    bool isUsingSmoothing() const { return use_smoothing; }

    void setUseKalman(bool v) { use_kalman = v; }
    bool isUsingKalman() const { return use_kalman; }

    // Настройка «гладкости»
    void setSmoothness(int s) { smoothness = (s > 0 ? s : 1); }
    int  getSmoothness() const { return smoothness; }

    // API движения и кликов
    void moveMousePivot(double pivotX, double pivotY);
    std::pair<double, double> predict_target_position(double x, double y);
    void moveMouse(const AimbotTarget& target);
    void pressMouse(const AimbotTarget& target);
    void releaseMouse();
    void resetPrediction();
    void checkAndResetPredictions();
    bool check_target_in_scope(double x, double y,
        double w, double h, double reduction_factor);

    // Будущие позиции
    std::vector<std::pair<double, double>> predictFuturePositions(
        double pivotX, double pivotY, int frames);
    void storeFuturePositions(
        const std::vector<std::pair<double, double>>& pos);
    void clearFuturePositions();
    std::vector<std::pair<double, double>> getFuturePositions();

    // Сеттеры интерфейсов
    void setSerialConnection(SerialConnection* s);
    void setKmboxConnection(KmboxConnection* k);
    void setKmboxNetConnection(KmboxNetConnection* k);
    void setGHubMouse(GhubMouse* g);
    void setHidConnection(HIDConnection* h);
    void setMakcuConnection(MakcuConnection* m);
    void setHidConnectionV2(HidConnectionV2* a);

    void setSmoothnessValue(int value) { smoothness = value; }
    int getSmoothnessValue() const { return smoothness; }
    void moveMouseWithKalmanSmoothing(double targetX, double targetY);
    void setKalmanParams(double processNoise, double measurementNoise);

    void setTargetDetected(bool d) {
        target_detected.store(d);
    }
    void setLastTargetTime(
        const std::chrono::steady_clock::time_point& t)
    {
        last_target_time = t;
    }
};

#endif // MOUSE_H
