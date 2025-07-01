#define WIN32_LEAN_AND_MEAN
#define _WINSOCKAPI_
#include <winsock2.h>
#include <Windows.h>

#define _USE_MATH_DEFINES
#include <cmath>
#include <algorithm>
#include <chrono>
#include <mutex>
#include <atomic>
#include <vector>

#include "mouse.h"
#include "capture.h"
#include "SerialConnection.h"
#include "sunone_aimbot_cpp.h"
#include "ghub.h"

MouseThread::MouseThread(
    int resolution,
    int fovX,
    int fovY,
    double minSpeedMultiplier,
    double maxSpeedMultiplier,
    double predictionInterval,
    bool auto_shoot,
    float bScope_multiplier,
    SerialConnection* serialConnection,
    GhubMouse* gHubMouse,
    KmboxConnection* kmboxConnection,
    KmboxNetConnection* Kmbox_Net_Connection,
    MakcuConnection* makcu, HIDConnection* hid, HidConnectionV2* arduinoHid)
    : screen_width(resolution),
    screen_height(resolution),
    prediction_interval(predictionInterval),
    fov_x(fovX),
    fov_y(fovY),
    max_distance(std::hypot(resolution, resolution) / 2.0),
    min_speed_multiplier(minSpeedMultiplier),
    max_speed_multiplier(maxSpeedMultiplier),
    center_x(resolution / 2.0),
    center_y(resolution / 2.0),
    auto_shoot(auto_shoot),
    bScope_multiplier(bScope_multiplier),
    serial(serialConnection),
    kmbox(kmboxConnection),
    kmbox_net(Kmbox_Net_Connection),
    gHub(gHubMouse),
    makcu(makcu),
    hid(hid),
    arduinoHid(arduinoHid),

    prev_velocity_x(0.0),
    prev_velocity_y(0.0),
    prev_x(0.0),
    prev_y(0.0)
{
    prev_time = std::chrono::steady_clock::time_point();
    last_target_time = std::chrono::steady_clock::now();

    wind_mouse_enabled = config.wind_mouse_enabled;
    wind_G = config.wind_G;
    wind_W = config.wind_W;
    wind_M = config.wind_M;
    wind_D = config.wind_D;

    use_smoothing = config.use_smoothing;
    use_kalman = config.use_kalman;

    kfX = Kalman1D(config.kalman_process_noise, config.kalman_measurement_noise);
    kfY = Kalman1D(config.kalman_process_noise, config.kalman_measurement_noise);

    moveWorker = std::thread(&MouseThread::moveWorkerLoop, this);
}

void MouseThread::updateConfig(
    int resolution,
    int fovX,
    int fovY,
    double minSpeedMultiplier,
    double maxSpeedMultiplier,
    double predictionInterval,
    bool auto_shoot,
    float bScope_multiplier
)
{
    screen_width = screen_height = resolution;
    fov_x = fovX;  fov_y = fovY;
    min_speed_multiplier = minSpeedMultiplier;
    max_speed_multiplier = maxSpeedMultiplier;
    prediction_interval = predictionInterval;
    this->auto_shoot = auto_shoot;
    this->bScope_multiplier = bScope_multiplier;

    center_x = center_y = resolution / 2.0;
    max_distance = std::hypot(resolution, resolution) / 2.0;

    wind_mouse_enabled = config.wind_mouse_enabled;
    wind_G = config.wind_G; wind_W = config.wind_W;
    wind_M = config.wind_M; wind_D = config.wind_D;

    use_smoothing = config.use_smoothing;
    use_kalman = config.use_kalman;

    kfX = Kalman1D(config.kalman_process_noise, config.kalman_measurement_noise);
    kfY = Kalman1D(config.kalman_process_noise, config.kalman_measurement_noise);

}

MouseThread::~MouseThread()
{
    workerStop = true;
    queueCv.notify_all();
    if (moveWorker.joinable()) moveWorker.join();
}

void MouseThread::queueMove(int dx, int dy)
{
    std::lock_guard lg(queueMtx);
    if (moveQueue.size() >= queueLimit) moveQueue.pop();
    moveQueue.push({ dx,dy });
    queueCv.notify_one();
}

void MouseThread::moveWorkerLoop()
{
    while (!workerStop)
    {
        std::unique_lock ul(queueMtx);
        queueCv.wait(ul, [&] { return workerStop || !moveQueue.empty(); });

        while (!moveQueue.empty())
        {
            Move m = moveQueue.front();
            moveQueue.pop();
            ul.unlock();
            sendMovementToDriver(m.dx, m.dy);
            ul.lock();
        }
    }
}

void MouseThread::windMouseMoveRelative(int dx, int dy)
{
    if (dx == 0 && dy == 0) return;

    constexpr double SQRT3 = 1.7320508075688772;
    constexpr double SQRT5 = 2.23606797749979;

    double sx = 0, sy = 0;
    double dxF = static_cast<double>(dx);
    double dyF = static_cast<double>(dy);
    double vx = 0, vy = 0, wX = 0, wY = 0;
    int    cx = 0, cy = 0;

    while (std::hypot(dxF - sx, dyF - sy) >= 1.0)
    {
        double dist = std::hypot(dxF - sx, dyF - sy);
        double wMag = std::min(wind_W, dist);

        if (dist >= wind_D)
        {
            wX = wX / SQRT3 + ((double)rand() / RAND_MAX * 2.0 - 1.0) * wMag / SQRT5;
            wY = wY / SQRT3 + ((double)rand() / RAND_MAX * 2.0 - 1.0) * wMag / SQRT5;
        }
        else
        {
            wX /= SQRT3;  wY /= SQRT3;
            wind_M = wind_M < 3.0 ? ((double)rand() / RAND_MAX) * 3.0 + 3.0 : wind_M / SQRT5;
        }

        vx += wX + wind_G * (dxF - sx) / dist;
        vy += wY + wind_G * (dyF - sy) / dist;

        double vMag = std::hypot(vx, vy);
        if (vMag > wind_M)
        {
            double vClip = wind_M / 2.0 + ((double)rand() / RAND_MAX) * wind_M / 2.0;
            vx = (vx / vMag) * vClip;
            vy = (vy / vMag) * vClip;
        }

        sx += vx;  sy += vy;
        int rx = static_cast<int>(std::round(sx));
        int ry = static_cast<int>(std::round(sy));
        int step_x = rx - cx;
        int step_y = ry - cy;
        if (step_x || step_y)
        {
            queueMove(step_x, step_y);
            cx = rx; cy = ry;
        }
    }
}

std::pair<double, double> MouseThread::predict_target_position(double target_x, double target_y)
{
    auto current_time = std::chrono::steady_clock::now();

    if (prev_time.time_since_epoch().count() == 0 || !target_detected.load())
    {
        prev_time = current_time;
        prev_x = target_x;
        prev_y = target_y;
        prev_velocity_x = 0.0;
        prev_velocity_y = 0.0;
        return { target_x, target_y };
    }

    double dt = std::chrono::duration<double>(current_time - prev_time).count();
    if (dt < 1e-8) dt = 1e-8;

    double vx = (target_x - prev_x) / dt;
    double vy = (target_y - prev_y) / dt;

    vx = std::clamp(vx, -20000.0, 20000.0);
    vy = std::clamp(vy, -20000.0, 20000.0);

    prev_time = current_time;
    prev_x = target_x;
    prev_y = target_y;
    prev_velocity_x = vx;
    prev_velocity_y = vy;

    double predictedX = target_x + vx * prediction_interval;
    double predictedY = target_y + vy * prediction_interval;

    double detectionDelay = 0.05;
    if (config.backend == "DML")
    {
        detectionDelay = dml_detector->lastInferenceTimeDML.count();
    }
#ifdef USE_CUDA
    else
    {
        detectionDelay = trt_detector.lastInferenceTime.count();
    }
#endif
    predictedX += vx * detectionDelay;
    predictedY += vy * detectionDelay;

    return { predictedX, predictedY };
}

void MouseThread::sendMovementToDriver(int dx, int dy)
{
    std::lock_guard<std::mutex> lock(input_method_mutex);
    if (makcu)
    {
        makcu->move(dx, dy);
    }
    else if (hid)
    {
        hid->move(dx, dy);
    }
    else if (arduinoHid)
    {
        arduinoHid->move(dx, dy);
    }
    else if (kmbox)
    {
        kmbox->move(dx, dy);
    }
    else if (kmbox_net)
    {
        kmbox_net->move(dx, dy);
    }
    else if (serial)
    {
        serial->move(dx, dy);
    }
    else if (gHub)
    {
        gHub->mouse_xy(dx, dy);
    }
    else
    {
        INPUT in{ 0 };
        in.type = INPUT_MOUSE;
        in.mi.dx = dx;  in.mi.dy = dy;
        in.mi.dwFlags = MOUSEEVENTF_MOVE | MOUSEEVENTF_VIRTUALDESK;
        SendInput(1, &in, sizeof(INPUT));
    }
}

std::pair<double, double> MouseThread::calc_movement(double tx, double ty)
{
    double offx = tx - center_x;
    double offy = ty - center_y;
    double dist = std::hypot(offx, offy);
    double speed = calculate_speed_multiplier(dist);

    double degPerPxX = fov_x / screen_width;
    double degPerPxY = fov_y / screen_height;

    double mmx = offx * degPerPxX;
    double mmy = offy * degPerPxY;

    double corr = 1.0;
    double fps = static_cast<double>(captureFps.load());
    if (fps > 30.0) corr = 30.0 / fps;

    auto counts_pair = config.degToCounts(mmx, mmy, fov_x);
    double move_x = counts_pair.first * speed * corr;
    double move_y = counts_pair.second * speed * corr;

    return { move_x, move_y };
}

double MouseThread::calculate_speed_multiplier(double distance)
{
    if (distance < config.snapRadius)
        return min_speed_multiplier * config.snapBoostFactor;

    if (distance < config.nearRadius)
    {
        double t = distance / config.nearRadius;
        double curve = 1.0 - std::pow(1.0 - t, config.speedCurveExponent);
        return min_speed_multiplier +
            (max_speed_multiplier - min_speed_multiplier) * curve;
    }

    double norm = std::clamp(distance / max_distance, 0.0, 1.0);
    return min_speed_multiplier +
        (max_speed_multiplier - min_speed_multiplier) * norm;
}

bool MouseThread::check_target_in_scope(double target_x, double target_y, double target_w, double target_h, double reduction_factor)
{
    double center_target_x = target_x + target_w / 2.0;
    double center_target_y = target_y + target_h / 2.0;

    double reduced_w = target_w * (reduction_factor / 2.0);
    double reduced_h = target_h * (reduction_factor / 2.0);

    double x1 = center_target_x - reduced_w;
    double x2 = center_target_x + reduced_w;
    double y1 = center_target_y - reduced_h;
    double y2 = center_target_y + reduced_h;

    return (center_x > x1 && center_x < x2 && center_y > y1 && center_y < y2);
}

void MouseThread::pressMouse(const AimbotTarget& target)
{
    std::lock_guard<std::mutex> lock(input_method_mutex);

    bool bScope = check_target_in_scope(target.x, target.y, target.w, target.h, bScope_multiplier);
    if (bScope && !mouse_pressed)
    {
        if (makcu)
        {
            makcu->press(0);
        }
        else if (hid)
        {
            hid->press();
        }
        else if (arduinoHid)
        {
            arduinoHid->press();
        }
        else if (kmbox)
        {
            kmbox->press(0);
        }
        else if (kmbox_net)
        {
            kmbox_net->keyDown(0);
        }
        else if (serial)
        {
            serial->press();
        }
        else if (gHub)
        {
            gHub->mouse_down();
        }
        else
        {
            INPUT input = { 0 };
            input.type = INPUT_MOUSE;
            input.mi.dwFlags = MOUSEEVENTF_LEFTDOWN;
            SendInput(1, &input, sizeof(INPUT));
        }
        mouse_pressed = true;
    }
    else if (!bScope && mouse_pressed)
    {
        if (hid)
        {
            hid->release();
        }
        else if (makcu)
        {
            makcu->release(0);
        }
        else if (arduinoHid)
        {
            arduinoHid->release();
        }
        else if (kmbox)
        {
            kmbox->release(0);
        }
        else if (kmbox_net)
        {
            kmbox_net->keyUp(0);
        }
        else if (serial)
        {
            serial->release();
        }
        else if (gHub)
        {
            gHub->mouse_up();
        }
        else
        {
            INPUT input = { 0 };
            input.type = INPUT_MOUSE;
            input.mi.dwFlags = MOUSEEVENTF_LEFTUP;
            SendInput(1, &input, sizeof(INPUT));
        }
        mouse_pressed = false;
    }
}

void MouseThread::releaseMouse()
{
    std::lock_guard<std::mutex> lock(input_method_mutex);

    if (mouse_pressed)
    {
        if (hid)
        {
            hid->release();
        }
        else if (makcu)
        {
            makcu->release(0);
        }
        else if (arduinoHid)
        {
            arduinoHid->release();
        }
        else if (kmbox)
        {
            kmbox->release(0);
        }
        else if (kmbox_net)
        {
            kmbox_net->keyUp(0);
        }
        else if (serial)
        {
            serial->release();
        }
        else if (gHub)
        {
            gHub->mouse_up();
        }
        else
        {
            INPUT input = { 0 };
            input.type = INPUT_MOUSE;
            input.mi.dwFlags = MOUSEEVENTF_LEFTUP;
            SendInput(1, &input, sizeof(INPUT));
        }
        mouse_pressed = false;
    }
}

void MouseThread::resetPrediction()
{
    prev_time = std::chrono::steady_clock::time_point();
    prev_x = 0;
    prev_y = 0;
    prev_velocity_x = 0;
    prev_velocity_y = 0;
    target_detected.store(false);
}

void MouseThread::checkAndResetPredictions()
{
    auto current_time = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(current_time - last_target_time).count();

    if (elapsed > 0.5 && target_detected.load())
    {
        resetPrediction();
    }
}

std::vector<std::pair<double, double>> MouseThread::predictFuturePositions(double pivotX, double pivotY, int frames)
{
    std::vector<std::pair<double, double>> result;
    result.reserve(frames);

    const double fixedFps = 30.0;
    double frame_time = 1.0 / fixedFps;

    auto current_time = std::chrono::steady_clock::now();
    double dt = std::chrono::duration<double>(current_time - prev_time).count();

    if (prev_time.time_since_epoch().count() == 0 || dt > 0.5)
    {
        return result;
    }

    double vx = prev_velocity_x;
    double vy = prev_velocity_y;

    for (int i = 1; i <= frames; i++)
    {
        double t = frame_time * i;

        double px = pivotX + vx * t;
        double py = pivotY + vy * t;

        result.push_back({ px, py });
    }

    return result;
}

void MouseThread::storeFuturePositions(const std::vector<std::pair<double, double>>& positions)
{
    std::lock_guard<std::mutex> lock(futurePositionsMutex);
    futurePositions = positions;
}

void MouseThread::clearFuturePositions()
{
    std::lock_guard<std::mutex> lock(futurePositionsMutex);
    futurePositions.clear();
}

std::vector<std::pair<double, double>> MouseThread::getFuturePositions()
{
    std::lock_guard<std::mutex> lock(futurePositionsMutex);
    return futurePositions;
}

void MouseThread::setSerialConnection(SerialConnection* newSerial)
{
    std::lock_guard<std::mutex> lock(input_method_mutex);
    serial = newSerial;
}

void MouseThread::setKmboxConnection(KmboxConnection* newKmbox)
{
    std::lock_guard<std::mutex> lock(input_method_mutex);
    kmbox = newKmbox;
}

void MouseThread::setKmboxNetConnection(KmboxNetConnection* newKmbox_net)
{
    std::lock_guard<std::mutex> lock(input_method_mutex);
    kmbox_net = newKmbox_net;
}

void MouseThread::setGHubMouse(GhubMouse* newGHub)
{
    std::lock_guard<std::mutex> lock(input_method_mutex);
    gHub = newGHub;
}

void MouseThread::setHidConnection(HIDConnection* newHid)
{
    std::lock_guard<std::mutex> lock(input_method_mutex);
    hid = newHid;
}
void MouseThread::setMakcuConnection(MakcuConnection* newMakcu)
{
    std::lock_guard<std::mutex> lock(   input_method_mutex);
    makcu = newMakcu;
}
void MouseThread::setHidConnectionV2(HidConnectionV2* newArduinoHid)
{
    std::lock_guard<std::mutex> lock(input_method_mutex);
    arduinoHid = newArduinoHid;
}

void MouseThread::moveMouseWithKalmanSmoothing(double targetX, double targetY) {
    double rawX = targetX, rawY = targetY;
    static double lastRawX = 0.0, lastRawY = 0.0;
    static bool   firstCall = true;
    const double  resetThreshold = 2.0; // порог в пикселях, можно крутить в config

    // 0) Сброс при большом «прыжке» цели или первом вызове
    if (firstCall || std::hypot(rawX - lastRawX, rawY - lastRawY) > resetThreshold) {
        kfX.x = rawX;  kfX.v = 0.0;  kfX.P = 1.0;
        kfY.x = rawY;  kfY.v = 0.0;  kfY.P = 1.0;
        prevKalmanTime = std::chrono::steady_clock::now();
        firstCall = false;
    }
    lastRawX = rawX;
    lastRawY = rawY;

    // 1) dt
    auto now = std::chrono::steady_clock::now();
    double dt = prevKalmanTime.time_since_epoch().count() == 0
        ? 1.0 / static_cast<double>(config.capture_fps)
        : std::max(std::chrono::duration<double>(now - prevKalmanTime).count(), 1e-8);
    prevKalmanTime = now;

    // 2) Predict+Update
    double filtX = kfX.update(rawX, dt);
    double filtY = kfY.update(rawY, dt);

    // 3) В дельту мыши
    auto [mvX, mvY] = calc_movement(filtX, filtY);
    mvX *= kalman_speed_multiplier_x;
    mvY *= kalman_speed_multiplier_y;

    // 4) Отправка
    int dx = static_cast<int>(std::round(mvX));
    int dy = static_cast<int>(std::round(mvY));
    if (wind_mouse_enabled) windMouseMoveRelative(dx, dy);
    else                   queueMove(dx, dy);
}

// easing-функция для плавности
double MouseThread::easeInOut(double t) {
    return -0.5 * (std::cos(M_PI * t) - 1.0);
}

// Управление дробной частью, чтобы не терять пиксели
std::pair<double, double> MouseThread::addOverflow(
    double dx, double dy,
    double& overflow_x, double& overflow_y)
{
    double int_x = 0.0, int_y = 0.0;
    double frac_x = std::modf(dx + overflow_x, &int_x);
    double frac_y = std::modf(dy + overflow_y, &int_y);

    // Если дробная часть вдруг вышла за [-1;1], корректируем
    if (std::abs(frac_x) > 1.0) {
        double extra = 0.0;
        frac_x = std::modf(frac_x, &extra);
        int_x += extra;
    }
    if (std::abs(frac_y) > 1.0) {
        double extra = 0.0;
        frac_y = std::modf(frac_y, &extra);
        int_y += extra;
    }

    overflow_x = frac_x;
    overflow_y = frac_y;
    return { int_x, int_y };
}

void MouseThread::moveMouseWithSmoothingKalma(double smoothX, double smoothY)
{
    // TODO: Сделать после Калма
    return;
}

// «Микрошаговая» плавная наводка
void MouseThread::moveMouseWithSmoothing(double targetX, double targetY)
{
    if (smoothness <= 0) smoothness = 1;

    static double startX = 0.0, startY = 0.0;
    static double prevX = 0.0, prevY = 0.0;
    static double lastTX = 0.0, lastTY = 0.0;
    static int    frame = 0;

    std::lock_guard<std::mutex> lg(input_method_mutex);

    // Если цель сильно сместилась или это первый кадр — сброс
    if (frame == 0 || std::hypot(targetX - lastTX, targetY - lastTY) > 1.0) {
        startX = center_x;
        startY = center_y;
        prevX = startX;
        prevY = startY;
        frame = 0;
    }
    lastTX = targetX;
    lastTY = targetY;

    // Прокручиваем кадр сглаживания
    int N = smoothness;
    frame = std::min(frame + 1, N);
    double t = double(frame) / N;
    double p = easeInOut(t);

    double curX = startX + (targetX - startX) * p;
    double curY = startY + (targetY - startY) * p;

    double dx = curX - prevX;
    double dy = curY - prevY;

    auto mv = addOverflow(dx, dy, move_overflow_x, move_overflow_y);
    int ix = static_cast<int>(mv.first);
    int iy = static_cast<int>(mv.second);
    if (ix || iy) queueMove(ix, iy);

    prevX = curX;
    prevY = curY;
}

void MouseThread::setKalmanParams(double processNoise, double measurementNoise) {
    std::lock_guard<std::mutex> lg(input_method_mutex);
    // просто перезапускаем фильтры с новыми Q/R
    kfX = Kalman1D(processNoise, measurementNoise);
    kfY = Kalman1D(processNoise, measurementNoise);
}

// Основной метод наведения
void MouseThread::moveMouse(const AimbotTarget& target)
{
    // 1) DT для Калмана
    auto now = std::chrono::steady_clock::now();
    double dt;
    if (prevKalmanTime.time_since_epoch().count() == 0) {
        dt = 1.0 / static_cast<double>(config.capture_fps);
    }
    else {
        dt = std::chrono::duration<double>(now - prevKalmanTime).count();
        dt = std::max(dt, 1e-8);
    }
    prevKalmanTime = now;

    // 2) Сырые предсказанные координаты центра цели
    double rawX = target.x + target.w * 0.5;
    double rawY = target.y + target.h * 0.5;
    auto [predX, predY] = predict_target_position(rawX, rawY);

    // 3) Ветвление по режимам
    if (use_kalman && !use_smoothing) {
        // === Только Калман-фильтр ===
        moveMouseWithKalmanSmoothing(predX, predY);
    }
    else if (!use_kalman && use_smoothing) {
        // === Только easing-сглаживание ===
        moveMouseWithSmoothing(predX, predY);
    }
    else if (use_kalman && use_smoothing) {
        return;
    }
    else {
        // === Старая механика «без всего» ===
        auto [mvX, mvY] = calc_movement(predX, predY);
        if (wind_mouse_enabled)
            windMouseMoveRelative(static_cast<int>(mvX), static_cast<int>(mvY));
        else
            queueMove(static_cast<int>(std::round(mvX)),
                static_cast<int>(std::round(mvY)));
    }
}
// Аналогично для moveMousePivot
void MouseThread::moveMousePivot(double pivotX, double pivotY)
{
    auto now = std::chrono::steady_clock::now();

    // 1) Обновляем инерцию (вычисляем vx, vy)
    if (prev_time.time_since_epoch().count() == 0 || !target_detected.load()) {
        prev_time = now;
        prev_x = pivotX; prev_y = pivotY;
        prev_velocity_x = prev_velocity_y = 0.0;
    }
    else {
        double dt0 = std::max(1e-8,
            std::chrono::duration<double>(now - prev_time).count());
        prev_time = now;
        double vx = std::clamp((pivotX - prev_x) / dt0, -20000.0, 20000.0);
        double vy = std::clamp((pivotY - prev_y) / dt0, -20000.0, 20000.0);
        prev_x = pivotX; prev_y = pivotY;
        prev_velocity_x = vx; prev_velocity_y = vy;
    }

    // 2) Простое предсказание позиции
    double predX = pivotX + prev_velocity_x * (prediction_interval + 0.002);
    double predY = pivotY + prev_velocity_y * (prediction_interval + 0.002);

    // 3) Ветвление по режимам (как в moveMouse)
    if (use_kalman && !use_smoothing) {
        moveMouseWithKalmanSmoothing(predX, predY);
    }
    else if (!use_kalman && use_smoothing) {
        moveMouseWithSmoothing(predX, predY);
    }
    else if (use_kalman && use_smoothing) {
        return;
    }
    else {
        auto [mvX, mvY] = calc_movement(predX, predY);
        if (wind_mouse_enabled)
            windMouseMoveRelative(static_cast<int>(mvX), static_cast<int>(mvY));
        else
            queueMove(static_cast<int>(std::round(mvX)),
                static_cast<int>(std::round(mvY)));
    }
}