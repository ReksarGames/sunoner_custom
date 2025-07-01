#define WIN32_LEAN_AND_MEAN
#define _WINSOCKAPI_
#include <winsock2.h>
#include <Windows.h>

#include <shellapi.h>

#include "imgui/imgui.h"
#include <imgui_internal.h>

#include "sunone_aimbot_cpp.h"
#include "include/other_tools.h"
#include "kmbox_net/picture.h"

std::string ghub_version = get_ghub_version();

int prev_fovX = config.fovX;
int prev_fovY = config.fovY;
float prev_minSpeedMultiplier = config.minSpeedMultiplier;
float prev_maxSpeedMultiplier = config.maxSpeedMultiplier;
float prev_predictionInterval = config.predictionInterval;
float prev_snapRadius = config.snapRadius;
float prev_nearRadius = config.nearRadius;
float prev_speedCurveExponent = config.speedCurveExponent;
float prev_snapBoostFactor = config.snapBoostFactor;

int prev_smoothness = config.smoothness;
static bool  prev_use_smoothing = config.use_smoothing;
static bool  prev_use_kalman = config.use_kalman;
static float prev_kalman_process_noise = config.kalman_process_noise;
static float prev_kalman_measure_noise = config.kalman_measurement_noise;

bool  prev_wind_mouse_enabled = config.wind_mouse_enabled;
float prev_wind_G = config.wind_G;
float prev_wind_W = config.wind_W;
float prev_wind_M = config.wind_M;
float prev_wind_D = config.wind_D;

bool prev_auto_shoot = config.auto_shoot;
float prev_bScope_multiplier = config.bScope_multiplier;

static void draw_target_correction_demo()
{
    if (ImGui::CollapsingHeader("Visual demo"))
    {
        ImVec2 canvas_sz(220, 220);
        ImGui::InvisibleButton("##tc_canvas", canvas_sz);

        ImVec2 p0 = ImGui::GetItemRectMin();
        ImVec2 p1 = ImGui::GetItemRectMax();
        ImVec2 center{ (p0.x + p1.x) * 0.5f, (p0.y + p1.y) * 0.5f };

        ImDrawList* dl = ImGui::GetWindowDrawList();
        dl->AddRectFilled(p0, p1, IM_COL32(25, 25, 25, 255));

        const float scale = 4.0f;
        float near_px = config.nearRadius * scale;
        float snap_px = config.snapRadius * scale;
        near_px = ImClamp(near_px, 10.0f, canvas_sz.x * 0.45f);
        snap_px = ImClamp(snap_px, 6.0f, near_px - 4.0f);

        dl->AddCircle(center, near_px, IM_COL32(80, 120, 255, 180), 64, 2.0f);
        dl->AddCircle(center, snap_px, IM_COL32(255, 100, 100, 180), 64, 2.0f);

        static float  dist_px = near_px;
        static float  vel_px = 0.0f;
        static double last_t = ImGui::GetTime();
        double now = ImGui::GetTime();
        double dt = now - last_t;
        last_t = now;

        double dist_units = dist_px / scale;
        double speed_mult;
        if (dist_units < config.snapRadius)
            speed_mult = config.minSpeedMultiplier * config.snapBoostFactor;
        else if (dist_units < config.nearRadius)
        {
            double t = dist_units / config.nearRadius;
            double crv = 1.0 - pow(1.0 - t, config.speedCurveExponent);
            speed_mult = config.minSpeedMultiplier +
                (config.maxSpeedMultiplier - config.minSpeedMultiplier) * crv;
        }
        else
        {
            double norm = ImClamp(dist_units / config.nearRadius, 0.0, 1.0);
            speed_mult = config.minSpeedMultiplier +
                (config.maxSpeedMultiplier - config.minSpeedMultiplier) * norm;
        }

        double base_px_s = 60.0;
        vel_px = static_cast<float>(base_px_s * speed_mult);
        dist_px -= vel_px * static_cast<float>(dt);
        if (dist_px <= 0.0f) dist_px = near_px;

        ImVec2 dot{ center.x - dist_px, center.y };
        dl->AddCircleFilled(dot, 4.0f, IM_COL32(255, 255, 80, 255));

        ImGui::Dummy(ImVec2(0, 4));
        ImGui::TextColored(ImVec4(0.31f, 0.48f, 1.0f, 1.0f), "Near radius");
        ImGui::SameLine(130);
        ImGui::TextColored(ImVec4(1.0f, 0.39f, 0.39f, 1.0f), "Snap radius");
    }
}

struct DemoKalman1D {
    double x{ 0 }, v{ 0 }, P{ 1 }, Q, R;
    DemoKalman1D(double processNoise, double measurementNoise)
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

static void draw_smoothing_kalman_demo()
{
    if (!ImGui::CollapsingHeader("Smooth + Kalman Demo"))
        return;

    ImVec2 canvas_sz(220, 220);
    ImGui::InvisibleButton("##sk_canvas", canvas_sz);
    ImVec2 p0 = ImGui::GetItemRectMin();
    ImVec2 p1 = ImGui::GetItemRectMax();
    ImVec2 center{ (p0.x + p1.x) * 0.5f, (p0.y + p1.y) * 0.5f };
    ImDrawList* dl = ImGui::GetWindowDrawList();
    dl->AddRectFilled(p0, p1, IM_COL32(30, 30, 30, 255));

    // Время и dt
    static double last_t = ImGui::GetTime();
    double now = ImGui::GetTime();
    double dt = now - last_t;
    last_t = now;

    // Raw-цель по окружности
    static double angle = 0.0;
    angle += dt * 1.0;
    if (angle > 2 * 3.14159265358979323846) angle -= 2 * 3.14159265358979323846;
    double rad = canvas_sz.x * 0.4;
    double rawX = center.x + cos(angle) * rad;
    double rawY = center.y + sin(angle) * rad;

    // Калман-фильтр
    static DemoKalman1D kfX{ config.kalman_process_noise, config.kalman_measurement_noise },
        kfY{ config.kalman_process_noise, config.kalman_measurement_noise };
    static float lastQ = config.kalman_process_noise,
        lastR = config.kalman_measurement_noise;
    if (lastQ != config.kalman_process_noise || lastR != config.kalman_measurement_noise) {
        kfX = DemoKalman1D(config.kalman_process_noise, config.kalman_measurement_noise);
        kfY = DemoKalman1D(config.kalman_process_noise, config.kalman_measurement_noise);
        lastQ = config.kalman_process_noise;
        lastR = config.kalman_measurement_noise;
    }
    double kalX = kfX.update(rawX, dt);
    double kalY = kfY.update(rawY, dt);

    // Экспоненциальное сглаживание
    static double smX = center.x, smY = center.y;
    int   N = config.smoothness > 0 ? config.smoothness : 1;
    double alpha = 1.0 / N;
    // Сброс, если Kalman-координата «прыгает» слишком далеко
    const double resetThreshold = 5.0;
    if (hypot(kalX - smX, kalY - smY) > resetThreshold) {
        smX = kalX;
        smY = kalY;
    }
    else {
        smX += (kalX - smX) * alpha;
        smY += (kalY - smY) * alpha;
    }

    // Рисуем точки: белая = raw, красная = kalman, зелёная = smoothed
    dl->AddCircleFilled({ (float)rawX, (float)rawY }, 4.0f, IM_COL32(255, 255, 255, 200));
    dl->AddCircleFilled({ (float)kalX, (float)kalY }, 4.0f, IM_COL32(255, 100, 100, 200));
    dl->AddCircleFilled({ (float)smX,  (float)smY }, 4.0f, IM_COL32(100, 255, 100, 200));

    // Легенда
    dl->AddText({ p0.x + 5, p0.y + 5 },
        IM_COL32(200, 200, 200, 255),
        "W=Raw  R=Kalman  G=Smoothed");
}


void draw_mouse()
{
    ImGui::SeparatorText("FOV");
    ImGui::SliderInt("FOV X", &config.fovX, 10, 120);
    ImGui::SliderInt("FOV Y", &config.fovY, 10, 120);

    ImGui::SliderInt("Smoothness", &config.smoothness, 1, 200, "%d");
    if (ImGui::Checkbox("Enable Smooth Movement", &config.use_smoothing))
    {
        config.saveConfig();
        input_method_changed.store(true);
        globalMouseThread->setUseSmoothing(config.use_smoothing);
    }

    if (ImGui::Checkbox("Enable Kalman Filter", &config.use_kalman))
    {
        config.saveConfig();
        input_method_changed.store(true);
        globalMouseThread->setUseKalman(config.use_kalman);
    }


    if (config.use_kalman)
    {
        bool changed = false;
        changed |= ImGui::SliderFloat("Kalman Process Noise", &config.kalman_process_noise, 0.10f, 1.0f, "%.4f");
        changed |= ImGui::SliderFloat("Kalman Measurement Noise", &config.kalman_measurement_noise, 0.50f, 1.0f, "%.4f");
        changed |= ImGui::SliderFloat("Kalman Speed Multiplier X", &config.kalman_speed_multiplier_x, 0.1f, 5.0f, "%.2f");
        changed |= ImGui::SliderFloat("Kalman Speed Multiplier Y", &config.kalman_speed_multiplier_y, 0.1f, 5.0f, "%.2f");

        if (changed)
        {
            config.saveConfig();
            input_method_changed.store(true);
            globalMouseThread->setKalmanParams(
                config.kalman_process_noise,
                config.kalman_measurement_noise
            );
            globalMouseThread->setKalmanSpeedMultiplierX(
                config.kalman_speed_multiplier_x);
            globalMouseThread->setKalmanSpeedMultiplierY(
                config.kalman_speed_multiplier_y);
        }
    }


    draw_smoothing_kalman_demo();

    ImGui::SeparatorText("Speed Multiplier");
    ImGui::SliderFloat("Min Speed Multiplier", &config.minSpeedMultiplier, 0.1f, 5.0f, "%.1f");
    ImGui::SliderFloat("Max Speed Multiplier", &config.maxSpeedMultiplier, 0.1f, 5.0f, "%.1f");

    ImGui::SeparatorText("Prediction");
    ImGui::SliderFloat("Prediction Interval", &config.predictionInterval, 0.00f, 0.5f, "%.2f");
    if (config.predictionInterval == 0.00f)
    {
        ImGui::SameLine();
        ImGui::TextColored(ImVec4(255, 0, 0, 255), "-> Disabled");
    }
    else
    {
        
        if (ImGui::SliderInt("Future Positions", &config.prediction_futurePositions, 1, 40))
        {
            config.saveConfig();
            input_method_changed.store(true);
        }

        ImGui::SameLine();
        if (ImGui::Checkbox("Draw##draw_future_positions_button", &config.draw_futurePositions))
        {
            config.saveConfig();
        }
    }

    ImGui::SeparatorText("Target corrention");
    ImGui::SliderFloat("Snap Radius", &config.snapRadius, 0.1f, 5.0f, "%.1f");
    ImGui::SliderFloat("Near Radius", &config.nearRadius, 1.0f, 40.0f, "%.1f");
    ImGui::SliderFloat("Speed Curve Exponent", &config.speedCurveExponent, 0.1f, 10.0f, "%.1f");
    ImGui::SliderFloat("Snap Boost Factor", &config.snapBoostFactor, 0.01f, 4.00f, "%.2f");
    draw_target_correction_demo();

    ImGui::SeparatorText("Game Profile");
    std::vector<std::string> profile_names;
    for (const auto& kv : config.game_profiles)
        profile_names.push_back(kv.first);

    static int selected_index = 0;
    for (size_t i = 0; i < profile_names.size(); ++i)
    {
        if (profile_names[i] == config.active_game)
        {
            selected_index = static_cast<int>(i);
            break;
        }
    }

    std::vector<const char*> profile_items;
    for (const auto& name : profile_names)
        profile_items.push_back(name.c_str());

    if (ImGui::Combo("Active Game Profile", &selected_index, profile_items.data(), static_cast<int>(profile_items.size())))
    {
        config.active_game = profile_names[selected_index];
        config.saveConfig();
        globalMouseThread->updateConfig(
            config.detection_resolution,
            config.fovX,
            config.fovY,
            config.minSpeedMultiplier,
            config.maxSpeedMultiplier,
            config.predictionInterval,
            config.auto_shoot,
            config.bScope_multiplier
        );
        input_method_changed.store(true);
    }

    const auto& gp = config.currentProfile();

    ImGui::Text("Current profile: %s", gp.name.c_str());
    ImGui::Text("Sens: %.4f", gp.sens);
    ImGui::Text("Yaw:  %.4f", gp.yaw);
    ImGui::Text("Pitch: %.4f", gp.pitch);
    ImGui::Text("FOV Scaled: %s", gp.fovScaled ? "true" : "false");

    if (gp.name != "UNIFIED")
    {
        Config::GameProfile& modifiable = config.game_profiles[gp.name];
        bool changed = false;

        float sens_f = static_cast<float>(modifiable.sens);
        float yaw_f = static_cast<float>(modifiable.yaw);
        float pitch_f = static_cast<float>(modifiable.pitch);
        float baseFOV_f = static_cast<float>(modifiable.baseFOV);

        changed |= ImGui::SliderFloat("Sensitivity", &sens_f, 0.001f, 10.0f, "%.4f");
        changed |= ImGui::SliderFloat("Yaw", &yaw_f, 0.001f, 0.1f, "%.4f");
        changed |= ImGui::SliderFloat("Pitch", &pitch_f, 0.001f, 0.1f, "%.4f");

        changed |= ImGui::Checkbox("FOV Scaled", &modifiable.fovScaled);
        if (modifiable.fovScaled)
        {
            changed |= ImGui::SliderFloat("Base FOV", &baseFOV_f, 10.0f, 180.0f, "%.1f");
        }

        if (changed)
        {
            modifiable.sens = static_cast<double>(sens_f);
            modifiable.yaw = static_cast<double>(yaw_f);

            if (gp.pitch == 0.0 || !gp.fovScaled)
                modifiable.pitch = modifiable.yaw;
            else
                modifiable.pitch = static_cast<double>(pitch_f);

            modifiable.baseFOV = static_cast<double>(baseFOV_f);

            config.saveConfig();
            input_method_changed.store(true);
        }
    }

    ImGui::SeparatorText("Manage Profiles");

    static char new_profile_name[64] = "";
    ImGui::InputText("New profile name", new_profile_name, sizeof(new_profile_name));
    ImGui::SameLine();
    if (ImGui::Button("Add Profile"))
    {
        std::string name = std::string(new_profile_name);
        if (!name.empty() && config.game_profiles.count(name) == 0)
        {
            Config::GameProfile gp;
            gp.name = name;
            gp.sens = 1.0;
            gp.yaw = 0.022;
            gp.pitch = 0.022;
            gp.fovScaled = false;
            gp.baseFOV = 90.0;
            config.game_profiles[name] = gp;
            config.active_game = name;
            config.saveConfig();
            input_method_changed.store(true);
            new_profile_name[0] = '\0'; // clear
        }
    }

    if (gp.name != "UNIFIED")
    {
        ImGui::PushStyleColor(ImGuiCol_Button, IM_COL32(200, 50, 50, 255));
        if (ImGui::Button("Delete Current Profile"))
        {
            config.game_profiles.erase(gp.name);
            if (!config.game_profiles.empty())
                config.active_game = config.game_profiles.begin()->first;
            else
                config.active_game = "UNIFIED";

            config.saveConfig();
            input_method_changed.store(true);
        }
        ImGui::PopStyleColor();
    }

    ImGui::SeparatorText("Easy No Recoil");
    ImGui::Checkbox("Easy No Recoil", &config.easynorecoil);
    if (config.easynorecoil)
    {
        ImGui::SliderFloat("No Recoil Strength", &config.easynorecoilstrength, 0.1f, 500.0f, "%.1f");
        ImGui::TextColored(ImVec4(1.0f, 1.0f, 0.0f, 1.0f), "Left/Right Arrow keys: Adjust recoil strength by 10");
        
        if (config.easynorecoilstrength >= 100.0f)
        {
            ImGui::TextColored(ImVec4(255, 255, 0, 255), "WARNING: High recoil strength may be detected.");
        }
    }

    ImGui::SeparatorText("Auto Shoot");

    ImGui::Checkbox("Auto Shoot", &config.auto_shoot);
    if (config.auto_shoot)
    {
        ImGui::SliderFloat("bScope Multiplier", &config.bScope_multiplier, 0.5f, 2.0f, "%.1f");
    }

    ImGui::SeparatorText("Wind mouse");

    if (ImGui::Checkbox("Enable WindMouse", &config.wind_mouse_enabled))
    {
        config.saveConfig();
        input_method_changed.store(true);
    }

    if (config.wind_mouse_enabled)
    {
        if (ImGui::SliderFloat("Gravity force", &config.wind_G, 4.00f, 40.00f, "%.2f"))
        {
            config.saveConfig();
        }

        if (ImGui::SliderFloat("Wind fluctuation", &config.wind_W, 1.00f, 40.00f, "%.2f"))
        {
            config.saveConfig();
        }

        if (ImGui::SliderFloat("Max step (velocity clip)", &config.wind_M, 1.00f, 40.00f, "%.2f"))
        {
            config.saveConfig();
        }

        if (ImGui::SliderFloat("Distance where behaviour changes", &config.wind_D, 1.00f, 40.00f, "%.2f"))
        {
            config.saveConfig();
        }

        if (ImGui::Button("Reset Wind Mouse to default settings"))
        {
            config.wind_G = 18.0f;
            config.wind_W = 15.0f;
            config.wind_M = 10.0f;
            config.wind_D = 8.0f;
            config.saveConfig();
        }
    }

    ImGui::SeparatorText("Input method");
    std::vector<std::string> input_methods = { "MAKCU", "HID", "ARDUINO_HID", "KMBOX_B", "ARDUINO"};

    std::vector<const char*> method_items;
    method_items.reserve(input_methods.size());
    for (const auto& item : input_methods)
    {
        method_items.push_back(item.c_str());
    }

    std::string combo_label = "Mouse Input method";
    int input_method_index = 0;
    for (size_t i = 0; i < input_methods.size(); ++i)
    {
        if (input_methods[i] == config.input_method)
        {
            input_method_index = static_cast<int>(i);
            break;
        }
    }

    if (ImGui::Combo("Mouse Input Method", &input_method_index, method_items.data(), static_cast<int>(method_items.size())))
    {
        std::string new_input_method = input_methods[input_method_index];

        if (new_input_method != config.input_method)
        {
            config.input_method = new_input_method;
            config.saveConfig();
            input_method_changed.store(true);
        }
    }

    else if (config.input_method == "MAKCU")
    {
        ImGui::Text("Makcu Settings");

        ImGui::InputText("Makcu Port", config.makcu_port.data(), config.makcu_port.size());
        //TODO: Убери строчну на 362
        ImGui::InputInt("Makcu Baudrate", &config.makcu_baudrate);

        if (!makcu)
        {
            makcu = new MakcuConnection(config.makcu_port, config.makcu_baudrate);
            if (!makcu->isOpen())
            {
                ImGui::TextColored(ImVec4(255, 0, 0, 255), "Failed to connect to Makcu.");
            }
            else
            {
                ImGui::TextColored(ImVec4(0, 255, 0, 255), "Makcu connected.");
            }
        }
    }


    else if (config.input_method == "HID")
    {
        if (hid && hid->isOpen())
            ImGui::TextColored(ImVec4(0, 255, 0, 255), "Arduino (HID) connected");
        else
            ImGui::TextColored(ImVec4(255, 0, 0, 255), "Arduino (HID) not connected");

    }

    else if (config.input_method == "ARDUINO_HID")
    {
        if (arduinoHid && arduinoHid->isOpen())
            ImGui::TextColored(ImVec4(0, 255, 0, 255), "Arduino (RawHID) connected");
        else
            ImGui::TextColored(ImVec4(255, 0, 0, 255), "Arduino (RawHID) not connected");

    }

    if (config.input_method == "ARDUINO")
    {
        if (arduinoSerial)
        {
            if (arduinoSerial->isOpen())
            {
                ImGui::TextColored(ImVec4(0, 255, 0, 255), "Arduino connected");
            }
            else
            {
                ImGui::TextColored(ImVec4(255, 0, 0, 255), "Arduino not connected");
            }
        }

        std::vector<std::string> port_list;
        for (int i = 1; i <= 30; ++i)
        {
            port_list.push_back("COM" + std::to_string(i));
        }

        std::vector<const char*> port_items;
        port_items.reserve(port_list.size());
        for (const auto& port : port_list)
        {
            port_items.push_back(port.c_str());
        }

        int port_index = 0;
        for (size_t i = 0; i < port_list.size(); ++i)
        {
            if (port_list[i] == config.arduino_port)
            {
                port_index = static_cast<int>(i);
                break;
            }
        }

        if (ImGui::Combo("Arduino Port", &port_index, port_items.data(), static_cast<int>(port_items.size())))
        {
            config.arduino_port = port_list[port_index];
            config.saveConfig();
            input_method_changed.store(true);
        }

        std::vector<int> baud_rate_list = { 9600, 19200, 38400, 57600, 115200 };
        std::vector<std::string> baud_rate_str_list;
        for (const auto& rate : baud_rate_list)
        {
            baud_rate_str_list.push_back(std::to_string(rate));
        }

        std::vector<const char*> baud_rate_items;
        baud_rate_items.reserve(baud_rate_str_list.size());
        for (const auto& rate_str : baud_rate_str_list)
        {
            baud_rate_items.push_back(rate_str.c_str());
        }

        int baud_rate_index = 0;
        for (size_t i = 0; i < baud_rate_list.size(); ++i)
        {
            if (baud_rate_list[i] == config.arduino_baudrate)
            {
                baud_rate_index = static_cast<int>(i);
                break;
            }
        }

        if (ImGui::Combo("Arduino Baudrate", &baud_rate_index, baud_rate_items.data(), static_cast<int>(baud_rate_items.size())))
        {
            config.arduino_baudrate = baud_rate_list[baud_rate_index];
            config.saveConfig();
            input_method_changed.store(true);
        }

        if (ImGui::Checkbox("Arduino 16-bit Mouse", &config.arduino_16_bit_mouse))
        {
            config.saveConfig();
            input_method_changed.store(true);
        }
        if (ImGui::Checkbox("Arduino Enable Keys", &config.arduino_enable_keys))
        {
            config.saveConfig();
            input_method_changed.store(true);
        }
    }
    else if (config.input_method == "GHUB")
    {
        if (ghub_version == "13.1.4")
        {
            std::string ghub_version_label = "The correct version of Ghub is installed: " + ghub_version;
            ImGui::Text(ghub_version_label.c_str());
        }
        else
        {
            if (ghub_version == "")
            {
                ghub_version = "unknown";
            }

            std::string ghub_version_label = "Installed Ghub version: " + ghub_version;
            ImGui::Text(ghub_version_label.c_str());
            ImGui::Text("The wrong version of Ghub is installed or the path to Ghub is not set by default.\nDefault system path: C:\\Program Files\\LGHUB");
            if (ImGui::Button("GHub Docs"))
            {
                ShellExecute(0, 0, L"https://github.com/SunOner/sunone_aimbot_docs/blob/main/tips/ghub.md", 0, 0, SW_SHOW);
            }
        }
        ImGui::TextColored(ImVec4(255, 0, 0, 255), "Use at your own risk, the method is detected in some games.");
    }
    else if (config.input_method == "WIN32")
    {
        ImGui::TextColored(ImVec4(255, 255, 255, 255), "This is a standard mouse input method, it may not work in most games. Use GHUB or ARDUINO.");
        ImGui::TextColored(ImVec4(255, 0, 0, 255), "Use at your own risk, the method is detected in some games.");
    }
    else if (config.input_method == "KMBOX_B")
    {
        std::vector<std::string> port_list;
        for (int i = 1; i <= 30; ++i)
        {
            port_list.push_back("COM" + std::to_string(i));
        }
        std::vector<const char*> port_items;
        port_items.reserve(port_list.size());
        for (auto& p : port_list) port_items.push_back(p.c_str());

        int port_index = 0;
        for (size_t i = 0; i < port_list.size(); ++i)
        {
            if (port_list[i] == config.kmbox_b_port)
            {
                port_index = (int)i;
                break;
            }
        }

        if (ImGui::Combo("kmbox Port", &port_index, port_items.data(), (int)port_items.size()))
        {
            config.kmbox_b_port = port_list[port_index];
            config.saveConfig();
            input_method_changed.store(true);
        }

        std::vector<int> baud_list = { 9600, 19200, 38400, 57600, 115200 };
        std::vector<std::string> baud_str_list;
        for (int b : baud_list) baud_str_list.push_back(std::to_string(b));
        std::vector<const char*> baud_items;
        baud_items.reserve(baud_str_list.size());
        for (auto& bs : baud_str_list) baud_items.push_back(bs.c_str());

        int baud_index = 0;
        for (size_t i = 0; i < baud_list.size(); ++i)
        {
            if (baud_list[i] == config.kmbox_b_baudrate)
            {
                baud_index = (int)i;
                break;
            }
        }

        if (ImGui::Combo("kmbox Baudrate", &baud_index, baud_items.data(), (int)baud_items.size()))
        {
            config.kmbox_b_baudrate = baud_list[baud_index];
            config.saveConfig();
            input_method_changed.store(true);
        }

        if (ImGui::Button("Run boot.py"))
        {
            kmboxSerial->start_boot();
        }

        if (ImGui::Button("Reboot KMBOX"))
        {
            kmboxSerial->reboot();
        }

        if (ImGui::Button("Send Stop"))
        {
            kmboxSerial->send_stop();
        }
    }
    else if (config.input_method == "KMBOX_NET")
    {
        static char ip[32], port[8], uuid[16];
        strncpy(ip, config.kmbox_net_ip.c_str(), sizeof(ip));
        strncpy(port, config.kmbox_net_port.c_str(), sizeof(port));
        strncpy(uuid, config.kmbox_net_uuid.c_str(), sizeof(uuid));

        ImGui::InputText("kmboxNet IP", ip, sizeof(ip));
        ImGui::InputText("Port", port, sizeof(port));
        ImGui::InputText("UUID", uuid, sizeof(uuid));

        if (ImGui::Button("Save & Reconnect"))
        {
            config.kmbox_net_ip = ip;
            config.kmbox_net_port = port;
            config.kmbox_net_uuid = uuid;
            config.saveConfig();
            input_method_changed.store(true);
        }

        if (kmboxNetSerial && kmboxNetSerial->isOpen())
        {
            ImGui::TextColored(ImVec4(0, 255, 0, 255), "kmboxNet connected");
        }
        else
        {
            ImGui::TextColored(ImVec4(255, 0, 0, 255), "kmboxNet not connected");
        }
        
        if (ImGui::Button("Reboot box"))
        {
            if (kmboxNetSerial)
            {
                kmboxNetSerial->reboot();
            }
        }

        if (ImGui::Button("Change Kmbox image"))
        {
            if (kmboxNetSerial)
            {
                kmboxNetSerial->lcdColor(0);
                kmboxNetSerial->lcdPicture(gImage_128x160);
            }
        }
    }

    ImGui::Separator();
    ImGui::TextColored(ImVec4(255, 255, 255, 100), "Do not test shooting and aiming with the overlay is open.");

    if (prev_fovX != config.fovX ||
        prev_fovY != config.fovY ||
        config.smoothness != prev_smoothness ||
        prev_use_smoothing != config.use_smoothing ||
        prev_use_kalman != config.use_kalman ||
        prev_kalman_process_noise != config.kalman_process_noise ||
        prev_kalman_measure_noise != config.kalman_measurement_noise ||
        prev_minSpeedMultiplier != config.minSpeedMultiplier ||
        prev_maxSpeedMultiplier != config.maxSpeedMultiplier ||
        prev_predictionInterval != config.predictionInterval ||
        prev_snapRadius != config.snapRadius ||
        prev_nearRadius != config.nearRadius ||
        prev_speedCurveExponent != config.speedCurveExponent ||
        prev_snapBoostFactor != config.snapBoostFactor)
    {
        prev_fovX = config.fovX;
        prev_fovY = config.fovY;
        prev_smoothness = config.smoothness;
        prev_use_smoothing            = config.use_smoothing;
        prev_use_kalman               = config.use_kalman;
        prev_kalman_process_noise     = config.kalman_process_noise;
        prev_kalman_measure_noise     = config.kalman_measurement_noise;
        prev_minSpeedMultiplier = config.minSpeedMultiplier;
        prev_maxSpeedMultiplier = config.maxSpeedMultiplier;
        prev_predictionInterval = config.predictionInterval;
        prev_snapRadius = config.snapRadius;
        prev_nearRadius = config.nearRadius;
        prev_speedCurveExponent = config.speedCurveExponent;
        prev_snapBoostFactor = config.snapBoostFactor;

        globalMouseThread->updateConfig(
            config.detection_resolution,
            config.fovX,
            config.fovY,
            config.minSpeedMultiplier,
            config.maxSpeedMultiplier,
            config.predictionInterval,
            config.auto_shoot,
            config.bScope_multiplier);

        config.saveConfig();
        globalMouseThread->setSmoothnessValue(config.smoothness);
        globalMouseThread->setUseSmoothing(config.use_smoothing);
        globalMouseThread->setUseKalman(config.use_kalman);
        globalMouseThread->setKalmanParams(
            config.kalman_process_noise,
            config.kalman_measurement_noise
        );
    }

    if (prev_wind_mouse_enabled != config.wind_mouse_enabled ||
        prev_wind_G != config.wind_G ||
        prev_wind_W != config.wind_W ||
        prev_wind_M != config.wind_M ||
        prev_wind_D != config.wind_D)
    {
        prev_wind_mouse_enabled = config.wind_mouse_enabled;
        prev_wind_G = config.wind_G;
        prev_wind_W = config.wind_W;
        prev_wind_M = config.wind_M;
        prev_wind_D = config.wind_D;

        globalMouseThread->updateConfig(
            config.detection_resolution,
            config.fovX,
            config.fovY,
            config.minSpeedMultiplier,
            config.maxSpeedMultiplier,
            config.predictionInterval,
            config.auto_shoot,
            config.bScope_multiplier);

        config.saveConfig();
    }

    if (prev_auto_shoot != config.auto_shoot ||
        prev_bScope_multiplier != config.bScope_multiplier)
    {
        prev_auto_shoot = config.auto_shoot;
        prev_bScope_multiplier = config.bScope_multiplier;

        globalMouseThread->updateConfig(
            config.detection_resolution,
            config.fovX,
            config.fovY,
            config.minSpeedMultiplier,
            config.maxSpeedMultiplier,
            config.predictionInterval,
            config.auto_shoot,
            config.bScope_multiplier);

        config.saveConfig();
    }
}