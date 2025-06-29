// HIDConnection.cpp
#include "HIDConnection.h"
#include <iostream>
#include <stdexcept>
#include <algorithm>  // for std::clamp
#include <cstdint>    // for INT8_MIN, INT8_MAX

HIDConnection::HIDConnection(uint16_t vid, uint16_t pid)
    : mouse_(nullptr)
{
    try {
        // один раз открываем и сохраняем объект MouseInstruct
        mouse_ = new MouseInstruct(MouseInstruct::getMouse(vid, pid));
        std::cout << "HID-устройство успешно открыто\n";
    }
    catch (const DeviceNotFoundException& e) {
        std::cerr << "Не удалось найти HID-устройство: " << e.what() << "\n";
        // mouse_ остаётся nullptr, isOpen() даст false
    }
}

HIDConnection::~HIDConnection() {
    delete mouse_;
}

bool HIDConnection::isOpen() const {
    return mouse_ != nullptr;
}

// Слишком центрировано
//void HIDConnection::move(int8_t x, int8_t y) {
//    if (!mouse_) throw std::runtime_error("HID-устройство не открыто");
//    std::cout << "[INFO] move(" << +x << ", " << +y << ")\n";
//    mouse_->move(x, y);
//}

void HIDConnection::move(int8_t x, int8_t y) {
    if (!mouse_)
        throw std::runtime_error("HID-устройство не открыто");

    //constexpr int yOffset = 0;

    //int adjY = static_cast<int>(y) + yOffset;
    //// Ограничиваем результат диапазоном int8_t
    //adjY = std::clamp(adjY, static_cast<int>(INT8_MIN), static_cast<int>(INT8_MAX));

    //int8_t finalY = static_cast<int8_t>(adjY);

    //std::cout << "[INFO] move(" << +x << ", " << +finalY
    //    << ")  // y + " << yOffset << "\n";

    mouse_->move(x, y);
}

void HIDConnection::click() {
    if (!mouse_) throw std::runtime_error("HID-устройство не открыто");
    mouse_->click();
}

void HIDConnection::press() {
    if (!mouse_) throw std::runtime_error("HID-устройство не открыто");
    mouse_->press();
}

void HIDConnection::release() {
    if (!mouse_) throw std::runtime_error("HID-устройство не открыто");
    mouse_->release();
}
