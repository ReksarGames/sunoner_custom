#pragma once

#include <cstdint>
#include <mutex>
#include <windows.h>
#include <vector>
#include <hidapi/hidapi.h>

// Структура пакета для HID (например, для мыши)
#pragma pack(push, 1)
struct RawHIDPacket64 {
    int16_t dx;   // Движение по оси X (16 бит)
    int16_t dy;   // Движение по оси Y (16 бит)
    uint8_t buttons; // Кнопки мыши
    uint8_t padding[60];  // Дополнительное место для выравнивания
    uint8_t pingCode; // Пинг-код для проверки соединения
};
#pragma pack(pop)

class HIDConnection {
public:
    HIDConnection(uint16_t vid, uint16_t pid);
    ~HIDConnection();

    void move(int16_t x, int16_t y);
    void click();
    void press();
    void release();

    bool isOpen() const noexcept {
        return device_ != nullptr;
    }

private:
    void sendRawHID(const RawHIDPacket64& packet);
    uint8_t* makeReport(const int16_t& x, const int16_t& y);

    hid_device* device_{ nullptr };
    uint8_t currentButtons_{ 0 };
    int16_t limit_xy(int16_t xy);
};
