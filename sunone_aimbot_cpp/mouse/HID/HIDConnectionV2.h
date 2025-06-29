#pragma once

#include <cstdint>
#include <mutex>
#include <hidapi/hidapi.h>
#include <vector>

// Структура пакета ровно 64 байта
#pragma pack(push, 1)
struct RawHIDPacket64 {
    int8_t   dx;
    int8_t   dy;
    uint8_t  buttons;
    uint8_t  padding[61];
};
#pragma pack(pop)

class HidConnectionV2 {
public:
    HidConnectionV2(uint16_t vid, uint16_t pid);
    ~HidConnectionV2();

    void move(int8_t x, int8_t y);
    void click();
    void press();
    void release();

    bool isOpen() const noexcept {
        return device_ != nullptr;
    }

private:
    void sendRawHID(const RawHIDPacket64& packet);

    hid_device* device_{ nullptr };
    std::mutex writeMutex_;
    uint8_t currentButtons_{ 0 };
};
