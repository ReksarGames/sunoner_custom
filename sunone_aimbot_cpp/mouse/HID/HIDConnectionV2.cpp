#include "HidConnectionV2.h"
#include <iostream>
#include <stdexcept>
#include <vector>
#include <cstring>

HidConnectionV2::HidConnectionV2(uint16_t vid, uint16_t pid) {
    // Инициализация hidapi
    if (hid_init()) {
        throw std::runtime_error("Failed to initialize hidapi");  // std::string
    }

    // Открытие устройства по VID и PID
    device_ = hid_open(vid, pid, nullptr);
    if (!device_) {
        // Преобразование строки wchar_t в std::string
        throw std::runtime_error("Failed to open HID device: ");
    }

    std::wcout << L"[HIDConnection] Device opened: VID=0x"
        << std::hex << vid << L" PID=0x" << pid << std::dec << std::endl;
}

HidConnectionV2::~HidConnectionV2() {
    if (device_) {
        hid_close(device_);
    }
    hid_exit();  // Завершаем работу с hidapi
}

void HidConnectionV2::move(int8_t x, int8_t y) {
    RawHIDPacket64 packet{};
    packet.dx = x;
    packet.dy = y;
    sendRawHID(packet);
}

void HidConnectionV2::click() {
    press();
    release();
}

void HidConnectionV2::press() {
    RawHIDPacket64 packet{};
    packet.buttons = 1;  // Предположим, что кнопка 1 — это левая кнопка мыши
    sendRawHID(packet);
    currentButtons_ = packet.buttons;
}

void HidConnectionV2::release() {
    RawHIDPacket64 packet{};
    sendRawHID(packet);
    currentButtons_ = 0;
}

void HidConnectionV2::sendRawHID(const RawHIDPacket64& packet) {
    std::lock_guard<std::mutex> lock(writeMutex_);

    if (!device_) {
        throw std::runtime_error("HID device is not open");
    }

    // Отправляем данные в устройство
    std::vector<uint8_t> report(sizeof(RawHIDPacket64) + 1, 0);  // REPORT_ID + данные пакета
    report[0] = 1;  // REPORT_ID
    std::memcpy(report.data() + 1, &packet, sizeof(packet));

    int res = hid_write(device_, report.data(), report.size());
    if (res == -1) {
        // Преобразование ошибки в std::string
        throw std::runtime_error("Failed to send HID report: ");
    }
}
