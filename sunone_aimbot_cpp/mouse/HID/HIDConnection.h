// HIDConnection.h
#pragma once

#include <cstdint>
#include "MouseInstruct.h"

class HIDConnection {
public:
    HIDConnection(uint16_t vid, uint16_t pid);
    ~HIDConnection();

    void move(int8_t x, int8_t y);
    void click();
    void press();
    void release();

    bool isOpen() const;

private:
    MouseInstruct* mouse_;   // единственный объект, через который мы шлём команды
};
