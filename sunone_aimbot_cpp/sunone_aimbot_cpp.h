#ifndef SUNONE_AIMBOT_CPP_H
#define SUNONE_AIMBOT_CPP_H

#include "config.h"
#ifdef USE_CUDA
#include "trt_detector.h"
#endif
#include "dml_detector.h"
#include "mouse.h"
#include "SerialConnection.h"
#include "detection_buffer.h"
#include "Kmbox_b.h"
#include "KmboxNetConnection.h"
#include "MakcuConnection.h"
#include "HID/HIDConnection.h"
#include "HID/HIDConnectionV2.h"
#include "color_detector.h"

extern Config config;
#ifdef USE_CUDA
extern TrtDetector trt_detector;
#endif
extern DirectMLDetector* dml_detector;
extern DetectionBuffer detectionBuffer;
extern MouseThread* globalMouseThread;
extern SerialConnection* arduinoSerial;
extern KmboxConnection* kmboxSerial;
extern KmboxNetConnection* kmboxNetSerial;
extern HIDConnection* hid;
extern MakcuConnection* makcu;
extern HidConnectionV2* arduinoHid;
extern ColorDetector* color_detector;

extern std::atomic<bool> input_method_changed;
extern std::atomic<bool> aiming;
extern std::atomic<bool> shooting;
extern std::atomic<bool> zooming;

#endif // SUNONE_AIMBOT_CPP_H