// cuda_12_8_fused_preprocess.cu
#include <cuda_runtime.h>
#include <stdint.h>
#include <math.h>

// ---------------------------------------------
// Вспомогательная структура для letterbox
// ---------------------------------------------
struct Letterbox {
    int in_w, in_h;     // исходный размер
    int out_w, out_h;   // размер к которому приводим
    float scale;        // min(out_w/in_w, out_h/in_h)
    int pad_x;          // отступ слева
    int pad_y;          // отступ сверху
    float mean[3];      // mean по каналам (в [0..1])
    float std_[3];      // std по каналам (в [0..1])
};

// ---------------------------------------------
// Билинейная выборка из RGB uchar3
// src_pitch_pixels — шаг строки в пикселях (НЕ в байтах)
// ---------------------------------------------
__device__ __forceinline__ float3 sample_bilinear_rgb(
    const uchar3* __restrict__ src,
    int sw, int sh, int src_pitch_pixels,
    float x, float y)
{
    // clamp к валидным координатам
    x = fminf(fmaxf(x, 0.0f), (float)(sw - 1));
    y = fminf(fmaxf(y, 0.0f), (float)(sh - 1));

    int x0 = (int)floorf(x);
    int y0 = (int)floorf(y);
    int x1 = min(x0 + 1, sw - 1);
    int y1 = min(y0 + 1, sh - 1);

    float dx = x - x0;
    float dy = y - y0;

    const uchar3* row0 = src + y0 * src_pitch_pixels;
    const uchar3* row1 = src + y1 * src_pitch_pixels;

    uchar3 c00 = row0[x0];
    uchar3 c01 = row0[x1];
    uchar3 c10 = row1[x0];
    uchar3 c11 = row1[x1];

    // линейная интерполяция по X для двух строк
    float3 f0 = make_float3(
        c00.x + (c01.x - c00.x) * dx,
        c00.y + (c01.y - c00.y) * dx,
        c00.z + (c01.z - c00.z) * dx
    );
    float3 f1 = make_float3(
        c10.x + (c11.x - c10.x) * dx,
        c10.y + (c11.y - c10.y) * dx,
        c10.z + (c11.z - c10.z) * dx
    );

    // интерполяция по Y
    return make_float3(
        f0.x + (f1.x - f0.x) * dy,
        f0.y + (f1.y - f0.y) * dy,
        f0.z + (f1.z - f0.z) * dy
    );
}

// ---------------------------------------------
// Основное ядро: RGB8 -> NCHW float
// - Letterbox (масштаб + паддинг)
// - Билинейный ресайз
// - Нормализация (x-mean)/std
// ---------------------------------------------
__global__ void fused_preprocess_kernel_rgb8_to_nchw32f(
    const uchar3* __restrict__ src,
    int sw, int sh, int src_pitch_pixels,
    float* __restrict__ dst,
    Letterbox lb)
{
    int ox = blockIdx.x * blockDim.x + threadIdx.x; // x в выходе
    int oy = blockIdx.y * blockDim.y + threadIdx.y; // y в выходе
    if (ox >= lb.out_w || oy >= lb.out_h) return;

    // индекс пикселя в плоскости (H*W)
    int out_idx = oy * lb.out_w + ox;

    // координаты источника до паддинга
    float ix = (ox - lb.pad_x) / lb.scale;
    float iy = (oy - lb.pad_y) / lb.scale;

    float3 rgb;

    // если попали в зону паддинга — заливаем mean
    bool in_pad =
        (ox < lb.pad_x) ||
        (oy < lb.pad_y) ||
        (ox >= lb.pad_x + (int)(lb.in_w * lb.scale)) ||
        (oy >= lb.pad_y + (int)(lb.in_h * lb.scale));

    if (in_pad) {
        rgb = make_float3(lb.mean[0] * 255.0f, lb.mean[1] * 255.0f, lb.mean[2] * 255.0f);
    }
    else {
        rgb = sample_bilinear_rgb(src, sw, sh, src_pitch_pixels, ix, iy);
    }

    // в [0..1]
    rgb.x *= (1.0f / 255.0f);
    rgb.y *= (1.0f / 255.0f);
    rgb.z *= (1.0f / 255.0f);

    // нормализация
    rgb.x = (rgb.x - lb.mean[0]) / lb.std_[0];
    rgb.y = (rgb.y - lb.mean[1]) / lb.std_[1];
    rgb.z = (rgb.z - lb.mean[2]) / lb.std_[2];

    // запись в NCHW
    const int plane = lb.out_w * lb.out_h;
    dst[out_idx] = rgb.x;            // C0
    dst[plane + out_idx] = rgb.y;            // C1
    dst[plane * 2 + out_idx] = rgb.z;            // C2
}

// ---------------------------------------------
// Экспортируемый хелпер запуска (C-linkage!)
// mean/std — массивы из 3 элементов в [0..1]
// src_pitch_pixels — шаг строки в ПИКСЕЛЯХ (для uchar3)
// ---------------------------------------------
extern "C" void launch_fused_preprocess(
    const uchar3* d_rgb, int in_w, int in_h, int src_pitch_pixels,
    float* d_out, int out_w, int out_h,
    const float mean[3], const float std_[3],
    cudaStream_t stream)
{
    Letterbox lb{};
    lb.in_w = in_w;   lb.in_h = in_h;
    lb.out_w = out_w;  lb.out_h = out_h;

    lb.scale = fminf((float)out_w / (float)in_w, (float)out_h / (float)in_h);
    lb.pad_x = (int)((out_w - in_w * lb.scale) * 0.5f);
    lb.pad_y = (int)((out_h - in_h * lb.scale) * 0.5f);

    lb.mean[0] = mean[0]; lb.mean[1] = mean[1]; lb.mean[2] = mean[2];
    lb.std_[0] = std_[0]; lb.std_[1] = std_[1]; lb.std_[2] = std_[2];

    dim3 block(32, 16);
    dim3 grid((out_w + block.x - 1) / block.x,
        (out_h + block.y - 1) / block.y);

    fused_preprocess_kernel_rgb8_to_nchw32f << <grid, block, 0, stream >> > (
        d_rgb, in_w, in_h, src_pitch_pixels,
        d_out, lb
        );

    // (необязательно) проверка запуска:
    // cudaError_t err = cudaGetLastError();
    // if (err != cudaSuccess) printf("fused_preprocess: %s\n", cudaGetErrorString(err));
}
