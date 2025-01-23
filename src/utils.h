#pragma once

#include <stdexcept>

#define CHECK_CUDA(_error) async_torchwindow::check_cuda(_error, __FILE__, __LINE__)
#define LOG(format, ...) printf("%s" format, "[async_torchwindow] ", ##__VA_ARGS__)

namespace async_torchwindow
{
inline void check_cuda(cudaError_t error, char const* file, int line)
{
    if (error != cudaSuccess) {
        const char* cuda_error = cudaGetErrorString(error);
        LOG("CUDA error: %s (%s:%d)", cuda_error, file, line);
        throw std::runtime_error(cuda_error);
    }
}
} // namespace async_torchwindow
