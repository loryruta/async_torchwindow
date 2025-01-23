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

struct Buffer {
    const char* const name; // For debugging
    void* data_d = nullptr;
    size_t size = 0;

    explicit Buffer(const char* name) : name(name) {}
    ~Buffer() = default;

    inline void resize(size_t new_size)
    {
        if (!data_d || size < new_size) {
            LOG("Resizing CUDA buffer \"%s\" from %zu to %zu bytes\n", name, size, new_size);
            void* new_data_d;
            CHECK_CUDA(cudaMalloc(&new_data_d, new_size));
            CHECK_CUDA(cudaMemcpy(new_data_d, data_d, size, cudaMemcpyDeviceToDevice));
            size = new_size;
            if (data_d) CHECK_CUDA(cudaFree(data_d));
            data_d = new_data_d;
        }
    }

    inline void destroy()
    {
        if (data_d) {
            CHECK_CUDA(cudaFree(data_d));
            data_d = nullptr;
        }
    }
};
} // namespace async_torchwindow
