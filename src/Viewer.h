#pragma once

#include "utils.h"

namespace async_torchwindow
{
enum ViewerType {
    IMAGE,
    GAUSSIAN_SPLATTING,
};

struct Colorbuffer {
    int width = 0;
    int height = 0;
    /// Pointer to the device image data. Memory layout: `(H, W, 4)`.
    float* data_d = nullptr;

    [[nodiscard]] bool is_invalid() const { return data_d == nullptr; }

    void allocate()
    {
        if (width == 0 || height == 0) throw std::runtime_error("Invalid colorbuffer size");
        if (data_d) throw std::runtime_error("Colorbuffer already allocated");
        CHECK_CUDA(cudaMalloc(&data_d, width * height * 4 * sizeof(float)));
    }

    void destroy()
    {
        if (!data_d) return;
        CHECK_CUDA(cudaFree(data_d));
        data_d = nullptr;
    }

    static Colorbuffer invalid() { return Colorbuffer{}; }
};

class Viewer
{
public:
    const ViewerType type;

    explicit Viewer(ViewerType type) : type(type) {}
    ~Viewer() = default;

    /// Update the view and handle events (e.g. for controlling the camera).
    /// Must be thread-safe.
    virtual void update(float dt) = 0;

    /// Render the view and return the output colorbuffer.
    /// Must be thread-safe.
    virtual Colorbuffer render() = 0;
};
} // namespace async_torchwindow
