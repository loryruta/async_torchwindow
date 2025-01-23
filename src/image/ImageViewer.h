#pragma once

#include <mutex>

#include "Viewer.h"

namespace async_torchwindow
{
class ImageViewer : public Viewer
{
private:
    Colorbuffer m_colorbuffer{};

    std::mutex m_mutex;

public:
    explicit ImageViewer() : Viewer(ViewerType::IMAGE) {};
    ~ImageViewer() = default;

    /// Set the image to visualize.
    void set_image(int width, int height, const float* data_d)
    {
        std::lock_guard<std::mutex> lock(m_mutex);

        m_colorbuffer.width = width;
        m_colorbuffer.height = height;
        m_colorbuffer.data_d = const_cast<float*>(data_d);
    }

    void update(float dt) override {}

    Colorbuffer render() override
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_colorbuffer;
    }
};
} // namespace async_torchwindow
