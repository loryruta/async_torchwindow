#pragma once

#include <mutex>
#include <optional>

#include <glm/glm.hpp>

#include "Camera.h"
#include "Viewer.h"

namespace async_torchwindow
{
// Forward decl
class Window;

class GSViewer : public Viewer
{
private:
    Window* m_window;

    Camera m_camera;

    /* Render buffers */
    Buffer m_geometry_buffer{"geometryBuffer"};
    Buffer m_binning_buffer{"binningBuffer"};
    Buffer m_image_buffer{"imageBuffer"};

    /* Gaussian Splatting scene */
    struct Scene {
        int P = 0;
        float* background_d = nullptr; // (3,)
        float* means3d_d = nullptr;    // (P, 3)
        float* shs_d = nullptr;        // (P, M, 3)
        int sh_degree = 0;
        int M = 0;
        float* opacity_d = nullptr;   // (P,)
        float* scales_d = nullptr;    // (P, 3)
        float* rotations_d = nullptr; // (P, 4)
    } m_scene;

    Colorbuffer m_colorbuffer;

    std::optional<glm::dvec2> m_last_cursor_pos{};

    std::mutex m_mutex;

public:
    explicit GSViewer(Window* window);
    GSViewer(const GSViewer&) = delete;
    GSViewer(GSViewer&&) = delete;
    ~GSViewer();

    void resize_screenbuffers(int width, int height);

    /// Set the Gaussian Splatting scene to visualize.
    void set_scene(int P,
                   float* background_d,
                   float* means3d_d,
                   float* shs_d,
                   int sh_degree,
                   int M,
                   float* opacity_d,
                   float* scales_d,
                   float* rotations_d);

    void update(float dt) override;

    /// Render the set Gaussian Splatting scene and return a pointer to the internal color buffer.
    ///
    /// @return
    ///     A device pointer to the color buffer. Memory layout: `(H, W, 4)`.
    Colorbuffer render() override;
};
} // namespace async_torchwindow
