#include "GSViewer.h"

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/transform.hpp>

#include "rasterizer/rasterizer.h"

#include "Window.h"

using namespace async_torchwindow;

namespace
{
template <typename T> std::function<char*(size_t N)> resize_functional(Buffer& buffer)
{
    return [&buffer](size_t N) -> char* {
        buffer.resize(N * sizeof(T));
        return reinterpret_cast<char*>(buffer.data_d);
    };
};
} // namespace

GSViewer::GSViewer(Window* window) : Viewer(ViewerType::GAUSSIAN_SPLATTING), m_window(window)
{
    auto [width, height] = window->get_size();
    m_camera.width = width;
    m_camera.height = height; // TODO resize fx and fy as well
    m_camera.update();

    // Init screenbuffer (colorbuffer) to the initial window size
    resize_screenbuffers(width, height);
}

GSViewer::~GSViewer()
{
    m_geometry_buffer.destroy();
    m_binning_buffer.destroy();
    m_image_buffer.destroy();
}

void GSViewer::resize_screenbuffers(int width, int height)
{
    if (m_colorbuffer.data_d) {
        if (m_colorbuffer.width >= width && m_colorbuffer.height >= height) {
            return; // Colorbuffer is big enough
        } else {
            m_colorbuffer.destroy();
        }
    }

    m_colorbuffer.width = width;
    m_colorbuffer.height = height;
    m_colorbuffer.allocate();
}

void GSViewer::set_scene(int P,
                         float* background_d,
                         float* means3d_d,
                         float* shs_d,
                         int sh_degree,
                         int M,
                         float* opacity_d,
                         float* scales_d,
                         float* rotations_d)
{
    std::lock_guard<std::mutex> lock(m_mutex);

    m_scene.P = P;
    m_scene.background_d = background_d;
    m_scene.means3d_d = means3d_d;
    m_scene.shs_d = shs_d;
    m_scene.sh_degree = sh_degree;
    m_scene.M = M;
    m_scene.opacity_d = opacity_d;
    m_scene.scales_d = scales_d;
    m_scene.rotations_d = rotations_d;
}

void GSViewer::update(float dt)
{
    static const float k_camera_speed = 1.0f;
    static const float k_camera_sensitivity = 0.1f;

    bool do_update = false;

    // Handle camera movement
    glm::vec3 dir{};
    if (m_window->get_key(GLFW_KEY_W)) dir += m_camera.forward();
    if (m_window->get_key(GLFW_KEY_S)) dir -= m_camera.forward();
    if (m_window->get_key(GLFW_KEY_A)) dir -= m_camera.right();
    if (m_window->get_key(GLFW_KEY_D)) dir += m_camera.right();
    if (m_window->get_key(GLFW_KEY_SPACE)) dir += m_camera.up();
    if (m_window->get_key(GLFW_KEY_LEFT_SHIFT)) dir -= m_camera.up();
    if (dir != glm::vec3(0)) {
        m_camera.position += glm::normalize(dir) * k_camera_speed * dt;
        do_update = true;
    }

    // Handle camera rotation
    auto [cur_x, cur_y] = m_window->get_cursor_pos();
    if (m_last_cursor_pos) {
        float cur_dx = (float) (cur_x - m_last_cursor_pos->x);
        float cur_dy = (float) (cur_y - m_last_cursor_pos->y);
        // Rotate yaw
        if (cur_dx > 0) {
            m_camera.rotation =
                glm::rotate(glm::mat4(m_camera.rotation), cur_dx * k_camera_sensitivity * dt, m_camera.up());
            do_update = true;
        }
        // Rotate pitch
        if (cur_dy > 0) {
            m_camera.rotation =
                glm::rotate(glm::mat4(m_camera.rotation), cur_dy * k_camera_sensitivity * dt, m_camera.right());
            do_update = true;
        }
    }
    m_last_cursor_pos = {cur_x, cur_y};

    if (do_update) {
        m_camera.update();
    }
}

Colorbuffer GSViewer::render()
{
    Scene scene;
    // Thread-safe copy the current scene parameters.
    // This to avoid situations where the user calls set_scene and this
    // function is executing.
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        scene = m_scene;
    }
    // We don't need the same for the camera as it's modified only by
    // the rendering thread.

    int P = scene.P;
    if (P == 0) return Colorbuffer{}; // Invalid

    // clang-format off
    CudaRasterizer::Rasterizer::forward(
        resize_functional<char>(m_geometry_buffer),
        resize_functional<char>(m_binning_buffer),
        resize_functional<char>(m_image_buffer),
        P,
        scene.sh_degree,
        scene.M,
        scene.background_d,
        m_colorbuffer.width,
        m_colorbuffer.height,
        scene.means3d_d,
        scene.shs_d,
        nullptr, // colors_precomp
        scene.opacity_d,
        scene.scales_d,
        1.0f, // scale_modifier
        scene.rotations_d,
        nullptr, // cov3D_precomp
        m_camera.viewmatrix_d,
        m_camera.projmatrix_d,
        m_camera.campos_d,
        m_camera.tan_fovx,
        m_camera.tan_fovy,
        false, // prefiltered
        m_colorbuffer.data_d, // (H, W, 4)
        nullptr, // radii
        false // debug
    );
    // clang-format on

    return m_colorbuffer;
}
