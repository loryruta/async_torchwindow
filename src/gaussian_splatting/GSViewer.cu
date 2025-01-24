#include "GSViewer.h"

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/transform.hpp>

#include "rasterizer/rasterizer.h"

#include "Window.h"

using namespace async_torchwindow;

namespace
{
template <typename T>
std::function<char*(size_t N)> resize_functional(Buffer& buffer)
{
    return [&buffer](size_t N) -> char* {
        bool resized = buffer.resize(N * sizeof(T));
        if (resized) {
            return reinterpret_cast<char*>(buffer.data_d);
        } else {
            return nullptr;
        }
    };
};
} // namespace

GSViewer::GSViewer(Window* window) : Viewer(ViewerType::GAUSSIAN_SPLATTING), m_window(window)
{
    auto [width, height] = window->size();
    m_camera.width = width;
    m_camera.height = height; // TODO resize fx and fy as well
    m_camera.update();

    // Init screenbuffer (colorbuffer) to the initial window size
    resize_screenbuffers(width, height);

    // Register window listeners
    GLFWwindow* w_handle = window->handle();
    void* listener = glfwGetWindowUserPointer(w_handle);
    if (listener) throw std::runtime_error("Window already has a listener");
    glfwSetWindowUserPointer(w_handle, this);
    glfwSetMouseButtonCallback(w_handle, [](GLFWwindow* w_handle, int button, int action, int mods) {
        if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS) {
            glfwSetInputMode(w_handle, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
        }
    });
    glfwSetKeyCallback(w_handle, [](GLFWwindow* w_handle, int key, int scancode, int action, int mods) {
        if (glfwGetInputMode(w_handle, GLFW_CURSOR) == GLFW_CURSOR_NORMAL) {
            return;
        }
        if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
            glfwSetInputMode(w_handle, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
        }
    });
}

GSViewer::~GSViewer()
{
    LOG("Destroying GSViewer");

    // Unregister window listeners
    GLFWwindow* w_handle = m_window->handle();
    glfwSetMouseButtonCallback(w_handle, nullptr);
    glfwSetKeyCallback(w_handle, nullptr);
    glfwSetWindowUserPointer(w_handle, nullptr);

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

    /* Handle camera input */
    do {
        GLFWwindow* w_handle = m_window->handle();

        if (glfwGetInputMode(w_handle, GLFW_CURSOR) != GLFW_CURSOR_DISABLED) break;

        bool do_update = false;

        // Handle camera movement
        float speed = k_camera_speed;
        glm::vec3 dir{};
        if (glfwGetKey(w_handle, GLFW_KEY_W) == GLFW_PRESS) dir += m_camera.forward();
        if (glfwGetKey(w_handle, GLFW_KEY_S) == GLFW_PRESS) dir -= m_camera.forward();
        if (glfwGetKey(w_handle, GLFW_KEY_A) == GLFW_PRESS) dir -= m_camera.right();
        if (glfwGetKey(w_handle, GLFW_KEY_D) == GLFW_PRESS) dir += m_camera.right();
        if (glfwGetKey(w_handle, GLFW_KEY_SPACE) == GLFW_PRESS) dir -= m_camera.up(); // Y negative
        if (glfwGetKey(w_handle, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS) dir += m_camera.up();
        if (glfwGetKey(w_handle, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS) speed *= 10.0f;
        if (dir != glm::vec3(0)) {
            m_camera.position += glm::normalize(dir) * speed * dt;
            do_update = true;
        }

        // Handle camera rotation
        double cur_x, cur_y;
        glfwGetCursorPos(w_handle, &cur_x, &cur_y);
        if (m_last_cursor_pos) {
            float cur_dx = (float) (cur_x - m_last_cursor_pos->x);
            float cur_dy = (float) (cur_y - m_last_cursor_pos->y);
            // Rotate yaw
            if (cur_dx != 0) {
                m_camera.yaw += cur_dx * k_camera_sensitivity * dt;
                do_update = true;
            }
            // Rotate pitch
            if (cur_dy != 0) {
                m_camera.pitch += cur_dy * k_camera_sensitivity * dt;
                do_update = true;
            }
        }
        m_last_cursor_pos = {cur_x, cur_y};

        if (do_update) {
            m_camera.update();
        }
    } while (false);
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
    int result = CudaRasterizer::Rasterizer::forward(
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

    // An error occurred during forward (e.g. Out of memory)
    if (result < 0) return Colorbuffer{};

    return m_colorbuffer;
}
