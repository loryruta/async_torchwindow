#pragma once

#include <mutex>
#include <thread>

// clang-format off
#include <glad/gl.h>
#include <GLFW/glfw3.h>
// clang-format on

#include "Viewer.h"
#include "utils.h"

namespace async_torchwindow
{
class Window
{
private:
    GLFWwindow* m_window;

    /* Screenquad renderer */
    GLuint m_screenquad_program;
    GLuint m_vao;

    /* CUDA <-> OpenGL interop resources */
    cudaGraphicsResource_t m_screen_resource = nullptr;
    cudaArray_t m_screen_array = nullptr;
    GLuint m_screen_texture = 0;
    int m_screenbuffer_w = -1;
    int m_screenbuffer_h = -1;

    /* Viewer */
    std::unique_ptr<Viewer> m_viewer;

    std::unique_ptr<std::thread> m_loop_thread;
    std::mutex m_user_image_mutex;

    bool m_running = false;
    bool m_destroyed = false;
    double m_fps = 0;

public:
    Window(int width, int height, const char* title);
    ~Window();

    [[nodiscard]] GLFWwindow* handle() const { return m_window; }

    std::pair<int, int> get_size();

    [[nodiscard]] double get_fps() const { return m_fps; }

    [[nodiscard]] int get_key(int key) const;
    [[nodiscard]] std::pair<double, double> get_cursor_pos() const;

    [[nodiscard]] int get_cursor_mode() const;
    void set_cursor_mode(int mode);

    /// A function to let the user set an image to be displayed on the window.
    /// The image memory layout is expected to be (H, W, 4), and 32-bit floating per channel ranged [0, 1].
    ///
    /// @param image_width  The width of the image in pixels
    /// @param image_height The height of the image in pixels
    /// @param image_data_d Image data on device memory
    void set_image(int width, int height, float* data_d);

    void set_gaussian_splatting_scene(int P,
                                      float* background_d,
                                      float* means3d_d,
                                      float* shs_d,
                                      int sh_degree,
                                      int M,
                                      float* opacity_d,
                                      float* scales_d,
                                      float* rotations_d);

    /// Start the window rendering/event loop.
    void start(bool blocking = true);
    bool is_running() const;

    void destroy();

private:
    void setup_gl();
    void resize_screenbuffers(int width, int height);

    void start0();
    void render();
};
} // namespace async_torchwindow
