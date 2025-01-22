#pragma once

#include <mutex>
#include <thread>

// clang-format off
#include <glad/gl.h>
#include <GLFW/glfw3.h>
// clang-format on

#define CHECK_CUDA(_error) check_cuda(_error, __FILE__, __LINE__)
#define LOG(format, ...) printf("%s" format, "[async_torchwindow] ", ##__VA_ARGS__)

namespace async_torchwindow
{
void check_cuda(cudaError_t error, const char* file, int line);

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

    /* User-provided tensor */
    int m_user_image_w = -1;
    int m_user_image_h = -1;
    const void* m_user_image_d = nullptr;

    std::unique_ptr<std::thread> m_loop_thread;
    std::mutex m_user_image_mutex;

    bool m_running = false;
    bool m_destroyed = false;
    double m_fps = 0;

public:
    Window(int width, int height, const char* title);
    ~Window();

    std::pair<int, int> get_size();

    double get_fps() const { return m_fps; }

    void set_title(const char* title);

    int get_key(int key) const;
    std::pair<double, double> get_cursor_pos() const;

    int get_cursor_mode() const;
    void set_cursor_mode(int mode);

    /// A function to let the user set an image to be displayed on the window.
    /// The image memory layout is expected to be (H, W, 4), and 32-bit floating per channel ranged [0, 1].
    ///
    /// @param image_width         The width of the image in pixels
    /// @param image_height        The height of the image in pixels
    /// @param image_data_cuda_ptr Image data on device memory
    void set_image(int image_width, int image_height, const void* image_data_cuda_ptr);

    /// Start the window rendering/event loop.
    void start(bool blocking = true);
    bool is_running() const;
    void close();

    void destroy();

private:
    void setup_gl();
    void resize_screenbuffers(int width, int height);

    void start0();
    void render();
};
} // namespace async_torchwindow
