#include "lib.h"

#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <thread>

// clang-format off
#include <glad/gl.h>
#include <GLFW/glfw3.h>
// clang-format on
#include <cuda_gl_interop.h>

using namespace async_torchwindow;

static int g_num_window_alive = 0;

void async_torchwindow::check_cuda(cudaError_t error, char const* file, int line)
{
    if (error != cudaSuccess) {
        const char* cuda_error = cudaGetErrorString(error);
        LOG("CUDA error: %s (%s:%d)", cuda_error, file, line);
        throw std::runtime_error(cuda_error);
    }
}

Window::Window(int width, int height, const char* title)
{
    if (g_num_window_alive == 0) {
        if (!glfwInit()) {
            throw std::runtime_error("Failed to initialize GLFW");
        }
        LOG("GLFW initialized\n");
    }
    ++g_num_window_alive;

    // TODO LIMITATION: the window is not resizable now
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
    // The window is visible only after calling Window::start()
    glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);

    m_window = glfwCreateWindow(width, height, title, NULL, NULL);
    if (!m_window) {
        throw std::runtime_error("Failed to create GLFW window");
    }

    glfwSetWindowUserPointer(m_window, this);
}

Window::~Window()
{
    if (m_destroyed) return;

    destroy();
}

std::pair<int, int> Window::get_size()
{
    int width, height;
    glfwGetWindowSize(m_window, &width, &height);
    return {width, height};
}

void Window::set_title(const char* title) { glfwSetWindowTitle(m_window, title); }

int Window::get_key(int key) const { return glfwGetKey(m_window, key); }

std::pair<double, double> Window::get_cursor_pos() const
{
    double pos_x, pos_y;
    glfwGetCursorPos(m_window, &pos_x, &pos_y);
    return {pos_x, pos_y};
}

int Window::get_cursor_mode() const { return glfwGetInputMode(m_window, GLFW_CURSOR); }

void Window::set_cursor_mode(int value) { glfwSetInputMode(m_window, GLFW_CURSOR, value); }

void Window::set_image(int image_width, int image_height, const void* image_data_d)
{
    std::lock_guard<std::mutex> lock(m_user_image_mutex);
    m_user_image_w = image_width;
    m_user_image_h = image_height;
    m_user_image_d = image_data_d;
}

void Window::start0()
{
    // Initialize OpenGL now (possibly not on the main-thread)
    glfwMakeContextCurrent(m_window);
    int version = gladLoadGL(glfwGetProcAddress);
    if (version == 0) {
        throw std::runtime_error("Failed to initialize GL context");
    }
    LOG("GL initialized (version %d.%d)\n", GLAD_VERSION_MAJOR(version), GLAD_VERSION_MINOR(version));

    // Setup screenquad renderer
    setup_gl();

    // Initialize screenbuffers
    auto [width, height] = get_size();
    resize_screenbuffers(width, height);

    // Rendering/event loop
    int frame_counter = 0;
    double fps_last_t = 0.0;
    while (m_running) {
        // FPS
        ++frame_counter;
        double fps_t = glfwGetTime();
        double fps_dt = fps_t - fps_last_t;
        if (fps_dt >= 1.0) {
            m_fps = double(frame_counter) / fps_dt;
            frame_counter = 0;
            fps_last_t = fps_t;
        }

        render();

        glfwSwapBuffers(m_window);
        glfwPollEvents();

        m_running &= !glfwWindowShouldClose(m_window);
    }

    // Only hide the window and leave the destruction to the user
    // (call Window::destroy() method)
    glfwHideWindow(m_window);
}

void Window::start(bool blocking)
{
    if (m_running) {
        throw std::runtime_error("Window is already started");
    }
    m_running = true;

    glfwShowWindow(m_window);

    if (blocking) {
        // Start synchronously
        start0();
    } else {
        // Start asynchronously
        m_loop_thread = std::make_unique<std::thread>([this]() { start0(); });
    }
}

bool Window::is_running() const { return m_running; }

void Window::destroy()
{
    if (m_destroyed) return;

    if (m_running) {
        LOG("Sending the close signal...\n");
        m_running = false;
    }

    if (m_loop_thread) {
        m_loop_thread->join();
        m_loop_thread.reset();
    }

    glfwDestroyWindow(m_window);
    LOG("Window destroyed\n");

    --g_num_window_alive;
    if (g_num_window_alive == 0) {
        glfwTerminate();
        LOG("GLFW terminated\n");
    }

    m_destroyed = true;
}

void Window::setup_gl()
{
    // Create screenquad program
    static const char* s_vert_shader_src = R"(#version 460 core

        out vec2 v_texcoord;

        void main()
        {
            const vec2 k_texcoords[] = vec2[](
                vec2(0.0, 0.0), // 0
                vec2(1.0, 0.0), // 1
                vec2(0.0, 1.0), // 3
                vec2(1.0, 0.0), // 1
                vec2(1.0, 1.0), // 2
                vec2(0.0, 1.0)  // 3
            );

            vec2 position = k_texcoords[gl_VertexID];
            position = position * 2.0 - 1.0;
            gl_Position = vec4(position, 0.0, 1.0);

            v_texcoord = k_texcoords[gl_VertexID];
        }
    )";

    static const char* s_frag_shader_src = R"(#version 460 core

        in vec2 v_texcoord;

        uniform sampler2D u_texture;

        layout(location = 0) out vec4 f_color;

        void main()
        {
            f_color = texture(u_texture, v_texcoord);
        }
    )";

    GLuint v_shader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(v_shader, 1, &s_vert_shader_src, NULL);
    glCompileShader(v_shader);

    GLuint f_shader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(f_shader, 1, &s_frag_shader_src, NULL);
    glCompileShader(f_shader);

    m_screenquad_program = glCreateProgram();
    glAttachShader(m_screenquad_program, v_shader);
    glAttachShader(m_screenquad_program, f_shader);
    glLinkProgram(m_screenquad_program);

    // Create VAO
    glGenVertexArrays(1, &m_vao);

    LOG("GL screenquad renderer setup\n");
}

void Window::resize_screenbuffers(int width, int height)
{
    if (m_screen_resource) {
        LOG("Resizing to (%d, %d), deleting old screenbuffers...", width, height);
        CHECK_CUDA(cudaGraphicsUnmapResources(1, &m_screen_resource));
        CHECK_CUDA(cudaGraphicsUnregisterResource(m_screen_resource));
    }
    if (m_screen_texture) {
        glDeleteTextures(1, &m_screen_texture);
    }

    // Create OpenGL screen texture
    glGenTextures(1, &m_screen_texture);
    glBindTexture(GL_TEXTURE_2D, m_screen_texture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, width, height, 0, GL_RGBA, GL_FLOAT, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glBindTexture(GL_TEXTURE_2D, 0);

    // Create a CUDA mapped resource to access the screen texture
    CHECK_CUDA(cudaGraphicsGLRegisterImage(
        &m_screen_resource, m_screen_texture, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard));
    CHECK_CUDA(cudaGraphicsMapResources(1, &m_screen_resource));
    CHECK_CUDA(cudaGraphicsSubResourceGetMappedArray(&m_screen_array, m_screen_resource, 0, 0));

    m_screenbuffer_w = width;
    m_screenbuffer_h = height;

    LOG("Screenbuffers resized to (%d, %d)\n", width, height);
}

void Window::render()
{
    if (m_user_image_d) {
        int user_image_w;
        int user_image_h;
        const void* user_image_d;
        // Thread-safe access to user set data
        {
            std::lock_guard<std::mutex> lock(m_user_image_mutex);
            user_image_w = m_user_image_w;
            user_image_h = m_user_image_h;
            user_image_d = m_user_image_d;
        }

        // If the texture is smaller than the image tensor, crop the
        // tensor to fit the texture.
        int copy_region_w = min(user_image_w, m_screenbuffer_w);
        int copy_region_h = min(user_image_h, m_screenbuffer_h);

        // Copy tensor data to screen texture
        CHECK_CUDA(cudaMemcpy2DToArray(m_screen_array,
                                       0, // wOffset
                                       0, // hOffset
                                       user_image_d,
                                       copy_region_w * 4 * sizeof(float), // spitch (tightly packed)
                                       copy_region_w * 4 * sizeof(float), // width
                                       copy_region_h,
                                       cudaMemcpyDeviceToDevice));
        CHECK_CUDA(cudaDeviceSynchronize());

        // Draw texture to screen
        glUseProgram(m_screenquad_program);

        glBindVertexArray(m_vao);
        glBindTexture(GL_TEXTURE_2D, m_screen_texture);
        glDrawArrays(GL_TRIANGLES, 0, 6);
    } else {
        // No tensor to draw, display a blue image
        glClearColor(0, 1, 0, 0);
        glClear(GL_COLOR_BUFFER_BIT);
    }
}

#ifdef PYTHON_BINDINGS
#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_MODULE(_async_torchwindow, m)
{
    py::class_<Window>(m, "Window")
        .def(py::init<int, int, const char*>())
        .def("get_size", &Window::get_size)
        .def("get_fps", &Window::get_fps)
        .def("set_title", &Window::set_title)
        .def("get_key", &Window::get_key)
        .def("get_cursor_pos", &Window::get_cursor_pos)
        .def("get_cursor_mode", &Window::get_cursor_mode)
        .def("set_cursor_mode", &Window::set_cursor_mode)
        .def("set_image",
             [](Window& window, int image_width, int image_height, uintptr_t image_data_d) {
                 window.set_image(image_width, image_height, (const void*) image_data_d);
             })
        .def("start", &Window::start)
        .def("is_running", &Window::is_running)
        .def("destroy", &Window::destroy);
}
#endif
