#include "Window.h"

#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <thread>

// clang-format off
#include <glad/gl.h>
#include <GLFW/glfw3.h>
// clang-format on
#include <cuda_gl_interop.h>

#include "utils.h"

/* Viewers */
#include "gaussian_splatting/GSViewer.h"
#include "image/ImageViewer.h"

using namespace async_torchwindow;

static int g_num_window_alive = 0;

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
 
    // Destroy the viewer if any
    if (m_viewer) m_viewer.reset();
    
    destroy();
}

std::pair<int, int> Window::get_size()
{
    int width, height;
    glfwGetWindowSize(m_window, &width, &height);
    return {width, height};
}

int Window::get_key(int key) const { return glfwGetKey(m_window, key); }

std::pair<double, double> Window::get_cursor_pos() const
{
    double pos_x, pos_y;
    glfwGetCursorPos(m_window, &pos_x, &pos_y);
    return {pos_x, pos_y};
}

int Window::get_cursor_mode() const { return glfwGetInputMode(m_window, GLFW_CURSOR); }

void Window::set_cursor_mode(int value) { glfwSetInputMode(m_window, GLFW_CURSOR, value); }

void Window::set_image(int width, int height, float* data_d)
{
    if (!m_viewer || m_viewer->type != ViewerType::IMAGE) {
        m_viewer = std::unique_ptr<Viewer>(new ImageViewer);
    }
    dynamic_cast<ImageViewer*>(m_viewer.get())->set_image(width, height, data_d);
}

void Window::set_gaussian_splatting_scene(int P,
                                          float* background_d,
                                          float* means3d_d,
                                          float* shs_d,
                                          int sh_degree,
                                          int M,
                                          float* opacity_d,
                                          float* scales_d,
                                          float* rotations_d)
{
    if (!m_viewer || m_viewer->type != ViewerType::GAUSSIAN_SPLATTING) {
        m_viewer = std::unique_ptr<Viewer>(new GSViewer(this));
    }
    dynamic_cast<GSViewer*>(m_viewer.get())
        ->set_scene(P, background_d, means3d_d, shs_d, sh_degree, M, opacity_d, scales_d, rotations_d);
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

    // Init screenbuffers
    auto [width, height] = get_size();
    resize_screenbuffers(width, height);

    /* Window loop */
    int frame_counter = 0;
    double fps_last_t = 0.0;
    double last_frame_t = -1.0;

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

        // Update window title
        char title[256];
        if (!m_viewer) {
            snprintf(title, sizeof(title), "Empty| FPS: %.1f", m_fps);
        } else {
            switch (m_viewer->type) {
            case IMAGE:
                snprintf(title, sizeof(title), "Image| FPS: %.1f", m_fps);
                break;
            case GAUSSIAN_SPLATTING:
                snprintf(title, sizeof(title), "Gaussian Splatting| FPS: %.1f", m_fps);
                break;
            default:
                throw std::runtime_error("Unknown viewer type");
            }
        }
        glfwSetWindowTitle(m_window, title);

        // Update
        double t = glfwGetTime();
        double dt = last_frame_t >= 0.0 ? t - last_frame_t : 0.0f;
        last_frame_t = t;
        if (m_viewer) m_viewer->update(float(dt));

        // Render
        render();

        //
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
            position.y = -position.y;
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
    glClearColor(0, 0, 0, 0);
    glClear(GL_COLOR_BUFFER_BIT);

    do {
        if (!m_viewer) break;
        Colorbuffer colorbuffer = m_viewer->render();
        if (colorbuffer.is_invalid()) break;

        // If the texture is smaller than the image tensor, crop the
        // tensor to fit the texture.
        int copy_region_w = std::min(colorbuffer.width, m_screenbuffer_w);
        int copy_region_h = std::min(colorbuffer.height, m_screenbuffer_h);

        // Copy tensor data to screen texture
        CHECK_CUDA(cudaMemcpy2DToArray(m_screen_array,
                                       0, // wOffset
                                       0, // hOffset
                                       colorbuffer.data_d,
                                       copy_region_w * 4 * sizeof(float), // spitch (tightly packed)
                                       copy_region_w * 4 * sizeof(float), // width
                                       copy_region_h,
                                       cudaMemcpyDeviceToDevice));
        CHECK_CUDA(cudaStreamSynchronize(cudaStreamPerThread));

        // Draw texture to screen
        glUseProgram(m_screenquad_program);

        glBindVertexArray(m_vao);
        glBindTexture(GL_TEXTURE_2D, m_screen_texture);
        glDrawArrays(GL_TRIANGLES, 0, 6);
    } while (false);
}
