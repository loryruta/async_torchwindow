#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "utils.h"

namespace async_torchwindow
{
inline glm::mat4 build_projection_matrix(float fov_x, float fov_y, float z_near, float z_far)
{
    float tan_half_fovY = tan((fov_y / 2));
    float tan_half_fovX = tan((fov_x / 2));
    float top = tan_half_fovY * z_far;
    float bottom = -top;
    float right = tan_half_fovX * z_near;
    float left = -right;

    float z_sign = 1.0f;

    glm::mat4 P;
    P[0][0] = 2.0 * z_near / (right - left);
    P[1][1] = 2.0 * z_near / (top - bottom);
    P[0][2] = (right + left) / (right - left);
    P[1][2] = (top + bottom) / (top - bottom);
    P[3][2] = z_sign;
    P[2][2] = z_sign * z_far / (z_far - z_near);
    P[2][3] = -(z_far * z_near) / (z_far - z_near);
    return P;
}

struct Camera {
    glm::vec3 position{};
    glm::mat3 rotation = glm::mat3(1.0f);
    float fx = 1.0f;
    float fy = 1.0f;
    int width = 1080;
    int height = 720;
    float z_near = 0.01f;
    float z_far = 1000.0f;

    /* GPU data */
    float* campos_d = nullptr;     ///< Camera position on GPU
    float* viewmatrix_d = nullptr; ///< View matrix on GPU
    float* projmatrix_d = nullptr; ///< Proj @ View matrix on GPU
    float tan_fovx;
    float tan_fovy;

    explicit Camera()
    {
        CHECK_CUDA(cudaMalloc(&campos_d, 3 * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&viewmatrix_d, 16 * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&projmatrix_d, 16 * sizeof(float)));
    }

    ~Camera() = default;

    glm::vec3 right() const { return glm::normalize(rotation[0]); }
    glm::vec3 up() const { return glm::normalize(rotation[1]); }
    glm::vec3 forward() const { return glm::normalize(rotation[2]); }

    void update()
    {
        // View matrix
        glm::mat4 Rt = rotation;
        Rt[3] = glm::vec4(position, 1.0f);
        glm::mat4 view_matrix = glm::inverse(Rt);

        // Projection matrix
        float fov_y = 2 * atan(1.0f / fy);
        float fov_x = 2 * atan(1.0f / fx);
        tan_fovx = tan(fov_x * 0.5f);
        tan_fovy = tan(fov_y * 0.5f);
        glm::mat4 proj_matrix = build_projection_matrix(fov_x, fov_y, z_near, z_far);
        glm::mat4 viewproj_matrix = proj_matrix * view_matrix;

        // Upload to GPU
        CHECK_CUDA(cudaMemcpy(campos_d, glm::value_ptr(position), 3 * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(viewmatrix_d, glm::value_ptr(view_matrix), 16 * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(
            cudaMemcpy(projmatrix_d, glm::value_ptr(viewproj_matrix), 16 * sizeof(float), cudaMemcpyHostToDevice));
    }
};
} // namespace async_torchwindow
