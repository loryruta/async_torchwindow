#include <cstdio>

#include "Window.h"

using namespace async_torchwindow;

#define WINDOW_W 1080
#define WINDOW_H 720

namespace
{
__global__ void fill_image_creatively_kernel(int width, int height, float* image_data, float t)
{
    int x = blockIdx.x * 16 + threadIdx.x;
    int y = blockIdx.y * 16 + threadIdx.y;
    if (x >= width || y >= height) {
        return;
    }

    int base_addr = (y * width + x) * 4;
    image_data[base_addr + 0] = abs(sin(x * 1000.0f)) * (cos(t) * .5f + 0.5f);
    image_data[base_addr + 1] = abs(cos(y * 1000.0f)) * (cos(t * 0.01f) * .5f + .5f);
    image_data[base_addr + 2] = abs(sin(x * 1000.0f)) * (cos(t * 0.1f) * .5f + .5f);
    image_data[base_addr + 3] = 1.0f;
}

void fill_image_creatively(int width, int height, float* image_data, float t)
{
    dim3 num_blocks;
    num_blocks.x = (width + 15) / 16;
    num_blocks.y = (height + 15) / 16;
    num_blocks.z = 1;
    dim3 block_dim;
    block_dim.x = 16;
    block_dim.y = 16;
    block_dim.z = 1;
    fill_image_creatively_kernel<<<num_blocks, block_dim>>>(width, height, image_data, t);
}

} // namespace

int main(int argc, char* argv[])
{
    Window window(WINDOW_W, WINDOW_H, "My window");

    // Create a device image to display
    int image_w = WINDOW_W;
    int image_h = WINDOW_H;
    float* image_d;
    CHECK_CUDA(cudaMalloc(&image_d, image_w * image_h * 4 * sizeof(float)));

    //window.set_image(image_w, image_h, image_d);

    window.start(false /* blocking */);

    while (window.is_running()) {
        //auto [cur_pos_x, cur_pos_y] = window.get_cursor_pos();
        // LOG("Cursor: %.2f, %.2f\n", cur_pos_x, cur_pos_y);

        double t = glfwGetTime();
        fill_image_creatively(image_w, image_h, image_d, float(t));
        CHECK_CUDA(cudaStreamSynchronize()); // Needed to avoid queuing too many operations on device
        // TODO modify the image on a separate CUDA stream?
    }

    window.destroy();

    return 0;
}
