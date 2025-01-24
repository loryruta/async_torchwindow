#include <pybind11/pybind11.h>

#include "Window.h"

using namespace async_torchwindow;

namespace py = pybind11;

PYBIND11_MODULE(_async_torchwindow, m)
{
    py::class_<Window>(m, "Window")
        .def(py::init<int, int>())
        .def("size", &Window::size)
        .def("fps", &Window::fps)
        .def("set_image",
             [](Window& window, int width, int height, uintptr_t data_d) {
                 window.set_image(width, height, (float*) data_d);
             })
        .def("set_gaussian_splatting_scene",
             [](Window& window,
                int P,
                uintptr_t background_d,
                uintptr_t means3d_d,
                uintptr_t shs_d,
                int sh_degree,
                int M,
                uintptr_t opacity_d,
                uintptr_t scales_d,
                uintptr_t rotations_d) {
                 window.set_gaussian_splatting_scene(P,
                                                     (float*) background_d,
                                                     (float*) means3d_d,
                                                     (float*) shs_d,
                                                     sh_degree,
                                                     M,
                                                     (float*) opacity_d,
                                                     (float*) scales_d,
                                                     (float*) rotations_d);
             })
        .def("start",
             [](Window& window) {
                 window.start(false /* blocking */); // Always non-blocking
             })
        .def("is_running", &Window::is_running)
        .def("destroy", &Window::destroy);
}
