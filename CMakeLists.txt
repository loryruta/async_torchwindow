cmake_minimum_required(VERSION 3.18...3.22)
project(async_torchwindow LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 20)

if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 75)
endif ()

# ----------------------------------------------------------------
# Dependencies
# ----------------------------------------------------------------

# Python
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
execute_process(
        COMMAND ${Python_EXECUTABLE} -c "import site; print(site.getsitepackages()[0])"
        OUTPUT_VARIABLE PYTHON_SITE_PACKAGES
        OUTPUT_STRIP_TRAILING_WHITESPACE)

# GLFW
set(BUILD_SHARED_LIBS OFF CACHE BOOL "")
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "")
set(GLFW_BUILD_TESTS OFF CACHE BOOL "")
set(GLFW_BUILD_DOCS OFF CACHE BOOL "")
set(GLFW_INSTALL OFF CACHE BOOL "")
add_subdirectory("third_party/glfw")

# GLAD
add_library(glad STATIC "third_party/glad/src/gl.c")
# relocation R_X86_64_PC32 against symbol `GLAD_GL_VERSION_1_0' can not be used when
# making a shared object; recompile with -fPIC
target_compile_options(glad PRIVATE "-fPIC")
target_include_directories(glad PUBLIC "third_party/glad/include")

# ----------------------------------------------------------------
# Executable
# ----------------------------------------------------------------

if (DEFINED SKBUILD)
    set(TARGET_NAME "_async_torchwindow")
else ()
    set(TARGET_NAME "async_torchwindow")
endif ()

set(SOURCES
        ./src/gaussian_splatting/rasterizer/forward.cu
        ./src/gaussian_splatting/rasterizer/rasterizer_impl.cu
        ./src/gaussian_splatting/GSViewer.cu
        ./src/gaussian_splatting/GSViewer.h
        ./src/image/ImageViewer.h
        ./src/utils.h
        ./src/Viewer.h
        ./src/Window.cu
        ./src/Window.h)

if (DEFINED SKBUILD)
    # python_add_library has the same semantic as add_library() but takes care of
    # Python module naming rules (only applied if library is of type MODULE) and
    # add dependency to target Python::Python
    python_add_library(${TARGET_NAME} MODULE
            ${SOURCES}
            ./src/pybind.cu
            WITH_SOABI)
    target_include_directories(${TARGET_NAME} PRIVATE
            third_party/pybind11/include)
else ()
    add_executable(${TARGET_NAME}
            ${SOURCES}
            ./src/main.cu)
endif ()

target_include_directories(${TARGET_NAME} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        third_party/glfw/include
        third_party/glm)
target_link_libraries(${TARGET_NAME} PRIVATE
        glfw
        glad)

if (DEFINED SKBUILD)
    install(TARGETS ${TARGET_NAME} DESTINATION async_torchwindow)
endif ()
