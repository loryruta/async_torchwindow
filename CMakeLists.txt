cmake_minimum_required(VERSION 3.18...3.22)
project(async_torchwindow LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 20)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 75)
endif()

# ----------------------------------------------------------------
# Dependencies
# ----------------------------------------------------------------

# Python
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
execute_process(
    COMMAND ${Python_EXECUTABLE} -c "import site; print(site.getsitepackages()[0])"
    OUTPUT_VARIABLE PYTHON_SITE_PACKAGES
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

# pybind11
set(pybind11_ROOT "${PYTHON_SITE_PACKAGES}/pybind11/share/cmake/")
find_package(pybind11 CONFIG REQUIRED)

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
if(DEFINED SKBUILD)
    set(TARGET_NAME "_async_torchwindow")
else()
    set(TARGET_NAME "async_torchwindow")
endif()

if(DEFINED SKBUILD)
    # python_add_library have the same semantic as add_library() but take care of
    # Python module naming rules (only applied if library is of type MODULE) and
    # add dependency to target Python::Python
    python_add_library(${TARGET_NAME} MODULE
        "./src/lib.cu"
        WITH_SOABI
    )
    target_compile_definitions(${TARGET_NAME} PRIVATE -DPYTHON_BINDINGS)
    target_include_directories(${TARGET_NAME} PRIVATE ${pybind11_INCLUDE_DIRS})
else()
    add_executable(${TARGET_NAME}
        "./src/lib.cu"
        "./src/main.cu"
    )
endif()

# pybind11 headers are already included in PyTorch
target_include_directories(${TARGET_NAME} PRIVATE
    "third_party/glfw/include"
)
target_link_libraries(${TARGET_NAME} PRIVATE
    glfw
    glad
)

if(DEFINED SKBUILD)
    install(TARGETS ${TARGET_NAME} DESTINATION async_torchwindow)
endif()
