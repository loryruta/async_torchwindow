# async_torchwindow

`async_torchwindow` is an alternative to [`torchwindow`](https://github.com/jbaron34/TorchWindow/)
or to this [GitHub Gist](https://gist.github.com/victor-shepardson/3eb67c3664cde081a7e573376b1b0b54).

Shortly, it opens a window that allows you to visualize an image tensor.

- **No download to host memory**

  Many visualization libraries require image data to be on host before visualization. However, in many applications images are **device** tensors and downloading such tensors to host memory introduces latency, hindering real-time visualization.

- **No GIL lock**

  Differently from the cited alternatives, `async_torchwindow` allows to start the window asynchronously w.r.t. the caller thread (a Python thread). The callee thread is started from native code and thus won't interfere with the GIL lock, allowing Python to perform parallel tasks at full speed.

> ***NOTE:** This library was developed quickly (1 day) for personal use and, as such, lacks proper testing.
> If found useful, issues and PRs are warmly welcomed.*

Tested on:
- Ubuntu 22.04

## Build and Install

#### Dependencies

Would be great if your environment (virtualenv or conda) satisfies the following dependencies:

- Python >=3.11
- CUDA >=12.4
- PyTorch >=2.5
- scikit_build_core >=0.10

These are the versions with which I've developed the library, could possibly work with lower versions (not tested).

#### Clone, Build & Install

```
git submodule add https://github.com/loryruta/async_torchwindow
git submodule update --init --recursive
cd async_torchwindow
pip install .
```

## Usage

```py
from async_torchwindow import Window

H = 1080
W = 720

window = Window(H, W, "My window")
window.start(
    False)  # blocking = False

image = torch.rand((H, W, 3), dtype=torch.float32).cuda()

# Do some intensive task
#   ... *update image* ...
window.set_image(W, H, image.data_ptr())

# Ultimately when the App closes
window.destroy()
```

## Development

Some notes for collaborators or the future me in case I want to edit and test the library quick.

`pip install .` is how you would build the Python wheel, using `scikit_build_core`.
However **it's too slow**, this is why I prepared a test executable (`src/main.cu`).
To build it:

```
mkdir build
cd build
cmake ..
cmake --build .
./async_torchwindow
```
