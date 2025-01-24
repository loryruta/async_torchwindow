# async_torchwindow

A PyTorch extension library to visualize an **Image tensor** or a **Gaussian Splatting scene** asynchronously to Python.

<p align="center">
<img src="https://github.com/user-attachments/assets/1094661a-905e-4de8-948e-d1505d785884" width="350">
<img src="https://github.com/user-attachments/assets/2c07d5e0-100a-4752-a085-0cace405a2c2" width="350">
</p>

- ⏬ **No download to host memory**

  Many visualization libraries require image data to be on host before visualization.
  However, in many applications images are already device tensors and downloading such tensors to host memory introduces latency, hindering real-time visualization.

- 🔓 **No GIL lock**

  Differently from the cited alternatives, `async_torchwindow` allows to start the window asynchronously w.r.t. the caller thread (Python).
  The visualization thread is started from native code and thus don't interfere with the GIL lock, allowing Python to perform subsequent tasks at full speed.

> ***NOTE:** This library was developed quickly for personal use and, as such, lacks proper testing.
> If found useful, issues and PRs are warmly welcomed.*

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

#### Visualize an Image

```py
from time import time
from math import *

from async_torchwindow import Window

H = 500
W = 500

window = Window(W, H)
window.start()

image = torch.rand((H, W, 3), dtype=torch.float32).cuda()

window.set_image(image)

try:
    while window.is_running():
        # Update the image
        image[:, :, 0] = abs(sin(time()))
        sleep(0.01)  # To avoid overloading GPU
except:
    pass

window.destroy()
```

#### Visualize a Gaussian Splatting scene

```py
from time import time
from math import *

from async_torchwindow import Window

H = 500
W = 500

window = Window(W, H)
window.start()

(means3d, rotations, scales, opacity, shs) = load_gaussian_splatting_scene(scene_dir)
background = torch.tensor([0, 0, 0], dtype=torch.float32)
window.set_gaussian_splatting_scene(
    background, # (3,)
    means3d,    # (P, 3)
    shs,        # (P, M, 3), usually (P, 16, 3)
    opacity,    # (P,)  , activated
    scales,     # (P, 3), activated
    rotations   # (P, 4), activated
)

try:
    while window.is_running():
        # Update the Red DC component of SH (could be any scene's tensor) 
        shs[:, 0, 0] = abs(sin(time()))
        sleep(0.01)  # To avoid overloading GPU
except KeyboardInterrupt:
    pass

window.destroy()
```

Guess what? You can visualize a Gaussian Splatting scene during training! 🤩

## Development

Some notes for contributors or the future me in case I want to edit and test the library quick.

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
