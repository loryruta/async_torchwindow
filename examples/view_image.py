from async_torchwindow import Window
import torch

W = 500
H = 500

window = Window(W, H)
window.start()

# Run asynchronously w.r.t. the window
try:
    while window.is_running():
        image = torch.rand((H, W, 4), dtype=torch.float32).cuda()
        image[:, :, 3] = 1.0
        window.set_image(image)
except KeyboardInterrupt:
    pass

window.destroy()
