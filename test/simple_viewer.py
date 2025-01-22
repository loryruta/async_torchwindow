from async_torchwindow import Window
import torch
import torch.nn.functional as F
from torch.autograd import Variable

W = 300
H = 300

window = Window(W, H, "PyTorch window")
window.start(False)

# Run asynchronously w.r.t. the window
try:
    while window.is_running():
        print(f"FPS: {window.get_fps():.3f}")
        
        image = torch.rand((H, W, 4), dtype=torch.float32).cuda()
        image[:, :, 3] = 1.0
        
        window.set_image(W, H, image.data_ptr())
except KeyboardInterrupt:
    pass

window.destroy()
