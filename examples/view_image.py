from async_torchwindow import Window
from torch.autograd import Variable
import torch
import torch.nn.functional as F
from time import sleep

# Animation from:
# https://gist.github.com/victor-shepardson/5b3d3087dc2b4817b9bffdb8e87a57c4

W = 500
H = 500

window = Window(W, H)
window.start()


state = torch.cuda.FloatTensor(1, 3, H, W)
state.uniform_()
state = Variable(state, volatile=True)


def torch_process(state) -> torch.Tensor:
    """Random convolutions."""
    fs = 11
    filters, sgns = (
        Variable(init(torch.cuda.FloatTensor(3, 3, fs, fs)), volatile=True)
        for init in (lambda x: x.normal_(), lambda x: x.bernoulli_(0.52))
    )
    filters = F.softmax(filters) * (sgns * 2 - 1)
    state = F.conv2d(state, filters, padding=fs // 2)
    state = state - state.mean().expand(state.size())
    state = state / state.std().expand(state.size())
    return state


try:
    while window.is_running():
        state = torch_process(state).detach()
        img = F.tanh(state).abs()
        # Convert into proper format
        tensor = img.squeeze().permute(1, 2, 0).data  # Put in texture order
        tensor = torch.cat((tensor, tensor[:, :, 0].unsqueeze(-1)), dim=2)  # Add the alpha channel
        tensor[:, :, 3] = 1  # Set alpha
        
        window.set_image(tensor)
        
        sleep(0.01)  # Sleep not to overload the GPU too much
except KeyboardInterrupt:
    pass

window.destroy()
