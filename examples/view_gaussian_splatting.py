from time import time, sleep
from os import path
import torch
import math
from plyfile import PlyData
from async_torchwindow import Window


def load_scene(scene_folder: str):
    ply_filepath = path.join(
        scene_folder, "point_cloud/iteration_30000/point_cloud.ply"
    )
    with open(ply_filepath, "rb") as f:
        ply_data = PlyData.read(f)

    x = torch.from_numpy(ply_data["vertex"]["x"])
    y = torch.from_numpy(ply_data["vertex"]["y"])
    z = torch.from_numpy(ply_data["vertex"]["z"])
    # nx = torch.from_numpy(ply_data['vertex']['nx'])
    # ny = torch.from_numpy(ply_data['vertex']['ny'])
    # nz = torch.from_numpy(ply_data['vertex']['nz'])
    color_list = []
    f_dc_0 = torch.from_numpy(ply_data["vertex"]["f_dc_0"])  # (N,)
    f_dc_1 = torch.from_numpy(ply_data["vertex"]["f_dc_1"])  # (N,)
    f_dc_2 = torch.from_numpy(ply_data["vertex"]["f_dc_2"])  # (N,)
    color_list.append(torch.stack([f_dc_0, f_dc_1, f_dc_2], dim=1))  # (N, 3)
    # colors = torch.stack([f_dc_0, f_dc_1, f_dc_2], dim=1)
    for i in range(0, 45, 3):
        f_rest_0 = torch.from_numpy(ply_data["vertex"][f"f_rest_{i}"])
        f_rest_1 = torch.from_numpy(ply_data["vertex"][f"f_rest_{i+1}"])
        f_rest_2 = torch.from_numpy(ply_data["vertex"][f"f_rest_{i+2}"])
        color_list.append(torch.stack([f_rest_0, f_rest_1, f_rest_2], dim=1))  # (N, 3)
    shs = torch.stack(color_list, dim=0).permute(1, 0, 2)  # (N, S, 3)
    assert shs.ndim == 3 and shs.shape[1] == 16 and shs.shape[2] == 3

    opacity = torch.from_numpy(ply_data["vertex"]["opacity"])
    scale_0 = torch.from_numpy(ply_data["vertex"]["scale_0"])
    scale_1 = torch.from_numpy(ply_data["vertex"]["scale_1"])
    scale_2 = torch.from_numpy(ply_data["vertex"]["scale_2"])
    rot_0 = torch.from_numpy(ply_data["vertex"]["rot_0"])
    rot_1 = torch.from_numpy(ply_data["vertex"]["rot_1"])
    rot_2 = torch.from_numpy(ply_data["vertex"]["rot_2"])
    rot_3 = torch.from_numpy(ply_data["vertex"]["rot_3"])

    means3d = torch.stack([x, y, z], dim=1).contiguous().cuda()
    rotations = torch.stack([rot_0, rot_1, rot_2, rot_3], dim=1).contiguous().cuda()
    scales = torch.stack([scale_0, scale_1, scale_2], dim=1).contiguous().cuda()
    opacity = opacity.contiguous().cuda()
    shs = shs.contiguous().cuda()

    # Apply activations
    scales = torch.exp(scales)
    rotations = torch.nn.functional.normalize(rotations, dim=1)
    opacity = torch.sigmoid(opacity)

    return (means3d, rotations, scales, opacity, shs)


W = 500
H = 500

print("Loading the scene...")
(means3d, rotations, scales, opacity, shs) = load_scene(
    "/home/loryruta/projects/3dgs-stereo/data/scenes/train"
)

window = Window(W, H)
window.start()

# Set the Gaussian Splatting scene
window.set_gaussian_splatting_scene(
    torch.tensor([0, 0, 0], dtype=torch.float32),
    means3d,
    shs,
    opacity,
    scales,
    rotations,
)

try:
    while window.is_running():
        shs[:, :, 0] = math.sin(time()) * 0.5 + 0.5
        sleep(0.001)  # Needed to avoid overloading the device :(
except KeyboardInterrupt:
    pass

window.destroy()
