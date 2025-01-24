import torch
from ._async_torchwindow import Window as _Window


class Window:
    def __init__(self, width: int, height: int):
        """Initialize the window.
        Calling just the constructor won't open the window. A call to `start()` is needed."""
        
        self._window = _Window(width, height)

        # Image
        self._image = None

        # GS scene
        self._gs_background = None
        self._gs_means3d = None
        self._gs_shs = None
        self._gs_opacity = None
        self._gs_scales = None
        self._gs_rotations = None

    def size(self) -> tuple[int, int]:
        """Get the current size of the window."""
        
        return self._window.get_size()

    def fps(self) -> float:
        """Get the current FPS of the window."""
        
        return self._window.get_fps()

    def set_image(self, image: torch.Tensor) -> None:
        """Set the current visualization to an image.
        
        :param image:
            A (H, W, 4) or (1, H, W, 4) tensor for the image.
        """
        
        # Validation
        assert (image.ndim == 3 or image.ndim == 4) and image.shape[
            -1
        ] == 4, "image ndim must be (H, W, 4) or (1, H, W, 4)"

        H = image.shape[0] if image.ndim == 3 else image.shape[1]
        W = image.shape[1] if image.ndim == 3 else image.shape[2]

        # Save reference to avoid GC while being visualized
        self._image = image.contiguous().cuda()

        self._window.set_image(W, H, self._image.data_ptr())

    def set_gaussian_splatting_scene(
        self,
        background: torch.Tensor,
        means3d: torch.Tensor,
        shs: torch.Tensor,
        opacity: torch.Tensor,
        scales: torch.Tensor,
        rotations: torch.Tensor,
    ) -> None:
        """Set the current visualization to a Gaussian Splatting scene.
        
        :param background:
            A (3,) tensor for the background color.
        :param means3d:
            A (P, 3) tensor for the mean of the 3D gaussians.
            `P` is t he number of points.
        :param shs:
            A (P, M, 3) tensor for the SH coefficients of the 3D gaussians.
            `M` is the number of RGB coefficients, usually 16.
        :param opacity:
            A (P,) tensor for the *activated* opacities of the gaussians.
        :param scales:
            A (P, 3) tensor for the *activated* scales of the gaussians.
        :param rotations:
            A (P, 4) tensor for the *normalized* rotations of the gaussians (quaternions).
        """
        
        # Validation
        assert background.shape == (3,), "background shape must be (3,)"
        assert (
            means3d.ndim == 2 and means3d.shape[-1] == 3
        ), "means3d ndim must be (P, 3)"
        P = means3d.shape[0]
        assert (
            shs.ndim == 3 and shs.shape[0] == P and shs.shape[2] == 3
        ), "shs ndim must be (P, M, 3)"
        M = shs.shape[1]
        assert opacity.shape == (P,), "opacity shape must be (P,)"
        assert scales.shape == (P, 3), "scales shape must be (P, 3)"
        assert rotations.shape == (P, 4), "rotations shape must be (P, 4)"

        # Save references to avoid GC them while being visualized
        self._gs_background = background.contiguous().cuda()
        self._gs_means3d = means3d.contiguous().cuda()
        self._gs_shs = shs.contiguous().cuda()
        self._gs_opacity = opacity.contiguous().cuda()
        self._gs_scales = scales.contiguous().cuda()
        self._gs_rotations = rotations.contiguous().cuda()

        self._window.set_gaussian_splatting_scene(
            P,
            self._gs_background.data_ptr(),
            self._gs_means3d.data_ptr(),
            self._gs_shs.data_ptr(),
            3,
            M,
            self._gs_opacity.data_ptr(),
            self._gs_scales.data_ptr(),
            self._gs_rotations.data_ptr(),
        )

    def start(self):
        """Start the window asynchronously w.r.t. the caller thread."""
        self._window.start()

    def is_running(self) -> bool:
        """Check if the window is running."""
        return self._window.is_running()

    def destroy(self):
        """Destroy the window.
        This method has to be called manually once the Python program exits!"""
        self._window.destroy()
