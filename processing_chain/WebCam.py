from PIL import Image
import os
import cv2
import torch
import torchvision as tv


class WebCam:

    test_pattern: torch.Tensor
    test_pattern_gray: torch.Tensor

    camera_index: int
    framesize: tuple[int, int]
    fps: float
    cap_frame_width: int
    cap_frame_height: int
    cap_fps: float
    webcam_is_ready: bool

    default_dtype = torch.float32

    def __init__(
        self,
        camera_index: int = 0,
        framesize: tuple[int, int] = (640, 480),
        fps: float = 20.0,
    ):
        super().__init__()
        assert fps > 0

        pic_path: str = "pics"

        self.camera_index = camera_index
        self.framesize = framesize
        self.cap = None
        self.fps = fps
        self.webcam_is_ready = False

        test_pattern_filename: str = os.path.join(pic_path, "test-pattern.png")
        self.test_pattern = tv.transforms.functional.pil_to_tensor(
            Image.open(test_pattern_filename)
        )

       # self.test_pattern_gray = tv.transforms.functional.rgb_to_grayscale(
       #     self.test_pattern[0:3]
       # ).squeeze(0)

    def open_cam(self) -> bool:
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            self.webcam_is_ready = False

        self.cap = cv2.VideoCapture(self.camera_index)

        assert self.cap is not None

        if self.cap.isOpened() is not True:
            self.webcam_is_ready = False
            return False

        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.framesize[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.framesize[1])

        self.cap_frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.cap_frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.cap_fps = float(self.cap.get(cv2.CAP_PROP_FPS))

        print(
            (
                f"Capturing: {self.cap_frame_width:.0f} x "
                f"{self.cap_frame_height:.0f} @ "
                f"{self.cap_fps:.1f}."
            )
        )
        self.webcam_is_ready = True
        return True

    def close_cam(self) -> None:
        if self.cap is not None:
            self.cap.release()

    def get_frame(self) -> torch.Tensor | None:
        if self.cap is None:
            return None
        else:
            success, frame = self.cap.read()

        if success is False:
            self.webcam_is_ready = False
            return None

        output = (
            torch.tensor(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            .movedim(-1, 0)
            .type(dtype=self.default_dtype)
            / 255.0
        )
        return output
