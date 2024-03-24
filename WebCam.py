#%%
#
# WebCam.py
# ========================================================
# interface to cv2 for using a webcam or for reading from
# a video file
#
# Version 1.0, before 30.03.2023:
#   written by David...
#
# Version 1.1, 30.03.2023:
#   thrown out test image
#   added test code
#   added code to "capture" from video file
#


from PIL import Image
import os
import cv2
import torch
import torchvision as tv


class WebCam:

    # test_pattern: torch.Tensor
    # test_pattern_gray: torch.Tensor

    source: int
    framesize: tuple[int, int]
    fps: float
    cap_frame_width: int
    cap_frame_height: int
    cap_fps: float
    webcam_is_ready: bool
    cap_frames_available: int

    default_dtype = torch.float32

    def __init__(
        self,
        source: str | int = 1,
        framesize: tuple[int, int] = (640, 480),
        fps: float = 20.0,
    ):
        super().__init__()
        assert fps > 0

        # pic_path: str = "pics"

        self.source = source
        self.framesize = framesize
        self.cap = None
        self.fps = fps
        self.webcam_is_ready = False

    def open_cam(self) -> bool:
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            self.webcam_is_ready = False

        self.cap = cv2.VideoCapture(self.source)

        assert self.cap is not None

        if self.cap.isOpened() is not True:
            self.webcam_is_ready = False
            return False

        if type(self.source) != str:
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.framesize[0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.framesize[1])
            self.cap_frames_available = None
        else:
            self.cap_frames_available = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)

        self.cap_frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.cap_frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.cap_fps = float(self.cap.get(cv2.CAP_PROP_FPS))

        print(
            (
                f"Capturing or reading with: {self.cap_frame_width:.0f} x "
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
            torch.tensor(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            .movedim(-1, 0)
            .type(dtype=self.default_dtype)
            / 255.0
        )
        return output


# for testing the code if module is executed from command line
if __name__ == "__main__":

    import matplotlib.pyplot as plt

    print("Testing the WebCam interface")
    camera_index = 0
    file_name = "pics/level1.mp4"
    n_capture = 3
    # open
    print("Opening camera")
    w = WebCam(file_name)
    if not w.open_cam():
        raise OSError(f"Opening file with name {file_name} failed!")

    # print information
    print(f"Frame size {w.cap_frame_width} x {w.cap_frame_height} at {w.cap_fps} fps.")

    # capture three frames and show them
    for i in range(min([n_capture, w.cap_frames_available])):  # TODO: available?
        frame = w.get_frame()
        if frame == None:
            raise OSError(f"Can not get frame from file with name {file_name}!")
        print(f"frame {i} has shape {frame.shape}")

        frame_numpy = (frame.movedim(0, -1) * 255).type(dtype=torch.uint8).numpy()
        plt.imshow(frame_numpy)
        plt.show()

    # close
    print("Closing file")
    w.close_cam()

    # open
    print("Opening camera")
    w = WebCam(camera_index)
    if not w.open_cam():
        raise OSError(f"Opening web cam with index {camera_index} failed!")

    # print information
    print(f"Frame size {w.cap_frame_width} x {w.cap_frame_height} at {w.cap_fps} fps.")

    # capture three frames and show them
    for i in range(n_capture):
        frame = w.get_frame()
        if frame == None:
            raise OSError(f"Can not get frame from camera with index {camera_index}!")
        print(f"frame {i} has shape {frame.shape}")

        frame_numpy = (frame.movedim(0, -1) * 255).type(dtype=torch.uint8).numpy()
        plt.imshow(frame_numpy)
        plt.show()

    # close
    print("Closing camera")
    w.close_cam()

# %%
