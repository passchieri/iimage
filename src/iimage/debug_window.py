import cv2
import numpy as np


class DebugWindow:
    def __init__(
        self,
        name: str = "debugger",
        width: int = 600,
        height: int = 600,
        count: int = 3,
    ):
        self.name = name
        self.width = width
        self.height = height
        self.count = count
        self.running = True
        self.canvas = (
            np.ones((height + 2, (width + 1) * count + 1, 3), dtype=np.uint8) * 255
        )
        # draw borders
        self.canvas[0, :, :] = 0
        self.canvas[height + 1, :, :] = 0
        for i in range(count + 1):
            self.canvas[:, i * (width + 1), :] = 0

        cv2.namedWindow(self.name, cv2.WINDOW_OPENGL)
        cv2.resizeWindow(
            self.name, height=self.canvas.shape[0], width=self.canvas.shape[1]
        )
        cv2.imshow(self.name, self.canvas)
        cv2.waitKey(1) & 0xFF

    def draw_image(self, image: np.ndarray, index: int = 0, invert=True):
        if index >= self.count:
            raise ValueError(
                f"Index {index} out of range. Max index is {self.count - 1}."
            )
        h, w = image.shape[:2]
        if h > self.height or w > self.width:
            scale = min(self.width / w, self.height / h)
            image = cv2.resize(
                image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA
            )
            h, w = image.shape[:2]
        elif h < self.height and w < self.width:
            scale = min(self.width / w, self.height / h)
            image = cv2.resize(
                image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA
            )
            h, w = image.shape[:2]
        if len(image.shape) == 2:  # grayscale
            if image.dtype != np.uint8:
                image = cv2.normalize(
                    image, None, 0, 512, cv2.NORM_MINMAX, dtype=cv2.CV_8U
                )
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if invert:
            image = cv2.bitwise_not(image)
        self.canvas[
            1 : h + 1, index * (self.width + 1) + 1 : index * (self.width + 1) + 1 + w
        ] = image
        cv2.imshow(self.name, self.canvas)
        cv2.waitKey(1) & 0xFF

    def terminate(self):
        cv2.destroyAllWindows()
        self.running = False

    def wait_key(self, delay: int = 100):
        if not self.running:
            return 0
        k = 255
        while k == 255:
            k = cv2.waitKey(delay) & 0xFF
        return k
