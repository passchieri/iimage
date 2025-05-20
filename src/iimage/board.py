from typing import Optional
import numpy as np
import cv2

SIZE = 200


class Board:
    STRATEGIES = {
        "random": "random",
        "single": "single",
    }

    def __init__(
        self,
        image: cv2.typing.MatLike,
        operating_size: int = 200,
        pins: int = 60,
        straws: int = 1000,
        thickness: int = 1,
        strategy: str = "random",
    ):

        self.size = operating_size if operating_size else min(image.shape[:2])
        self.pin_count = pins
        self.straw_count = straws
        self.image = image.copy()
        self.thickness = thickness
        self.center = np.array([self.size // 2, self.size // 2])
        self.radius = self.size // 2
        assert strategy in self.STRATEGIES, f"Invalid strategy: {strategy}"
        self.strategy = strategy
        self.ink = int(0.6 * self.size / self.straw_count * 255)
        self.randomizer = self.pin_count // 2

        self.setup_pins()
        self.setup_straws()
        self.setup_mask()
        self.setup_target()
        self.counter = np.zeros((self.size, self.size), dtype=np.uint16)
        self.result = np.ones((self.size, self.size), dtype=self.image.dtype) * 255
        self.tmp = np.zeros((self.size, self.size), dtype=np.int16)
        self.calculate_q()

    def setup_target(self):
        h, w = self.image.shape[:2]
        scale = self.size / max(h, w)
        self.target = cv2.resize(
            self.image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA
        )
        if len(self.target.shape) == 2:  # grayscale
            pass
        elif len(self.target.shape) == 3:  # BGR
            self.target = cv2.cvtColor(self.target, cv2.COLOR_BGR2GRAY)
        elif len(self.target.shape) == 4:  # BGRA
            self.target = cv2.cvtColor(self.target, cv2.COLOR_BGRA2GRAY)
        self.target = cv2.equalizeHist(self.target)
        self.target[self.mask == 0] = 128

    def setup_pins(self):
        thetas = np.linspace(0, 2 * np.pi, self.pin_count, endpoint=False)
        self.pins = np.stack(
            (
                self.radius * np.sin(thetas) + self.center[0],
                self.radius * np.cos(thetas) + self.center[1],
            ),
            axis=1,
        )
        self.pins = np.round(self.pins).astype(int)
        self.pins = np.clip(self.pins, 0, self.size)

    def setup_straws(self):
        if self.strategy == self.STRATEGIES["random"]:
            p1 = np.random.randint(0, self.pin_count, self.straw_count)
            p2 = np.random.randint(0, self.pin_count, self.straw_count)
        elif self.strategy == self.STRATEGIES["single"]:
            p1 = np.random.randint(0, self.pin_count, self.straw_count)
            p2 = np.zeros(self.straw_count, dtype=np.int32)
            for i in range(self.straw_count - 1):
                p2[i] = p1[i + 1]
            p2[-1] = p1[0]
        else:
            raise ValueError(f"Invalid strategy: {self.strategy}")
        self.straws = np.stack((p1, p2), axis=1)

    def setup_mask(self):
        self.mask = np.zeros((self.size, self.size), dtype=np.uint8)
        cv2.circle(self.mask, self.center, round(self.radius), 255, -1)
        self.weight = np.zeros((self.size, self.size), dtype=np.float64)
        for i in range(self.size):
            for j in range(self.size):
                x = 1.0 * (i - self.center[0])
                y = 1.0 * (j - self.center[1])
                rr = (x**2 + y**2) / self.radius**2
                self.weight[i, j] = 1 - 0.5 * rr**0.5
        self.weight = np.clip(self.weight, 0, 1)
        # self.weight[:, :] = 1
        self.weight[self.mask == 0] = 0
        self.weight = self.weight / np.sum(self.weight)

    def update_result(self, mask: bool = True):
        self.result = np.clip(255 - self.counter, 0, 255).astype(np.uint8)
        if mask:
            self.result[self.mask == 0] = 128

    def reset_counter(self, mask: bool = True):
        result = np.zeros((self.size, self.size), dtype=np.int16)
        self.result.fill(0)
        self.counter.fill(0)
        for straw in self.straws:
            pt1 = tuple(self.pins[straw[0]])
            pt2 = tuple(self.pins[straw[1]])
            cv2.line(self.tmp, pt1, pt2, self.ink, self.thickness)
            self.counter = self.counter + self.tmp
            cv2.line(self.tmp, pt1, pt2, 0, self.thickness)
        hist = np.bincount(self.counter.flatten(), minlength=256)
        print(hist)
        self.update_result(mask)

    def calculate_q(self):

        self.diff = np.subtract(
            self.target.astype(np.float64), self.result.astype(np.float64)
        )
        self.diff = np.multiply(self.diff, self.diff) / 255.0  # w x h numbers of 0-255
        self.diff = np.multiply(
            self.diff, self.weight
        )  # sum of weight =1, so if weights equal 0 - 255/wxh

        self.quality = np.sum(self.diff)  # 0-255
        self.diff = np.clip(self.diff * self.diff.size, 0, 255)
        self.diff = self.diff.astype(np.uint8)
        # self.diff = (255 - self.diff / np.max(self.weight) * 255).astype(np.uint8)

    def cool_step(self, temp=0.1):
        old_q = self.quality
        old_straws = self.straws.copy()
        old_result = self.result.copy()
        old_diff = self.diff.copy()
        old_count = self.counter.copy()
        i = np.random.randint(0, self.straw_count)
        if self.strategy == self.STRATEGIES["random"]:
            p1 = (
                self.straws[i, 0] + np.random.randint(-self.randomizer, self.randomizer)
            ) % self.pin_count
            p2 = (
                self.straws[i, 1] + np.random.randint(-self.randomizer, self.randomizer)
            ) % self.pin_count
            # p1 = np.random.randint(0, self.pin_count)
            # p2 = np.random.randint(0, self.pin_count)
            old_s = self.straws[i : i + 1].copy()
            self.straws[i] = [p1, p2]
            new_s = self.straws[i : i + 1].copy()
        elif self.strategy == self.STRATEGIES["single"]:
            # p1 = np.random.randint(0, self.pin_count)
            p1 = (
                self.straws[i, 1] + np.random.randint(-self.randomizer, self.randomizer)
            ) % self.pin_count
            old_s = self.straws[i : i + 2].copy()
            self.straws[i, 1] = p1
            self.straws[(i + 1) % self.straw_count, 0] = p1
            new_s = self.straws[i : i + 2].copy()

        for straw in old_s:
            pt1 = tuple(self.pins[straw[0]])
            pt2 = tuple(self.pins[straw[1]])
            cv2.line(self.tmp, pt1, pt2, self.ink, self.thickness)
            self.counter = self.counter - self.tmp
            cv2.line(self.tmp, pt1, pt2, 0, self.thickness)
        for straw in new_s:
            pt1 = tuple(self.pins[straw[0]])
            pt2 = tuple(self.pins[straw[1]])
            cv2.line(self.tmp, pt1, pt2, self.ink, self.thickness)
            self.counter = self.counter + self.tmp
            cv2.line(self.tmp, pt1, pt2, 0, self.thickness)
        # print(f"{np.sum(self.counter)}")
        # self.draw_result(True)
        # print(f"{np.sum(self.counter)}")
        self.update_result(True)
        self.calculate_q()
        assert np.sum(self.tmp) == 0, f"tmp not empty {np.sum(self.tmp)}"

        if old_q < self.quality:
            diff = old_q - self.quality
            e = np.exp(-(diff**2) / temp**2)
            prob = np.random.random()
            if prob > e:
                # print("not accepted")
                self.straws = old_straws
                self.result = old_result
                self.diff = old_diff
                self.quality = old_q
                self.counter = old_count
                return False
            # print("temp accepted")
            return True
        else:
            # print("accepted")
            return True
