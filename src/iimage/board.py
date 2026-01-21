from abc import ABC, abstractmethod
from typing import Optional
import numpy as np
import cv2

SIZE = 200


class AbstractBoard(ABC):
    def __init__(
        self,
        image: cv2.typing.MatLike,
        operating_size: int = 200,
        result_size: int = 1200,
        pins: int = 60,
        straws: int = 1000,
        thickness: int = 1,
    ):

        self.size = operating_size if operating_size else min(image.shape[:2])
        self.pin_count = pins
        self.straw_count = straws
        self.image = image.copy()
        self.thickness = thickness
        self.center = np.array([self.size // 2, self.size // 2])
        self.radius = self.size // 2
        self.ink = int(0.6 * self.size / self.straw_count * 255)
        self.result_size = result_size

        self.setup_pins()
        self.setup_straws()
        self.setup_mask()
        self.setup_target()
        self.counter = np.zeros((self.size, self.size), dtype=np.float64)
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
        self.target = 255 - self.target.astype(np.float64)

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

    @abstractmethod
    def setup_straws(self):
        pass

    def setup_mask(self):
        self.mask = np.zeros((self.size, self.size), dtype=np.uint8)
        cv2.circle(self.mask, self.center, round(self.radius), 255, -1)
        self.weight = np.zeros((self.size, self.size), dtype=np.float64)
        for i in range(self.size):
            for j in range(self.size):
                x = 1.0 * (i - self.center[0])
                y = 1.0 * (j - self.center[1])
                rr = (x**2 + y**2) / self.radius**2
                self.weight[i, j] = 1 - 0.7 * rr**0.25
        self.weight = np.clip(self.weight, 0, 1)
        self.weight[self.mask == 0] = 0
        self.weight = self.weight / np.sum(self.weight)

    def reset_counter(self, mask: bool = True):
        result = np.zeros((self.size, self.size), dtype=np.int16)
        # self.result.fill(0)
        self.counter.fill(0)
        for straw in self.straws:
            pt1 = tuple(self.pins[straw[0]])
            pt2 = tuple(self.pins[straw[1]])
            cv2.line(self.tmp, pt1, pt2, self.ink, self.thickness)
            self.counter = self.counter + self.tmp
            cv2.line(self.tmp, pt1, pt2, 0, self.thickness)

    def calculate_q(self):

        # self.diff = np.subtract(self.target, self.result)
        self.diff = np.subtract(self.target, self.counter)
        self.diff = np.multiply(self.diff, self.diff) / 255.0  # w x h numbers of 0-255
        self.diff = np.multiply(
            self.diff, self.weight
        )  # sum of weight =1, so if weights equal 0 - 255/wxh

        self.quality = np.sum(self.diff)  # 0-255
        self.diff = np.clip(self.diff * self.diff.size, 0, 255)
        # self.diff = (255 - self.diff / np.max(self.weight) * 255).astype(np.uint8)

    def cool_step(self, temp=0.1):
        old_q = self.quality
        i = np.random.randint(0, self.straw_count)
        old_s, new_s = self.move_straw(i)
        self.update_counter(old_s, new_s)
        self.calculate_q()
        assert np.sum(self.tmp) == 0, f"tmp not empty {np.sum(self.tmp)}"

        if old_q < self.quality:
            diff = old_q - self.quality
            e = np.exp(-(diff**2) / temp**2)
            prob = np.random.random()
            if prob > e:  # rejecting worse quality at this temperature
                self.quality = old_q
                self.update_counter(new_s, old_s)
                self.revert_straw(i, old_s)
                return False
            return True
        else:
            return True

    def update_counter(self, remove, add):
        for straw in remove:
            pt1 = tuple(self.pins[straw[0]])
            pt2 = tuple(self.pins[straw[1]])
            cv2.line(self.tmp, pt1, pt2, self.ink, self.thickness)
            self.counter = self.counter - self.tmp
            cv2.line(self.tmp, pt1, pt2, 0, self.thickness)
        for straw in add:
            pt1 = tuple(self.pins[straw[0]])
            pt2 = tuple(self.pins[straw[1]])
            cv2.line(self.tmp, pt1, pt2, self.ink, self.thickness)
            self.counter = self.counter + self.tmp
            cv2.line(self.tmp, pt1, pt2, 0, self.thickness)

    @abstractmethod
    def move_straw(self, i):
        pass

    @abstractmethod
    def revert_straw(self, i, old_straws, new_straws):
        pass

    def total_straw_length(self):
        """
        Calculate the total length of all straws combined.
        :return: Total length (float)
        """
        # Get coordinates for all straw endpoints
        pts1 = self.pins[self.straws[:, 0]]
        pts2 = self.pins[self.straws[:, 1]]
        # Compute Euclidean distances for each straw
        lengths = np.linalg.norm(pts1 - pts2, axis=1)
        return np.sum(lengths)

    def save(self, filename: str):
        result = self.draw_result(mask=False)
        cv2.imwrite(filename, result)

    def draw_result(self):
        thetas = np.linspace(0, 2 * np.pi, self.pin_count, endpoint=False)
        pins = np.stack(
            (
                self.result_size // 2 * (1 + np.sin(thetas)),
                self.result_size // 2 * (1 + np.cos(thetas)),
            ),
            axis=1,
        ).astype(int)
        pins = np.clip(pins, 0, self.result_size)

        result = np.ones((self.result_size, self.result_size, 3), dtype=np.uint8) * 255

        for straw in self.straws:
            pt1 = tuple(pins[straw[0]])
            pt2 = tuple(pins[straw[1]])
            cv2.line(result, pt1, pt2, (0, 0, 0), 1, lineType=cv2.LINE_AA)

        return result


class RandomBoard(AbstractBoard):
    def setup_straws(self):
        p1 = np.random.randint(0, self.pin_count, self.straw_count)
        p2 = np.random.randint(0, self.pin_count, self.straw_count)
        self.straws = np.stack((p1, p2), axis=1)

    def move_straw(self, i):
        p1 = (self.straws[i, 0] + np.random.randint(1, self.pin_count)) % self.pin_count
        p2 = (self.straws[i, 1] + np.random.randint(1, self.pin_count)) % self.pin_count
        old_s = self.straws[i : i + 1].copy()
        self.straws[i] = [p1, p2]
        new_s = self.straws[i].copy()
        return old_s, new_s

    def revert_straw(self, i, old_straws):
        self.straws[i] = old_straws


class SingleBoard(AbstractBoard):
    def setup_straws(self):
        p1 = np.random.randint(0, self.pin_count, self.straw_count)
        p2 = np.zeros(self.straw_count, dtype=np.int32)
        for i in range(self.straw_count - 1):
            p2[i] = p1[i + 1]
        p2[-1] = p1[0]
        self.straws = np.stack((p1, p2), axis=1)

    def move_straw(self, i):
        p1 = (self.straws[i, 1] + np.random.randint(1, self.pin_count)) % self.pin_count
        if i == self.straw_count - 1:
            old_s = np.empty((2, self.straws.shape[1]), dtype=self.straws.dtype)
            old_s[0] = self.straws[i]
            old_s[1] = self.straws[(i + 1) % self.straw_count]
        else:
            old_s = self.straws[i : i + 2].copy()
        self.straws[i, 1] = p1
        self.straws[(i + 1) % self.straw_count, 0] = p1
        if i == self.straw_count - 1:
            new_s = np.empty((2, self.straws.shape[1]), dtype=self.straws.dtype)
            new_s[0] = self.straws[i]
            new_s[1] = self.straws[(i + 1) % self.straw_count]
        else:
            new_s = self.straws[i : i + 2].copy()

        return old_s, new_s

    def revert_straw(self, i, old_straws):
        self.straws[i] = old_straws[0]
        self.straws[(i + 1) % self.straw_count] = old_straws[1]
