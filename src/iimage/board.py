from typing import Optional
import numpy as np
import cv2

SIZE = 200


class Board:
    def __init__(
        self,
        image: cv2.typing.MatLike,
        operating_size: int = 200,
        pins: int = 60,
        straws: int = 1000,
    ):

        self.size = operating_size if operating_size else min(image.shape[:2])
        self.pin_count = pins
        self.straw_count = straws
        self.image = image.copy()
        self.center = np.array([self.size // 2, self.size // 2])
        self.radius = self.size // 2
        self.setup_pins()
        self.setup_straws()
        self.setup_mask()
        self.setup_target()
        self.result = np.ones((self.size, self.size), dtype=self.image.dtype) * 255
        self.create_diff()

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
        p1 = p2 = np.random.randint(0, self.pin_count, self.straw_count)
        p2 = np.random.randint(0, self.pin_count, self.straw_count)
        for i in range(self.straw_count):
            while p2[i] == p1[i]:
                p2[i] = np.random.randint(0, self.pin_count)
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

    def draw_result(self, mask: bool = True):
        result = np.zeros((self.size, self.size), dtype=np.int16)
        self.result.fill(0)
        tmp = np.zeros((self.size, self.size), dtype=np.int16)
        ink = int(0.6 * self.size / self.straw_count * 255)
        # ink = 255
        for straw in self.straws:
            pt1 = tuple(self.pins[straw[0]])
            pt2 = tuple(self.pins[straw[1]])
            tmp.fill(0)
            cv2.line(tmp, pt1, pt2, ink, 1)
            result = result + tmp
        # self.result[self.result > 255] = 255
        # masked = self.result.copy()
        result[self.mask == 0] = 0
        # max_intensity = np.max(masked)
        result = 255 - result  # invert, and scale
        self.result = np.clip(result, 0, 255).astype(np.uint8)
        # self.result[self.result > 255] = (
        #     255  # ensure max is 255, as we only looked at the max in the masked area
        # )
        if mask:
            self.result[self.mask == 0] = 128

    def create_diff(self):

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
        i = np.random.randint(0, self.straw_count)

        p1 = p2 = np.random.randint(0, self.pin_count)
        p2 = np.random.randint(0, self.pin_count)
        while p2 == p1:
            p2 = np.random.randint(0, self.pin_count)
        self.straws[i] = [p1, p2]
        self.draw_result()
        self.create_diff()

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
                return False
            # print("temp accepted")
            return True
        else:
            # print("accepted")
            return True
        # step_size = 4
        # new_straws = self.straws.copy()
        # straw = np.random.randint(0, len(new_straws))
        # i = np.random.randint(-step_size, step_size + 1)
        # j = np.random.randint(-step_size, step_size + 1)
        # s = (new_straws[straw][0] + i) % self.pin_count
        # e = (new_straws[straw][1] + j) % self.pin_count
        # while s == e:
        #     j = np.random.randint(-step_size, step_size + 1)
        #     e = (new_straws[straw][1] + j) % self.pin_count
        # new_straws[straw] = [s, e]
        # print(f"{self.straws[straw]}->{new_straws[straw]}, ")
        # old_match, oimg = self.match_image(self.straws)
        # new_match, nimg = self.match_image(new_straws)

        # print(old_match, new_match)
        # accepted = False
        # if new_match < old_match:
        #     self.straws = new_straws
        #     print("accepted")
        #     accepted = True
        # else:
        #     print("old match",old_match)
        #     print("new match",new_match)
        #     #accept with probability
        #     prob=np.exp((old_match-new_match)/0.0001)
        #     if np.random.random()<prob:
        #         self.straws=new_straws
        #         print("accepted")
        #         accepted=True
        #     else:
        #         print("rejected")
        # if accepted:
        #     self.straws = new_straws
        #     return nimg
        # return oimg

    def draw(self, img):
        """
        Draw the board on the image.
        :param img: The image to draw on.
        """
        img.fill(255)
        for pos in self.pins:
            cv2.circle(img, tuple(pos), 2, (0, 0, 0), -1)
        for straw in self.straws:
            pt1 = tuple(self.pins[straw[0]])
            pt2 = tuple(self.pins[straw[1]])
            overlay = img.copy()
            cv2.line(overlay, pt1, pt2, (0, 0, 0), 1)
            weight = 0.5
            cv2.addWeighted(overlay, weight, img, 1 - weight, 0, img)
        # cv2.circle(iomg, tuple(self.center), self.radius, (0, 0, 0), 1)
        # cv2.line(img, tuple(self.center), tuple(self.pin_positions[0]), (0, 0, 255), 1)
        # cv2.line(img, tuple(self.center), tuple(self.pin_positions[1]), (255, 0, 0), 1)
        self.match_image()
