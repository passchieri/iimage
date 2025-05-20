from typing import Optional
import numpy as np
import cv2 as cv

from .utils import line_circle_intersections

from .board import Board


def sample_pixel_by_intensity(img):
    """
    Sample a random pixel from the Greyscale image, with probability proportional to its intensity.
    :param rgb_img: The original greyscale image (numpy array, shape HxW).
    :return: (row, col) tuple of the selected pixel.
    """
    # Compute intensity (convert to grayscale)
    flat_intensity = 255 - img.flatten().astype(np.float64)
    prob = flat_intensity / flat_intensity.sum()
    idx = np.random.choice(len(flat_intensity), p=prob)
    row, col = np.unravel_index(idx, img.shape)
    return row, col


def add_line(img, pt1, pt2, color=(0, 255, 0), thickness=1):
    """
    Draw a line on the image.
    :param img: The image to draw on.
    :param pt1: The starting point of the line.
    :param pt2: The ending point of the line.
    :param color: The color of the line (BGR format).
    :param thickness: The thickness of the line.
    """
    cv.line(img, pt1, pt2, color, thickness)


def add_random_lines(img, num_lines=10):
    """
    Draw random lines on the image.
    :param img: The image to draw on.
    :param num_lines: The number of lines to draw.
    :param color: The color of the lines (BGR format).
    :param thickness: The thickness of the lines.
    """
    h, w = img.shape[:2]
    size = min(h, w)
    for _ in range(num_lines):
        r = np.random.random() * size * 0.5
        r = size * 0.5
        center = (w / 2, h / 2)
        theta = np.random.random() * 2 * np.pi
        pt1 = polar_to_cartesian(r, np.random.random() * 2 * np.pi, center=center)
        pt2 = polar_to_cartesian(r, np.random.random() * 2 * np.pi, center=center)
        # pt1 = (np.random.randint(0, w), np.random.randint(0, h))
        # pt2 = (np.random.randint(0, w), np.random.randint(0, h))
        color = (
            np.random.randint(0, 256),
            np.random.randint(0, 256),
            np.random.randint(0, 256),
        )
        thickness = np.random.randint(2, 4)
        overlay = img.copy()
        cv.line(overlay, pt1, pt2, color, thickness)
        alpha = np.random.random()
        cv.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)


def add_intensity_line(sample, draw_img):
    """
    Draw a line on the image with intensity proportional to the pixel intensity.
    :param img: The image to draw on.
    """
    h, w = sample.shape[:2]
    pt1 = sample_pixel_by_intensity(sample)
    pt2 = sample_pixel_by_intensity(sample)

    pts = line_circle_intersections(pt1, pt2, (w / 2, h / 2), min(h, w) * 0.5)
    if len(pts) == 2:
        pt1 = pts[0].astype(int)
        pt2 = pts[1].astype(int)
    else:
        print("no intersection")
        return
    # intensity = int(img[pt1[0], pt1[1]])
    # color = (intensity, intensity, intensity)
    thickness = 1
    cv.line(draw_img, pt1, pt2, [0, 0, 0], thickness)


def main():
    cv.namedWindow("image", cv.WINDOW_OPENGL)
    # img=np.zeros((400, 600, 3), dtype=np.uint8)
    img = cv.imread("resources/igor passchier.jpg")
    assert img is not None, "file could not be read, check with os.path.exists()"
    h, w = img.shape[:2]
    scale = 200 / max(h, w)
    if scale < 1:
        img = cv.resize(
            img, (int(w * scale), int(h * scale)), interpolation=cv.INTER_AREA
        )
        h, w = img.shape[:2]

    cv.resizeWindow("image", img.shape[0], img.shape[1] * 3)
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img = cv.equalizeHist(img)
    doubled = np.ones((h, w * 3), dtype=img.dtype) * 255
    draw = np.ones((h, w), dtype=img.dtype) * 255

    board = Board(img, 60)
    img = board.image
    debugger = board.mask
    doubled[:h, :w] = img
    doubled[:h, w * 2 :] = debugger
    cooling = False
    while True:
        if cooling:
            debugger = board.cool_step()
            board.draw(draw)
        doubled[:h, w : w * 2] = draw
        doubled[:h, w * 2 :] = debugger
        cv.imshow("image", doubled)
        k = cv.waitKey(100) & 0xFF
        if k == 27:  # ESC key
            break
        elif k == ord("x"):
            break
        elif k == ord("l"):
            add_random_lines(draw, num_lines=1)
        elif k == ord("c"):
            draw.fill(255)
        elif k == ord("b"):
            board.draw(draw)
        elif k == ord("s"):
            cooling = not cooling
        elif k == ord("i"):
            for _ in range(1000):
                add_intensity_line(img, draw)
                doubled[:h, w:] = draw
                cv.imshow("image", doubled)
        #     img = np.zeros(img.shape, dtype=np.uint8)
        # elif k == ord('w'):
        #     img = np.ones(img.shape, dtype=np.uint8) * 255
        # elif k == ord('r'):
        #     img = np.random.randint(0,256,img.shape, dtype=np.uint8)
    cv.destroyAllWindows()
    # assert img is not None, "file could not be read, check with os.path.exists()"


if __name__ == "__main__":
    main()
