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

