import numpy as np


def polar_to_cartesian(r, theta, center=(0, 0)):
    """
    Convert polar coordinates to Cartesian coordinates.
    :param r: The radius.
    :param theta: The angle in radians.
    :return: A tuple of (x, y) coordinates.
    """
    x = r * np.cos(theta) + center[0]
    y = r * np.sin(theta) + center[1]

    return int(x), int(y)


def line_circle_intersections(pt1, pt2, center, r) -> np.ndarray:
    """
    Find intersection points of the line through pt1 and pt2 with a circle.
    :param pt1: (x1, y1)
    :param pt2: (x2, y2)
    :param center: (cx, cy)
    :param r: radius
    :return: list of intersection points [(x, y), ...]
    """
    x1, y1 = pt1
    x2, y2 = pt2
    cx, cy = center

    dx = x2 - x1
    dy = y2 - y1

    a = dx**2 + dy**2
    b = 2 * (dx * (x1 - cx) + dy * (y1 - cy))
    c = (x1 - cx) ** 2 + (y1 - cy) ** 2 - r**2

    disc = b**2 - 4 * a * c
    if disc < 0:
        return []  # No intersection

    sqrt_disc = np.sqrt(disc)
    t1 = (-b + sqrt_disc) / (2 * a)
    t2 = (-b - sqrt_disc) / (2 * a)

    inter1 = np.array([x1 + t1 * dx, y1 + t1 * dy])
    inter2 = np.array([x1 + t2 * dx, y1 + t2 * dy])
    return np.array([inter1, inter2])
