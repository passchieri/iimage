import cProfile
import pstats
from iimage import AbstractBoard, RandomBoard, SingleBoard
import cv2
import numpy as np
from pathlib import Path

from iimage import DebugWindow


def points(size=300, count=1000):
    return np.random.randint(0, size, size=(count + 1, 2))


def redraw(size=300, points=None):
    tmp = np.zeros((size, size), dtype=np.int16)
    for i in range(len(points) - 1):
        cv2.line(tmp, points[i, :], points[i + 1, :], 1, 1)
        cv2.line(tmp, points[i, :], points[i + 1, :], 0, 1)
    return tmp


def reset(size=300, points=None):
    tmp = np.zeros((size, size), dtype=np.int16)
    for i in range(len(points) - 1):
        cv2.line(tmp, points[i, :], points[i + 1, :], 1, 1)
        tmp = np.zeros((size, size), dtype=np.int16)
    return tmp


def copy(size=300, points=None):
    tmp = np.zeros((size, size), dtype=np.int16)
    for i in range(len(points) - 1):
        t = tmp.copy()
        cv2.line(tmp, points[i, :], points[i + 1, :], 1, 1)
        tmp = t
    return tmp


def fill(size=300, points=None):
    tmp = np.zeros((size, size), dtype=np.int16)
    for i in range(len(points) - 1):
        cv2.line(tmp, points[i, :], points[i + 1, :], 1, 1)
        tmp.fill(0)
    return tmp


def f8(x):
    return "%12.8f" % x


def performance():
    profiler = cProfile.Profile()
    profiler.enable()
    pts = points(count=100000)
    result_redraw = redraw(points=pts)
    result_reset = reset(points=pts)
    result_copy = copy(points=pts)
    result_fill = fill(points=pts)
    profiler.disable()
    assert np.all(result_redraw == 0)
    assert np.all(result_reset == 0)
    assert np.all(result_copy == 0)
    assert np.all(result_fill == 0)
    stats = pstats.Stats(profiler).sort_stats("cumulative")
    stats.f8 = f8
    stats.print_stats("test_board")


def test_remmove_revert():
    for BoardClass in [SingleBoard, RandomBoard]:
        board = BoardClass(
            np.zeros((300, 300, 3), dtype=np.uint8),
            operating_size=300,
            pins=201,
            straws=4000,
            thickness=1,
        )
        straws = board.straws.copy()
        for i in range(board.straw_count):
            old_s = board.move_straw(i)
            assert not np.all(
                straws == board.straws
            ), f"Straws were not moved correctly for {BoardClass.__name__} {i=}"
            board.revert_straw(i, old_s)
            assert np.all(
                straws == board.straws
            ), f"Straws were not reverted correctly for {BoardClass.__name__} {i=}"


if __name__ == "__main__":
    # profiler = cProfile.Profile()
    # profiler.enable()
    performance()
    # profiler.disable()
    # stats = pstats.Stats(profiler).sort_stats("cumulative")
    # stats.print_stats(10)
