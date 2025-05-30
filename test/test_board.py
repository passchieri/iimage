import cProfile
import pstats
from iimage import Board
import cv2
import numpy as np
from pathlib import Path

from iimage import DebugWindow

FILE = "tessa"
PINS = 201
STRAWS = 4000

def main():
    final = DebugWindow("final", 1200, 1200, 2)
    debugger = DebugWindow("debugger", 400, 400, 3)
    img_path = Path(__file__).parent / "resources" / f"{FILE}.jpg"

    img = cv2.imread(img_path)
    assert img is not None, "file could not be read, check with os.path.exists()"
    h, w = img.shape[:2]
    scale = 1200 / max(h, w)
    if scale < 1:
        img = cv2.resize(
            img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA
        )
        h, w = img.shape[:2]
    board = Board(
        img,
        operating_size=300,
        strategy="single",
        pins=PINS,
        straws=STRAWS,
        thickness=1,
    )
    debugger.draw_image(board.image, 0)
    debugger.draw_image(board.target, 1)
    board.reset_counter()
    board.calculate_q()
    debugger.draw_image(board.result, 2)
    debugger.draw_image(board.diff, 0)
    temp = 0.02
    step_count = 1000
    print(
        "Press 't' to reset temperature, 's' to save, 'l' to load, 'r' to run, 'q' to quit"
    )
    tc = 0
    first = True
    key = ord("r")
    while first or (key := debugger.wait_key(100)) != 0:
        first = False
        if key == ord("t"):
            temp = 0.02
        elif key == ord("f"):
            print("Final...")
            make_final(final, img, board)
        elif key == ord("s"):
            print("Saving...")
            np.save(f"{FILE}_{board.pin_count}.npy", board.straws)
        elif key == ord("w"):
            print("Writing...")
            board.save(f"{FILE}.png")
            b = make_final(final, img, board)
            b.save(f"{FILE}_big.png")
        elif key == ord("l"):
            print("Reading...")
            board.straws = np.load(f"{FILE}_{board.pin_count}.npy")
            board.reset_counter()
            board.calculate_q()
            debugger.draw_image(board.result, 2)
            debugger.draw_image(board.diff, 0)
            temp = 1e-6
        elif key == ord("r"):
            c = step_count
            while c > step_count // 100:
                temp *= 0.9
                c = 0
                print("Running...")
                for _ in range(step_count):
                    if board.cool_step(temp):
                        c += 1
                        tc += 1
                debugger.draw_image(board.diff, 0)
                debugger.draw_image(board.result, 2)
                print(f"{c=}, {temp=:.4e}, {board.quality=:.4f}")
                if c > 0.9 * step_count:
                    temp = temp * 0.2
                # make_final(final, img, board)
            print(f"Done in {tc=} steps")
        elif key == ord("q"):
            print("Quitting...")
            debugger.terminate()
        else:
            print("Unknown key pressed")
            print(key)


def make_final(final, img, board):
    b = Board(
        img,
        operating_size=1200,
        strategy=board.strategy,
        pins=board.pin_count,
        straws=board.straw_count,
        thickness=board.thickness,
    )
    b.straws = board.straws.copy()
    b.reset_counter()
    b.calculate_q()
    final.draw_image(b.image, 0)
    final.draw_image(b.result, 1)
    return b


if __name__ == "__main__":
    profiler = cProfile.Profile()
    # profiler.enable()
    main()
    # profiler.disable()
    # stats = pstats.Stats(profiler).sort_stats("cumulative")
    # stats.print_stats(10)
