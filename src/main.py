import argparse
import cProfile
import pstats
from iimage import Board
import cv2
import numpy as np
from pathlib import Path

from iimage import DebugWindow

FILE = "tessa"


def cli_main():
    parser = argparse.ArgumentParser(description="Run string art board generator.")
    parser.add_argument("img_path", type=str, help="Path to the input image")
    parser.add_argument(
        "--pins", type=int, default=201, help="Number of pins (default: 201)"
    )
    parser.add_argument(
        "--straws", type=int, default=4000, help="Number of straws (default: 4000)"
    )
    parser.add_argument(
        "--strategy",
        type=str,
        choices=["single", "random"],
        default="single",
        help="Strategy to use: 'single' or 'random' (default: single)",
    )
    args = parser.parse_args()
    main(args.img_path, args.pins, args.straws, args.strategy)


def main(img_path, pins=201, straws=4000, strategy="single"):
    FILE = Path(img_path).stem
    final = None
    debugger = DebugWindow(f"debug {strategy} {pins} {straws}", 400, 400, 3)

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
        strategy=strategy,
        pins=pins,
        straws=straws,
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
            b, final = make_final(final, img, board)
        elif key == ord("s"):
            print("Saving...")
            np.save(f"{FILE}_{board.pin_count}.npy", board.straws)
        elif key == ord("w"):
            print("Writing...")
            board.save(f"{FILE}.png")
            b, final = make_final(final, img, board)
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
                print(
                    f"{c=}, {temp=:.4e}, {board.quality=:.4f}, length={board.total_straw_length():.0f}"
                )
                if c > 0.9 * step_count:
                    temp = temp * 0.2
                # make_final(final, img, board)
            print(f"Done in {tc=} steps, length={board.total_straw_length()}")
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
    if not final:
        final = DebugWindow(
            f"Final {b.strategy} {b.pin_count} {b.straw_count}", 1200, 1200, 2
        )
    final.draw_image(b.image, 0)
    final.draw_image(b.result, 1)
    return b, final


if __name__ == "__main__":
    # profiler = cProfile.Profile()
    # profiler.enable()
    cli_main()
    # profiler.disable()
    # stats = pstats.Stats(profiler).sort_stats("cumulative")
    # stats.print_stats(10)
