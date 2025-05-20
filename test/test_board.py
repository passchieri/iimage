from iimage import Board
import cv2
import numpy as np
from pathlib import Path

from iimage import DebugWindow


def main():
    debugger = DebugWindow("debugger", 400, 300, 3)
    img_path = Path(__file__).parent / "resources" / "test.jpg"

    img = cv2.imread(img_path)
    assert img is not None, "file could not be read, check with os.path.exists()"
    h, w = img.shape[:2]
    scale = 600 / max(h, w)
    if scale < 1:
        img = cv2.resize(
            img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA
        )
        h, w = img.shape[:2]
    # img[:, :] = 255
    # img[:, w // 2 :] = 0
    board = Board(img, operating_size=200, pins=180, straws=500)
    debugger.draw_image(board.image, 0)
    debugger.draw_image(board.target, 1)
    board.draw_result()
    board.create_diff()
    debugger.draw_image(board.result, 2)
    debugger.draw_image(board.diff, 0)
    # tmp = (board.weight.copy() * 255 / np.max(board.weight)).astype(np.uint8)
    # debugger.draw_image(tmp, 0)
    # debugger.wait_key(100)
    temp = 0.2
    step_count = 500
    print(
        "Press 't' to reset temperature, 's' to save, 'l' to load, 'r' to run, 'q' to quit"
    )
    while (key := debugger.wait_key(100)) != 0:
        if key == ord("t"):
            temp = 0.2
        elif key == ord("s"):
            print("Saving...")
            np.save("result.npy", board.straws)
        elif key == ord("l"):
            print("Reading...")
            board.straws = np.load("result.npy")
            board.draw_result()
            board.create_diff()
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
                debugger.draw_image(board.diff, 0)
                debugger.draw_image(board.result, 2)
                print(f"{c=}, {temp=:.4f}, {board.quality=:.4f}")
                if c > 0.9 * step_count:
                    temp = temp * 0.2
            print("Done")
        elif key == ord("q"):
            print("Quitting...")
            debugger.terminate()
        else:
            print("Unknown key pressed")
            print(key)


if __name__ == "__main__":
    main()
