#!/usr/bin/env python3
import argparse
from os import path
from os.path import isfile
from sys import stderr

import numpy as np
from cv2 import cv2


def place_on_top(a, b, upper_left):
    """

    :param a:
    :param b:
    :param upper_left: (row index, col index)
    """
    r, c = upper_left
    b[r : r + a.shape[0], c : c + a.shape[1]] = a


def main():
    parser = argparse.ArgumentParser(
        description="combine two images with an arrow in between. Spacing and color is automatically decided and is meant to be nice"
    )

    parser.add_argument(
        "image_file_1", action="store", type=str, help="the image file in the left"
    )
    parser.add_argument(
        "image_file_2", action="store", type=str, help="the image file in the right"
    )
    parser.add_argument(
        "--output",
        "-o",
        action="store",
        required=False,
        type=str,
        help="the output image file. (e.g. out.jpg out.png) If omitted, <img1>-<img2>.png will be generated under current directory",
    )

    parser.add_argument(
        "--scale",
        "-s",
        required=False,
        default=1.0,
        action="store",
        type=float,
        help="the scale of the generated image, 1 for no scaling. 0.5 for half the size, etc",
    )

    argv = parser.parse_args()

    img1 = cv2.imread(argv.image_file_1, 1)
    img2 = cv2.imread(argv.image_file_2, 1)

    s_height = min(img1.shape[0], img2.shape[0])
    b_height = max(img1.shape[0], img2.shape[0])

    s_width = min(img1.shape[1], img2.shape[1])
    b_width = max(img1.shape[1], img2.shape[1])

    frame = np.full((b_height, 2 * s_width + b_width, 3), 255, dtype=np.uint8)

    place_on_top(img1, frame, [(b_height - img1.shape[0]) // 2, 0])
    place_on_top(
        img2, frame, [(b_height - img2.shape[0]) // 2, s_width + img1.shape[1]]
    )

    m1 = cv2.mean(img1)
    m2 = cv2.mean(img2)

    mean_color = []
    for i in range(3):
        mean_color.append(int(m1[i] + m2[i]) // 2)

    cv2.arrowedLine(
        frame,
        (img1.shape[1] + s_width // 5, b_height // 2),
        (img1.shape[1] + s_width - s_width // 5, b_height // 2),
        mean_color,
        8,
    )

    assert argv.scale > 0, "scale has to be a positive float"
    frame = cv2.resize(
        frame, (int(frame.shape[1] * argv.scale), int(frame.shape[0] * argv.scale))
    )

    outname = argv.output
    if argv.output is not None:
        try:
            cv2.imwrite(argv.output, frame)
        except cv2.error as e:
            print(e, file=stderr)
            print("Failed to save the image")
            print("Did you forget to specify image format to the output file?")
    else:
        default_name = (
            path.splitext(path.basename(argv.image_file_1))[0]
            + "-"
            + path.splitext(path.basename(argv.image_file_2))[0]
            + ".png"
        )
        outname = default_name
        cv2.imwrite(default_name, frame)

    cv2.imshow(outname, frame)
    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == "__main__":

    main()
