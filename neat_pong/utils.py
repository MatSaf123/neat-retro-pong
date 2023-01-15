from typing import Tuple, List
import numpy as np
import cv2


def preprocess_frame(frame) -> np.ndarray:
    """Get rid of player's score and colors, leaving
    only interpolated, resized image of two paddles and the ball."""
    frame = frame[
        34:-16, 5:-4
    ]  # note to self: I changed the cropping even more than was in original
    frame = np.average(frame, axis=2)
    frame = cv2.resize(frame, (84, 84), interpolation=cv2.INTER_NEAREST)
    frame = np.array(frame, dtype=np.uint8)
    return frame


def get_ball_position(frame) -> Tuple[float, float]:
    coords = []
    for y in range(len(frame)):
        for x in range(len(frame[0])):
            if frame[y][x] == 236:  # Search for 236 value, these mean ball body
                coords.append((x, y))

    if not coords:
        return None, None

    coords = np.mean([c[0] for c in coords]), np.mean([c[1] for c in coords])

    return coords


def get_player_paddle_position(frame: List[List[int]]) -> Tuple[float, float]:
    # Player is on the right side of the screen, so look for
    # the "first" paddle from the right edge.

    y_vals = []
    for y in range(len(frame)):
        # We look for player's y coords, x are const; this means
        # we can simply look in the x rows that the player is always
        # present: 76 and 77.
        for x in [76, 77]:
            if frame[y][x] == 123:  # Search for 123 value, these mean paddle body
                y_vals.append(y)

    paddle_x_coord = 76.5
    y_vals_len = len(y_vals)
    paddle_y_coord = np.median(
        (y_vals[y_vals_len // 2] + y_vals[y_vals_len // 2 + 1]) / 2
    )

    return paddle_x_coord, paddle_y_coord
