from typing import Optional, Tuple, List
import numpy as np
import cv2

# Paddles can't move vertically, so these are constant
PLAYER_X_COORDS = 81, 82, 83
# PLAYER_X_COORDS = list(range(154, 160))

BALL_RGB = 236
PLAYER_PADDLE_RGB = 123, 146  # Two values possible
AI_PADDLE_RGB = 139
BACKGROUND_RGB = 77, 90  # Two values possible


def preprocess_frame(frame) -> np.ndarray:
    """Get rid of player's score and colors, leaving
    only interpolated, resized image of two paddles and the ball,
    and cut the sides of the frame."""

    frame = frame[34:-16, 17:-16]
    frame = np.average(frame, axis=2)
    frame = cv2.resize(frame, (84, 84), interpolation=cv2.INTER_NEAREST)
    frame = np.array(frame, dtype=np.uint8)

    # from matplotlib import pyplot as plt

    # plt.imshow(frame)
    # plt.show()

    return frame


def get_distance_between_points(
    point_a_coords: Tuple[float, float],
    point_b_coords: Tuple[float, float],
) -> float:
    """Get coordinates of point a and point b, compute and
    return distance between those two points."""

    x = np.array([point_a_coords[0], point_a_coords[1]])
    y = np.array([point_b_coords[0], point_b_coords[1]])
    return np.linalg.norm(x - y)


def get_ball_pixel_coords(frame) -> Optional[List[Tuple[float, float]]]:
    """Go through all pixels present in frame and look
    for ones with RGB value of 236: these are the building blocks
    for ball. Return a list of coordinates of these pixels."""

    # Search for ball pixel coords
    coords = np.argwhere(frame == BALL_RGB)

    # Map to list of tuples. For whatever reason
    # order of thes x-y coords is swapped, so swap them back
    coords = [(ele[1], ele[0]) for ele in coords]

    # In case the ball was nowhere to be found,
    # for whatever reason, return None
    if not coords:
        return None
    else:
        return coords


def get_object_position(pixel_coords: List[Tuple[float, float]]) -> Tuple[float, float]:
    """Computes coordinates of object based on it's pixels coordinates:
    in other words, computes avg of x's and median of y's based on list
    of pixels building the game object."""

    # Separete x and y values in pixel coords into two lists
    x_vals = list(set([c[0] for c in pixel_coords]))
    y_vals = list(set([c[1] for c in pixel_coords]))

    # Get "center" value of x's
    if len(x_vals) == 1:
        x_pos = x_vals[0]
    else:
        x_pos = (max(x_vals) + min(x_vals)) / 2

    # Get "center" value of y's
    if len(y_vals) == 1:
        y_pos = y_vals[0]
    else:
        y_pos = (max(y_vals) + min(y_vals)) / 2

    return x_pos, y_pos


def get_player_paddle_pixel_coords(frame) -> Optional[List[Tuple[float, float]]]:
    """Go through part of pixels in the frame and look for pixels matching
    the RGB of player's (or AI's, in our case) paddle. Since x coord of a paddle
    is const, we only look for y values."""

    pixels_coords = []

    for y in range(len(frame)):
        for x in PLAYER_X_COORDS:
            if (
                frame[y][x] == PLAYER_PADDLE_RGB[0]
                or frame[y][x] == PLAYER_PADDLE_RGB[1]
            ):
                pixels_coords.append((x, y))

    return pixels_coords
