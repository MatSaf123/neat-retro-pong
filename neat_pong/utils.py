from typing import Optional, Tuple, List
import numpy as np
import cv2


def preprocess_frame(frame) -> np.ndarray:
    """Get rid of player's score and colors, leaving
    only interpolated, resized image of two paddles and the ball."""
    frame = frame[34:-16, 5:-4]
    frame = np.average(frame, axis=2)
    frame = cv2.resize(frame, (84, 84), interpolation=cv2.INTER_NEAREST)
    frame = np.array(frame, dtype=np.uint8)

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

    # Search for 236 value, these mean ball body
    coords = np.argwhere(frame == 236)

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
    in other words, computes median of x's and median of y's based on list
    of pixels building the game object."""

    # Separete x and y values in pixel coords into two lists
    x_vals = [c[0] for c in pixel_coords]
    y_vals = [c[1] for c in pixel_coords]

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
        # We look for player's y coords, x are const; this means
        # we can simply look in the x rows that the player is always present: 76 and 77.
        for x in [76, 77]:
            if frame[y][x] == 123:  # Search for 123 value, these mean paddle body
                pixels_coords.append((x, y))
            elif frame[y][x] == 146:
                # For whatever reason in pong no-frameskip first frame has different colors,
                # so that's the case for it
                pixels_coords.append((x, y))
            else:
                continue

    return pixels_coords


def ball_has_hit_right_paddle(
    paddle_pixel_coords: List[Tuple[float, float]],
    ball_pixel_coords: List[Tuple[float, float]],
) -> bool:
    """AI should `like` hitting the ball with it's paddle
    and `love` getting the ball behind opponent's paddle.
    Therefore give one point when AI succesfully hits the ball
    - but that has to be lower than the points for scoring a goal.

    With this function we try to detect collision between player's paddle
    and the ball.
    """

    # Get ball's right edge pixel coordinate, I'm not sure if paddle can hit
    # the ball with it's top or bottom, but that's simplification of things
    most_right_ball_x = max([p[1] for p in ball_pixel_coords])

    # Get pixels corresponding to coords above
    ball_right_edge_pixel_coords = [
        pixel_coord
        for pixel_coord in ball_pixel_coords
        if pixel_coord[1] == most_right_ball_x
    ]

    # Check if these pixels are within x=76, that's the only way a ball can hit
    # the front of a paddle. Again, not sure about top/bottom, but we're simplifying.
    if not all([px_coords[0] == 76 for px_coords in ball_right_edge_pixel_coords]):
        return False

    # Finally, check if any of ball's pixels connect with paddle pixels - meaning, if those two objects connect
    objects_connect = False
    for ball_pixel_coord in ball_right_edge_pixel_coords:
        for paddle_pixel_coord in paddle_pixel_coords:

            # Check if y is the same
            if ball_pixel_coord[1] != paddle_pixel_coord[1]:
                continue

            # Check if x pixels are right next to each other
            # on x-axis -> take paddle[x] - ball[x] and check if
            # it's equal to one. If yes, we got a hit!
            if paddle_pixel_coord[0] - ball_pixel_coord[0] != 1:
                continue

            # We got a hit!
            objects_connect = True
            break

        # Break the ball points loop too
        if objects_connect:
            break

    return objects_connect
