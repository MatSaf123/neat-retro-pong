from typing import Optional, Tuple, List
import numpy as np
import cv2


def preprocess_frame(frame) -> np.ndarray:
    """Get rid of player's score and colors, leaving
    only interpolated, resized image of two paddles and the ball."""
    frame = frame[
        34:-16, 5:-4
    ]  # NOTE I changed the cropping even more than was in original tutorial
    frame = np.average(frame, axis=2)
    frame = cv2.resize(frame, (84, 84), interpolation=cv2.INTER_NEAREST)
    frame = np.array(frame, dtype=np.uint8)
    return frame


def get_distance_between_points(
    point_a_coords: Tuple[float, float],
    point_b_coords: Tuple[float, float],
) -> float:
    # TODO this could be prettier probably
    x = np.array([point_a_coords[0], point_a_coords[1]])
    y = np.array([point_b_coords[0], point_b_coords[1]])
    return np.linalg.norm(x - y)


def get_ball_pixel_coords(frame) -> Optional[List[Tuple[float, float]]]:

    # Search for 236 value, these mean ball body
    coords = np.argwhere(frame == 236)

    # Map to list of tuples. For whatever reason these x-y coords are swapped,
    # so swap them back
    coords = [(ele[1], ele[0]) for ele in coords]

    # print(coords)

    if not coords:
        return None

    return coords


def get_object_position(pixel_coords: List[Tuple[float, float]]) -> Tuple[float, float]:
    """TODO"""

    # print(pixel_coords)

    x_vals = [c[0] for c in pixel_coords]
    y_vals = [c[1] for c in pixel_coords]

    if len(x_vals) == 1:
        x_pos = x_vals[0]
    else:
        x_pos = (max(x_vals) + min(x_vals)) / 2

    if len(y_vals) == 1:
        y_pos = y_vals[0]
    else:
        y_pos = (max(y_vals) + min(y_vals)) / 2

    return x_pos, y_pos


def get_player_paddle_pixel_coords(frame) -> Optional[List[Tuple[float, float]]]:
    # Player is on the right side of the screen, so look for
    # the "first" paddle from the right edge.

    # y_vals = []
    pixel_coords = []
    for y in range(len(frame)):
        # We look for player's y coords, x are const; this means
        # we can simply look in the x rows that the player is always
        # present: 76 and 77.
        for x in [76, 77]:
            if frame[y][x] == 123:  # Search for 123 value, these mean paddle body
                # y_vals.append(y)
                pixel_coords.append((x, y))
            elif frame[y][x] == 146:
                # For whatever reason in pong no-frameskip first frame has different colors, lol
                pixel_coords.append((x, y))

    # # TODO UGLY, REDO
    # pixel_coords = []
    # pixel_coords.extend([(76, y) for y in y_vals])
    # pixel_coords.extend([(77, y) for y in y_vals])

    return pixel_coords


def ball_has_hit_right_paddle(
    paddle_pixel_coords: List[Tuple[float, float]],
    ball_pixel_coords: Optional[List[Tuple[float, float]]],
) -> bool:
    """AI should `like` hitting the ball with it's paddle
    and `love` getting the ball behind opponent's paddle.
    Therefore give one point when AI succesfully hits the ball
    - but that has to be lower than the points for scoring a goal!"""

    # print("\npxc: {}\nbpc:{}\n".format(paddle_pixel_coords, ball_pixel_coords))

    # If ball is not yet on the map
    if not ball_pixel_coords:
        return False

    # Get ball' right edge pixels, ball can collide with the paddle only with this side
    most_right_ball_x = max([p[1] for p in ball_pixel_coords])

    ball_right_edge_pixel_coords = [
        pixel_coord
        for pixel_coord in ball_pixel_coords
        if pixel_coord[1] == most_right_ball_x
    ]

    # Check if these pixels are within x=76, that's the only (?) x pos that it's able to collide with paddle
    if not all([px_coords[0] == 76 for px_coords in ball_right_edge_pixel_coords]):
        return False

    # Finally, check if any of ball' pixels connect with paddle pixels - meaning, if those two objects connect
    objects_connect = False
    for b_coords in ball_right_edge_pixel_coords:
        for p_coords in paddle_pixel_coords:

            # Check if y is the same
            if b_coords[1] != p_coords[1]:
                continue

            # Check if x pixels are right next to each other
            # on x-axis -> take paddle[x] - ball[x] and check if
            # it's equal to one. If yes, we got a hit!

            if p_coords[0] - b_coords[0] != 1:
                continue

            # We got a hit!
            objects_connect = True
            break

        # Break the ball points loop too
        if objects_connect:
            break

    return objects_connect


def ball_has_hit_right_paddle_V2(
    ball_coords: Tuple[float, float],
    paddle_pixel_coords: List[Tuple[float, float]],
    ball_paddle_dist: float,
):
    """Or, was close to hit, I guess?"""

    # paddle_y_values = [p[1] for p in paddle_pixel_coords]

    # # NOTE to be changed, also put it somewhere as a const
    # accepted_distance = 3.0

    # if ball_paddle_dist > accepted_distance:
    #     print("too far")
    #     return False

    # # Remember that pygame reverses axis!
    # most_top_pixel = min(paddle_y_values)
    # most_bottom_pixel = max(paddle_y_values)

    # # Check if ball is in between y scope of paddle (
    # # or rather check if it's NOT and return False in such case)
    # if ball_coords[1] < most_top_pixel or ball_coords[1] > most_bottom_pixel:
    #     print(
    #         "{} not between {} and {}".format(
    #             ball_coords[1], most_top_pixel, most_bottom_pixel
    #         )
    #     )
    #     return False

    # print(
    #     "ball: {} paddle: {}".format(
    #         int(ball_coords[1]), [p[1] for p in paddle_pixel_coords]
    #     )
    # )

    # # NOTE we parse float to integer here because ball y is often 0.5, 2.5 ...
    # # It could be a problem later but we're improvising for now
    # # print(ball_coords)
    # paddle_points_at_ball_y = [
    #     point for point in paddle_pixel_coords if point[1] == ball_coords[1]
    # ]

    # # print(paddle_points_at_ball_y)

    # if not paddle_points_at_ball_y:
    #     return False

    # # Get the pixel most to the left
    # match_point = min(paddle_points_at_ball_y, key=lambda t: t[1])

    # # print("ball:{} match_point:{}".format(ball_coords, match_point))

    # dist = get_distance_between_points(ball_coords, match_point)
    # # print("dist: {}".format(dist))

    # if dist < 2.0:
    #     print("TRUE")
    #     return True
    # else:
    #     print("FALSE")

    if ball_coords[0] > 74 and ball_coords[0] < 77:
        # Works for now, but TODO redo cause ugly

        paddle_y_values = [p[1] for p in paddle_pixel_coords]

        # NOTE to be changed, also put it somewhere as a const
        accepted_distance = 3.0

        # Remember that pygame reverses axis!
        most_top_pixel = min(paddle_y_values)
        most_bottom_pixel = max(paddle_y_values)

        if ball_coords[1] > most_top_pixel and ball_coords[1] < most_bottom_pixel:
            if ball_paddle_dist < accepted_distance:
                return True
        else:
            return False
    else:
        print("{} is not valid".format(ball_coords[0]))
        return False

    return False
