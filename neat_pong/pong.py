# Re-implementation of Python-NEAT Pong game, but this time with usage of openai gym.
from typing import Optional
import gym
import numpy as np
import pickle
import neat
from matplotlib import pyplot as plt

from .utils import (
    get_distance_between_points,
    preprocess_frame,
    get_player_paddle_pixel_coords,
    get_ball_pixel_coords,
    get_object_position,
    ball_has_hit_right_paddle,
    ball_has_hit_right_paddle_V2,
)
from pathlib import Path
from datetime import datetime

CENTER_OF_MAP_COORDS = 41.0, 43.5  # Almost every time!


class PongEnv:

    _env: gym.Env

    def __init__(self):

        self._env = gym.make(
            "PongNoFrameskip-v4",
            render_mode="rgb_array",  # TODO this is hardcoded again, fix somehow to control with CLI
            # render_mode="human",  # TODO this is hardcoded again, fix somehow to control with CLI
        )

    @property
    def get_env(self):
        if not self._env:
            raise Exception("Got no env initialized")
        else:
            return self._env


def eval_genome(genome, config) -> float:
    """This function has to be on top of the file to allow
    mutliprocessing logic to utilize it"""

    net = neat.nn.FeedForwardNetwork.create(genome, config)
    score = make_ai_play_game(net, 2000, False)  # Don't render

    return score


def make_ai_play_game(
    net: neat.nn.FeedForwardNetwork,
    timesteps: int,
    render: bool = False,
) -> float:
    """Play game of pong with given neural network as one of the players
    (second player is provided by gym environment)."""

    env = PongEnv().get_env
    env.reset()

    # Set initial environment state
    init_frame = env.step(0)[0]
    target_frame = preprocess_frame(init_frame)

    # Let's pretend that AI stays in place when sim inits
    ai_move = 0, 0

    # Set initial genome fitness to 0
    fitness = 0.0

    # After AI (our AI, the one on right) hits ball with its paddle, reset counter
    # to zero. This way when we detect collision we can see
    # if some time has passed from the last hit and NOT
    # count three hits at once for example, because sometimes
    # ball stays a bit longer than one frame in the range of a padle.
    frames_since_last_hit = 0

    for _ in range(timesteps):
        ball_pixel_coords = get_ball_pixel_coords(target_frame)

        if not ball_pixel_coords:
            ball_x, ball_y = CENTER_OF_MAP_COORDS
        else:
            ball_x, ball_y = get_object_position(ball_pixel_coords)

        player_pixel_coords = get_player_paddle_pixel_coords(target_frame)
        player_x, player_y = get_object_position(player_pixel_coords)

        player_ball_dist = get_distance_between_points(
            (ball_x, ball_y), (player_x, player_y)
        )

        outputs = net.activate((player_y, ball_y, player_ball_dist))

        ai_move = (
            np.argmax(outputs) + 1
        )  # function returns 0, 1, 2; controlls are 1, 2, 3

        # Take action and prepare frame for next loop iteration
        frame, reward, done, _, info = env.step(ai_move)
        target_frame = preprocess_frame(frame)

        # Add either 1 for goal scored, 0 for nothing and -1 for point lost in this frame
        fitness += reward

        # If AI hits ball with it's paddle, give it 1 points
        if (
            ball_has_hit_right_paddle(player_pixel_coords, ball_pixel_coords)
            and frames_since_last_hit > 10
        ):
            print("hit!")
            fitness += 0.5  # Add half of point for just hitting the ball
            # Set to -1 because we'll add 1 at the end of this loop anyway
            frames_since_last_hit = -1

        frames_since_last_hit += 1
        if render:
            env.render()
        if done:
            break

    return fitness


def train_ai(config, checkpoint_filename: Optional[str] = None):

    if checkpoint_filename:
        # Load checkpoint if path was passed
        print("***\nLoading checkpoint: {}\n***".format(checkpoint_filename))
        checkpoint_path = Path("checkpoints", checkpoint_filename).resolve()
        pop = neat.Checkpointer.restore_checkpoint(checkpoint_path)
    else:
        # Otherwise create new population
        print("***\nRunning neat with new Population\n***")
        pop = neat.Population(config)

    pop.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    pop.add_reporter(
        neat.Checkpointer(1, filename_prefix="checkpoints/neat-checkpoint-")
    )

    # Initialize pararell evaluator
    pe = neat.ParallelEvaluator(8, eval_genome)  # TODO take workers num from cli arg
    winner = pop.run(pe.evaluate, 1)

    # Show output of the most fit genome against training data.
    print(winner)

    # Save best network
    filename = "winner_{}.pkl".format(datetime.now().timestamp())
    with open(filename, "wb") as output:
        pickle.dump(winner, output, 1)

    print("Saved to: {}".format(filename))


def test_ai(config, filepath: str):

    # Load model from pickle
    with open(filepath, "rb") as f:
        saved_model = pickle.load(f)
    net = neat.nn.FeedForwardNetwork.create(saved_model, config)

    env = gym.make(  # TODO make this singleton?
        "PongNoFrameskip-v4",
        render_mode="human",  # TODO this is hardcoded again, fix somehow to control with CLI
    )

    env.reset()

    init_frame = env.step(0)[0]
    state = preprocess_frame(init_frame)  # Initial environment state, inputs for NN
    target_frame = state

    while True:

        ball_pixel_coords = get_ball_pixel_coords(target_frame)

        if not ball_pixel_coords:
            ball_x, ball_y = CENTER_OF_MAP_COORDS
        else:
            ball_x, ball_y = get_object_position(ball_pixel_coords)

        player_pixel_coords = get_player_paddle_pixel_coords(target_frame)
        player_x, player_y = get_object_position(player_pixel_coords)

        player_ball_dist = get_distance_between_points(
            (ball_x, ball_y), (player_x, player_y)
        )

        outputs = net.activate((player_y, ball_y, player_ball_dist))

        ai_move = (
            np.argmax(outputs) + 1
        )  # function returns 0, 1, 2; controlls are 1, 2, 3

        frame, _, done, _, info = env.step(ai_move)
        target_frame = preprocess_frame(frame)

        env.render()

        if done:
            break


def run(mode: str, render_mode: str, filepath: str):
    """Filepath leads either to checkpoint file (for training) or
    net pickled into a file (for testing)."""

    # Load config
    config_path = Path("config.txt").resolve()

    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path,
    )

    # initialize singleton?

    if mode == "train":
        train_ai(config, filepath)
    elif mode == "test":
        test_ai(config, filepath)
    else:
        raise Exception("No such mode as {} in neat-pong".format(mode))


if __name__ == "__main__":
    pass
