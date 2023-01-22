# Re-implementation of Python-NEAT Pong game, but this time with usage of openai gym.
import os
from typing import Optional
import gym
import numpy as np
import pickle
import neat

from .utils import (
    get_distance_between_points,
    preprocess_frame,
    get_player_paddle_pixel_coords,
    get_ball_pixel_coords,
    get_object_position,
    ball_has_hit_right_paddle,
)
from pathlib import Path
from datetime import datetime

CENTER_OF_MAP_COORDS = 41.0, 43.5  # Almost every time!


class PongEnv:

    _env: gym.Env

    def __init__(self):

        self._env = gym.make(
            "Pong-v4",
            render_mode="rgb_array",  # TODO this is hardcoded again, fix somehow to control with CLI
            # render_mode="human",  # TODO this is hardcoded again, fix somehow to control with CLI
        )

    @property
    def get_env(self):
        if not self._env:
            raise Exception("u dumbass")
        else:
            return self._env


def eval_genome(genome, config) -> float:
    """This function has to be on top of the file to allow
    mutliprocessing logic to utilize it"""

    net = neat.nn.FeedForwardNetwork.create(genome, config)
    score = make_ai_play_game(net, 500, False, genome.key)  # Don't render

    return score


def make_ai_play_game(
    net: neat.nn.FeedForwardNetwork,
    timesteps: int,
    render: bool = False,
    genome_key: str = "",
) -> float:
    """Play game of pong with given neural network as one of the players
    (second player is provided by gym environment)."""

    env = PongEnv().get_env

    env.reset()

    init_frame = env.step(0)[0]
    state = preprocess_frame(init_frame)  # Initial environment state, inputs for NN

    fitness = 0.0

    last_frame = state

    for _ in range(timesteps):

        ball_pixel_coords = get_ball_pixel_coords(last_frame)

        if not ball_pixel_coords:
            ball_x, ball_y = CENTER_OF_MAP_COORDS
        else:
            ball_x, ball_y = get_object_position(ball_pixel_coords)

        player_pixel_coords = get_player_paddle_pixel_coords(last_frame)
        player_x, player_y = get_object_position(player_pixel_coords)

        player_ball_dist = get_distance_between_points(
            (ball_x, ball_y), (player_x, player_y)
        )

        outputs = net.activate((player_y, ball_y, player_ball_dist))

        ai_move = (
            np.argmax(outputs) + 1
        )  # function returns 0, 1, 2; controlls are 1, 2, 3
        frame, reward, done, _, info = env.step(ai_move)
        last_frame = preprocess_frame(frame)

        # Either 1 for goal scored, 0 for nothing and -1 for point lost in this frame
        fitness += reward

        # If AI hits ball with it's paddle, give it 0.5 points
        if ball_has_hit_right_paddle(player_pixel_coords, ball_pixel_coords):
            fitness += 0.5

        if render:
            env.render()
        if done:
            break

    # TODO maybe add a small reward for staying close to the ball?
    # TODO Add points for time "survived" as well?
    # TODO add points for not spamming up/down but staying in place?

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
    winner = pop.run(pe.evaluate, 40)

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
        "Pong-v4",
        render_mode="human",  # TODO this is hardcoded again, fix somehow to control with CLI
    )

    env.reset()

    init_frame = env.step(0)[0]
    state = preprocess_frame(init_frame)  # Initial environment state, inputs for NN
    last_frame = state

    while True:

        ball_pixel_coords = get_ball_pixel_coords(last_frame)
        if not ball_pixel_coords:
            ball_x, ball_y = CENTER_OF_MAP_COORDS
        else:
            ball_x, ball_y = get_object_position(ball_pixel_coords)

        player_pixel_coords = get_player_paddle_pixel_coords(last_frame)
        player_x, player_y = get_object_position(player_pixel_coords)

        player_ball_dist = get_distance_between_points(
            (ball_x, ball_y), (player_x, player_y)
        )

        outputs = net.activate((player_y, ball_y, player_ball_dist))
        ai_move = np.argmax(outputs) + 1

        print("AI move: {}".format(ai_move))

        frame, _, done, _, info = env.step(ai_move)
        last_frame = preprocess_frame(frame)

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
    PongEnv()

    if mode == "train":
        train_ai(config, filepath)
    elif mode == "test":
        test_ai(config, filepath)
    else:
        raise Exception("No such mode as {} in neat-pong".format(mode))


if __name__ == "__main__":
    pass
