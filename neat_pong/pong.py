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
from matplotlib import pyplot as plt


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

    # print(
    #     "Running eval_genome for: {} (curr fitness: {})".format(
    #         genome.key, genome.fitness
    #     )
    # )

    net = neat.nn.FeedForwardNetwork.create(genome, config)
    score = make_ai_play_game(net, 500, False)  # Don't render

    print("Returning score for genome {}: {}".format(genome.key, score))
    # genome.score = score  # NOTE 1. its fitness not score 2. ParallelEv should be doing that already
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

    init_frame = env.step(0)[0]
    state = preprocess_frame(init_frame)  # Initial environment state, inputs for NN

    fitness = 0.0

    last_frame = state

    for _ in range(timesteps):

        ball_pixel_coords = get_ball_pixel_coords(last_frame)
        # TODO do it cleaner
        if not ball_pixel_coords:
            ball_x, ball_y = None, None
        else:
            ball_x, ball_y = get_object_position(ball_pixel_coords)

        player_pixel_coords = get_player_paddle_pixel_coords(last_frame)
        player_x, player_y = get_object_position(player_pixel_coords)

        if not ball_x or not ball_y:
            frame, reward, done, _, info = env.step(1)  # Stay still
            last_frame = preprocess_frame(frame)
        else:
            player_ball_dist = get_distance_between_points(
                (ball_x, ball_y), (player_x, player_y)
            )

            # Display plot if ball-paddle are close
            # print((ball_x, ball_y), (player_x, player_y), player_ball_dist)
            # if player_ball_dist < 15:
            #     plt.imshow(last_frame)
            #     plt.show()

            outputs = net.activate((player_x, player_y, player_ball_dist))
            ai_move = (
                np.argmax(outputs) + 1
            )  # function returns 0, 1, 2; controlls are 1, 2, 3
            frame, reward, done, _, info = env.step(ai_move)
            last_frame = preprocess_frame(frame)

        # Add reward from last frame to fitness
        fitness += reward

        # TODO should be up
        if ball_has_hit_right_paddle(player_pixel_coords, ball_pixel_coords):
            print("HIT! adding +0.5")
            fitness += 0.5

        # # TODO should be up
        # if opponent_scored(frame):  # Pass the unprocessed frame with game score!
        #     # We don't want some casul NN that loses points
        #     if fitness == 0:
        #         fitness -= 1.0  # NOTE thats debatable
        #     # print("bailin")
        #     break

        if render:
            env.render()
        if done:
            break

    # reward += reward  # TODO: ?????
    # TODO maybe add a small reward for staying close to the ball?
    # TODO Add points for time "survived" as well?

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
    winner = pop.run(pe.evaluate, 60)

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
            ball_x, ball_y = None, None
        else:
            ball_x, ball_y = get_object_position(ball_pixel_coords)

        player_pixel_coords = get_player_paddle_pixel_coords(last_frame)
        player_x, player_y = get_object_position(player_pixel_coords)

        if not ball_x or not ball_y:
            print("No ball, staying still")
            frame, _, done, _, info = env.step(1)  # Stay still
            last_frame = preprocess_frame(frame)
        else:
            player_ball_dist = get_distance_between_points(
                (ball_x, ball_y), (player_x, player_y)
            )

            outputs = net.activate((player_x, player_y, player_ball_dist))
            ai_move = np.argmax(outputs) + 1

            print("AI move: {}".format(ai_move))

            frame, _, done, _, info = env.step(ai_move)
            last_frame = preprocess_frame(frame)
        # env.render()
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
        test_ai(config, filepath)  # TODO implement?
    else:
        raise Exception("No such mode as {} in neat-pong".format(mode))


if __name__ == "__main__":
    pass
