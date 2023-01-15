# Re-implementation of Python-NEAT Pong game, but this time with usage of openai gym.
import os
from typing import Optional
import gym
import numpy as np
import pickle
import neat
from .utils import (
    preprocess_frame,
    get_player_paddle_position,
    get_ball_position,
    enemy_scored,
)
from pathlib import Path
from datetime import datetime


def eval_genome(genome, config) -> float:
    """This function has to be on top of the file to allow
    mutliprocessing logic to utilize it"""

    print("Running eval_genome for:", genome.key)

    net = neat.nn.FeedForwardNetwork.create(genome, config)
    score = make_ai_play_game(net, 4, 200, False)  # Don't render

    print("Score for genome {}: {}".format(genome.key, score))

    genome.score = score  # NOTE 1. its fitness not score 2. ParallelEv should be doing that already
    return score


def make_ai_play_game(
    net: neat.nn.FeedForwardNetwork,
    game_rounds: int,
    timesteps: int,
    render: bool = False,
) -> float:
    """Play game of pong with given neural network as one of the players
    (second player is provided by gym environment)."""

    env = gym.make(  # TODO make this singleton?
        "Pong-v4",
        render_mode="rgb_array",  # TODO this is hardcoded again, fix somehow to control with CLI
    )  # TODO: find a way to speed this up

    fitnesses = []

    for _ in range(game_rounds):

        env.reset()

        init_frame = env.step(0)[0]
        state = preprocess_frame(init_frame)  # Initial environment state, inputs for NN

        reward = 0.0  # TODO: ???

        last_frame = state

        for _ in range(timesteps):

            ball_x, ball_y = get_ball_position(last_frame)
            player_x, player_y = get_player_paddle_position(last_frame)

            if not ball_x or not ball_y:
                frame, reward, done, _, info = env.step(1)  # Stay still
                last_frame = preprocess_frame(frame)
            else:
                player_ball_dist = np.linalg.norm(
                    [(ball_x, ball_y), (player_x, player_y)]
                )

                # print((player_x, player_y, player_ball_dist))
                outputs = net.activate((player_x, player_y, player_ball_dist))
                # print("outputs:", outputs)
                ai_move = (
                    np.argmax(outputs) + 1
                )  # function returns 0, 1, 2; controlls are 1, 2, 3
                # print("{}: ai_move:".format(datetime.now()), ai_move)
                frame, reward, done, _, info = env.step(ai_move)
                last_frame = preprocess_frame(frame)

                if reward == 1:
                    print("*************************************")
                    print((player_x, player_y), ai_move, reward)
                    print("*************************************")

            if enemy_scored(frame):
                # We don't want some casul NN that loses points
                if reward == 0:
                    reward -= 1.0  # NOTE thats debatable
                # print("bailin")
                break

            if render:
                env.render()
            if done:
                break

            # reward += reward  # TODO: ?????
        fitnesses.append(reward)

    fitness = np.array(fitnesses).mean()
    print("fitnesses:{} fitness:{}".format(fitnesses, fitness))
    # print(f"Special fitness: {fitness}")
    return fitness


def train_ai(config, checkpoint_filename: Optional[str] = None):

    if checkpoint_filename:
        # Load checkpoint if path was passed
        print("***\nLoading checkpoint: {}\n***".format(checkpoint_filename))
        checkpoint_path = Path(checkpoint_filename).resolve()
        pop = neat.Checkpointer.restore_checkpoint(checkpoint_path)
    else:
        # Otherwise create new population
        print("***\nRunning neat with new Population\n***")
        pop = neat.Population(config)

    pop.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    pop.add_reporter(neat.Checkpointer(1))

    # Initialize pararell evaluator
    pe = neat.ParallelEvaluator(8, eval_genome)
    winner = pop.run(pe.evaluate, 20)

    # Show output of the most fit genome against training data.
    print(winner)

    # Save best network
    with open("winner_{}.pkl".format(datetime.now().timestamp()), "wb") as output:
        pickle.dump(winner, output, 1)


def test_ai(filepath: str, config):
    pass


def run(mode: str, render_mode: str, checkpoint_filepath: str):

    # NOTE Apparently pong env has no seed control: https://www.youtube.com/watch?v=WnSUQdFnKyY

    # TODO FIX BIG PYGAME WINDOW WHEN RENDERING https://github.com/openai/gym/issues/550

    # Load config
    config_path = Path("config.txt").resolve()

    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path,
    )

    if mode == "train":
        train_ai(config, checkpoint_filepath)
    elif mode == "test":
        test_ai()  # TODO implement?
        pass
    else:
        raise Exception("No such mode as {} in neat-pong".format(mode))


if __name__ == "__main__":
    pass
