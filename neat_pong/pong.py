# Re-implementation of Python-NEAT Pong game, but this time with usage of openai gym.
import os
from typing import Optional
import gym
import numpy as np
import pickle
import neat
from .utils import preprocess_frame, get_player_paddle_position, get_ball_position
from pathlib import Path
from datetime import datetime


def make_ai_play_game(
    net: neat.nn.FeedForwardNetwork,
    env: gym.Env,
    game_rounds: int,
    timesteps: int,
    render: bool = False,
) -> float:
    """Play game of pong with given neural network as one of the players
    (second player is provided by gym environment)."""

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

                outputs = net.activate((player_x, player_y, player_ball_dist))
                # print("outputs:", outputs)
                ai_move = (
                    np.argmax(outputs) + 1
                )  # function returns 0, 1, 2; controlls are 1, 2, 3
                # print("ai_move:", ai_move)
                frame, reward, done, _, info = env.step(ai_move)
                last_frame = preprocess_frame(frame)

            if render:
                env.render()
            if done:
                break

            reward += reward  # TODO: ?????
        fitnesses.append(reward)

    fitness = np.array(fitnesses).mean()
    # print(f"Special fitness: {fitness}")
    return fitness


def train_ai(env: gym.Env, config, checkpoint_filename: Optional[str] = None):
    def evaluate_genome(genome, config):
        print("evaluate_genome for:", genome[0])
        gen_id, gen = genome
        net = neat.nn.FeedForwardNetwork.create(gen, config)
        score = make_ai_play_game(net, env, 1, 200, False)  # Don't render
        return score

    def evaluate_fitness(genomes, config):
        for genome in genomes:
            print("evaluate_fitness for:", genome[0])
            fitness = evaluate_genome(genome, config)
            print(f"fitness of genome {genome[0]}=", fitness)
            genome[1].fitness = fitness

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

    pop.run(evaluate_fitness, 20)
    pop.save_checkpoint("checkpoint")

    # Show output of the most fit genome against training data.
    winner = pop.statistics.best_genome()

    # Save best network
    with open("winner_{}.pkl".format(datetime.now().timestamp()), "wb") as output:
        pickle.dump(winner, output, 1)


def run(mode: str, checkpoint_filepath: str):

    # NOTE Apparently pong env has no seed control: https://www.youtube.com/watch?v=WnSUQdFnKyY

    # TODO FIX BIG PYGAME WINDOW WHEN RENDERING https://github.com/openai/gym/issues/550

    environment = gym.make(
        "PongDeterministic-v4", render_mode="human"
    )  # TODO: find a way to speed this up

    # Load config
    config_path = Path("config.txt").resolve()

    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path,
    )

    # TODO take arg from command line that runs either test or train mode
    if mode == "train":
        train_ai(environment, config, checkpoint_filepath)
    elif mode == "test":
        # test_ai() # TODO implement?
        pass
    else:
        raise Exception("No such mode as {} in neat-python".format(mode))


if __name__ == "__main__":
    pass
