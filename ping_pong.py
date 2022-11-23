# Re-implementation of Python-NEAT Pong game, but this time with usage of openai gym.
import os
import gym
import numpy as np
import pickle
import neat
import cv2
import matplotlib.pyplot as plt
from typing import List, Tuple


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
    paddle_y_coord = np.median(
        (y_vals[3] + y_vals[4]) / 2
    )  # Paddle has a length of 8, so take the mean of two middle coords\

    return paddle_x_coord, paddle_y_coord


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
                ai_move = np.argmax(outputs)
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


def train_ai(env: gym.Env, config):
    def evaluate_genome(genome, config):
        print("evaluate_genome for:", genome[0])
        gen_id, gen = genome
        net = neat.nn.FeedForwardNetwork.create(gen, config)
        score = make_ai_play_game(net, env, 1, 200, True)
        return score

    def evaluate_fitness(genomes, config):
        for genome in genomes:
            print("evaluate_fitness for:", genome[0])
            fitness = evaluate_genome(genome, config)
            print(f"fitness of genome {genome[0]}=", fitness)
            genome[1].fitness = fitness

    pop = neat.Population(config)

    # Load checkpoint
    # TODO: read checkpoint if exists and if arg is passed

    pop.run(evaluate_fitness, 10)
    pop.save_checkpoint("checkpoint")

    pop.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    pop.add_reporter(neat.Checkpointer(1))

    # Show output of the most fit genome against training data.
    winner = pop.statistics.best_genome()

    # Save best network
    with open("winner.pkl", "wb") as output:
        pickle.dump(winner, output, 1)


if __name__ == "__main__":

    environment = gym.make(
        "Pong-v4", render_mode="human"
    )  # TODO: find a way to speed this up
    # environment.metadata["render_fps"] = 120 # Doesn't work

    # environment = gym.make("PongDeterministic-v4")
    # Load config
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config.txt")

    # TODO: this should fix stagnation_type attr, possibly
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path,
    )

    train_ai(environment, config)
    # test_ai()
