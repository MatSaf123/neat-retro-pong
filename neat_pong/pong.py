# Re-implementation of Python-NEAT Pong game, but this time with usage of openai gym.
from typing import Optional
import gym
import numpy as np
import pickle
import neat

from .visualize import draw_net, plot_stats, plot_species

from .utils import (
    get_distance_between_points,
    preprocess_frame,
    get_player_paddle_pixel_coords,
    get_ball_pixel_coords,
    get_object_position,
)
from pathlib import Path
from datetime import datetime

CENTER_OF_MAP_COORDS = 41.0, 43.5  # True almost every time!


class PongEnv:
    _env: gym.Env

    def __init__(self):
        self._env = gym.make(
            "ALE/Pong-v5",
            render_mode="rgb_array",  # TODO this is hardcoded again, fix somehow to control with CLI
            # render_mode="human",  # TODO this is hardcoded again, fix somehow to control with CLI
            frameskip=2,
            repeat_action_probability=0.0,
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

    # print("Genome {}: Starting simulation".format(genome.key))
    #
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    score = make_ai_play_game(net, 800, False)  # Don't render
    genome.fitness = score
    print("Genome {} fitness: {}".format(genome.key, genome.fitness))
    return score


def make_ai_play_game(
    net: neat.nn.FeedForwardNetwork,
    timesteps: int,
    render: bool = False,
) -> float:
    """Play game of pong with given neural network as one of the players
    (second player is provided by gym environment)."""

    env = PongEnv().get_env
    # Set initial environment state

    init_frame = env.reset()[0]
    target_frame = preprocess_frame(init_frame)

    # Let's pretend that AI stays in place when sim inits
    ai_move = None

    # Set initial genome fitness to 0
    fitness = 0.0

    for _ in range(timesteps):
        ball_pixel_coords = get_ball_pixel_coords(target_frame)

        if not ball_pixel_coords:
            ball_x, ball_y = CENTER_OF_MAP_COORDS
        else:
            ball_x, ball_y = get_object_position(ball_pixel_coords)

        player_pixel_coords = get_player_paddle_pixel_coords(target_frame)
        player_x, player_y = get_object_position(player_pixel_coords)

        player_ball_x_dist = get_distance_between_points((ball_x, 0), (player_x, 0))

        outputs = net.activate((player_y, ball_y, round(player_ball_x_dist, 2)))

        ai_move = (
            np.argmax(outputs) + 1
        )  # function returns 0, 1, 2 controlls are 1, 2, 3 for stay, up and down

        # Take action and prepare frame for next loop iteration
        frame, reward, done, _, info = env.step(ai_move)
        target_frame = preprocess_frame(frame)

        # Add either 1 for goal scored, 0 for nothing and -1 for point lost in this frame
        fitness += reward

        if render:
            print("rendering!")
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
    pe = neat.ParallelEvaluator(14, eval_genome)  # TODO take workers num from cli arg
    winner = pop.run(pe.evaluate, 10)

    # Show output of the most fit genome against training data.
    print(winner)

    # Save best network
    filename = "winner_{}.pkl".format(datetime.now().timestamp())
    with open(filename, "wb") as output:
        pickle.dump(winner, output, 1)

    print("Saved to: {}".format(filename))

    # Show info about stats
    plot_stats(stats, ylog=False, view=True)
    plot_species(stats, view=True)

    node_names = {
        -1: "Player Y",
        -2: "Ball Y",
        -3: "Player-Ball dist",
        0: "Stay",
        1: "Up",
        2: "Down",
    }

    draw_net(config, winner, True, node_names=node_names)


def test_ai(config, filepath: str):
    # Load model from pickle
    with open(filepath, "rb") as f:
        saved_model = pickle.load(f)
    net = neat.nn.FeedForwardNetwork.create(saved_model, config)

    env = gym.make(  # TODO make this singleton?
        "ALE/Pong-v5",
        render_mode="human",  # TODO this is hardcoded again, fix somehow to control with CLI
        frameskip=2,
        repeat_action_probability=0.0,
    )

    init_frame = env.reset()[0]
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

        player_ball_x_dist = get_distance_between_points((ball_x, 0), (player_x, 0))

        print(
            "Player y: {}, Ball y: {}, Player-Ball dist: {}".format(
                player_y, ball_y, round(player_ball_x_dist, 2)
            )
        )
        outputs = net.activate((player_y, ball_y, round(player_ball_x_dist, 2)))

        ai_move = (
            np.argmax(outputs) + 1
        )  # function returns 0, 1, 2 controlls are 1, 2, 3 for stay, up and down

        # Take action and prepare frame for next loop iteration
        frame, reward, done, _, info = env.step(ai_move)
        target_frame = preprocess_frame(frame)

        env.render()


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

    if mode == "train":
        train_ai(config, filepath)
    elif mode == "test":
        test_ai(config, filepath)
    elif mode == "visualize":
        # Load model from pickle
        with open(filepath, "rb") as f:
            genome = pickle.load(f)
            draw_net(config, genome, True)
    else:
        raise Exception("No such mode as {} in neat-pong".format(mode))


if __name__ == "__main__":
    pass
