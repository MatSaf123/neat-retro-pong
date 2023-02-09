# Code snippet taken from https://www.gymlibrary.dev/content/basic_usage/

import gym
from gym.utils.play import play

env = gym.make("PongNoFrameskip-v4", render_mode="rgb_array")
env.metadata["render_fps"] = 60

key_mappings = {"0": 0, "1": 1, "2": 2, "3": 3}

play(env, keys_to_action=key_mappings)
