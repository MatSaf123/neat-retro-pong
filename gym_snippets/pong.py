# Code snippet taken from https://www.gymlibrary.dev/content/basic_usage/

import gym
from gym.utils.play import play

env = gym.make("Pong-v4", render_mode="rgb_array")
env.metadata["render_fps"] = 15

key_mappings = {"w": 2, "s": 3}

play(env, keys_to_action=key_mappings)
