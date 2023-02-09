# Code snippet taken from https://www.gymlibrary.dev/content/basic_usage/

import gym

# Set render_mode to "human" in order to display pygame window
env = gym.make("LunarLander-v2", render_mode="human")


env.action_space.seed(42)
observation, info = env.reset(seed=42)

# Run this environment for 1000 timesteps
for _ in range(1000):
    observation, reward, terminated, truncated, info = env.step(
        # Make AI do random actions with sample()
        env.action_space.sample()
    )

    if terminated or truncated:
        observation, info = env.reset()

env.close()
