import gym
from gym_dobot.envs import DobotHRLMazeEnv

env = gym.make('DobotHRLMaze-v1')
env.reset()

for i in range(1000):
    env.step(env.action_space.sample())
    env.render()
