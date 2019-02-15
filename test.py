import gym
from gym_dobot.envs import DobotPickAndPlaceEnv, DobotPushEnv, DobotReachEnv


env = gym.make('DobotHRLClear-v1')
#env = gym.make('FetchPickAndPlace-v1')
env.reset()

for i in range(1000):
	env.render()
	env.step(env.action_space.sample())

