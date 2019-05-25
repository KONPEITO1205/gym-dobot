import gym
from gym_dobot.envs import DobotPickAndPlaceEnv, DobotPushEnv, DobotReachEnv, DobotHRLPushEnv, DobotHRLMazeEnv, DobotHRLClearEnv, DobotHRLPickEnv
import cv2
import numpy as np

#env = gym.make('DobotHRLPendulum-v1')
env = gym.make('FetchPickAndPlace-v1')


while True:
	env.reset()
	for i in range(150):
		env.render()
		# img = env.env.capture(depth=False)
		obs, _, _, _ = env.step(env.action_space.sample())
		#img = obs['observation'][9:].reshape((50,50,3))
		# cv2.imwrite('./images/test'+str(i)+'.png',img)
		# cv2.imwrite('./images/test'+str(i)+'.png',cv2.cvtColor(img, cv2.COLOR_RGB2BGR)[:,:])

