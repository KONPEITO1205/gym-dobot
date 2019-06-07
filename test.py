import gym
from gym_dobot.envs import DobotPickAndPlaceEnv, DobotPushEnv, DobotReachEnv, DobotHRLPushEnv, DobotHRLMazeEnv, DobotHRLClearEnv, DobotHRLPickEnv
import cv2
import numpy as np

#env = gym.make('DobotHRLPendulum-v1')
env = gym.make('FetchReach-v1')
#env = gym.make('CartPole-v0')


while True:
	env.reset()
	temp = 1
	import time
	time.sleep(1)
	for i in range(90):
		import time
		if i%9 == 0:
			time.sleep(.5)
			# temp = -temp
		env.render()
		# img = env.env.capture(depth=False)
		xx = env.action_space.sample()
		xx[:2] = temp * 0.1
		obs, _, _, _ = env.step(xx)
		# img = obs['observation'][9:].reshape((50,50,3))
		# cv2.imwrite('./images/test'+str(i)+'.png',img)
	#	cv2.imwrite('./images/test'+str(i)+'.png',cv2.cvtColor(img, cv2.COLOR_RGB2BGR)[:,:])

