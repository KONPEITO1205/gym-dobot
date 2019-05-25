import gym
import cv2
from PIL import Image
from gym_dobot.envs import DobotPickAndPlaceEnv, DobotPushEnv, DobotReachEnv, DobotHRLPushEnv, DobotHRLMazeEnv, DobotHRLClearEnv, DobotHRLPickEnv 

env = gym.make('DobotHRLPush-v1')
env.reset()
#for i in range(20):
env.render()
img = env.env.capture()#render(mode='rgb_array')
print(img)
j = Image.fromarray(img)
j.save('img_new1.png')
# env.close()

