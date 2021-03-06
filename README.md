# gym-dobot
Open AI Gym Environment for the [Dobot Magician Robotic Arm](https://www.dobot.cc/dobot-magician/product-overview.html).
Based on the [fetch](https://gym.openai.com/envs/#robotics) environments provided by gym.

Currently consists of -
 - DobotPickAndPlaceEnv
 - DobotPushEnv
 - DobotReachEnv
 - DobotClutterPickAndPlaceEnv
 - DobotClutterPushEnv

The ClutterEnv variations add additonal blocks as clutter. The number of blocks can be changed by setting ```clutter_num``` while creating the environment.(Max=40,Default=20)
 
## Dependencies
  - python 3 (Tested on Python 3.6)
  - gym > 0.10.3
  - mujoco_py > 1.5
  - mujoco - mjpro150 


## Basic Installation
```bash
git clone https://github.com/WarrG3X/gym-dobot
cd gym-dobot
pip install -e .
```
### Usage
```python
from gym_dobot.envs import DobotPickAndPlaceEnv
env = DobotPickAndPlaceEnv()
```

## Gym Installation
First follow the steps for Basic Installation.

To access these environments from within your existing gym installation, they must first be registered. 

If you have installed gym from their [git repository](https://github.com/openai/gym#installation) then you can directly register the environments by simply applying the patch provided in this repository.
```bash
# Go to gym repository
cd gym
git apply /path/to/repo/gym-dobot/gym_install.patch
```
Otherwise, follow the instructions stated [here](https://github.com/openai/gym/tree/master/gym/envs#how-to-add-new-environments-to-gym-within-this-repo-not-recommended-for-new-environments).

### Usage
```python
import gym
env = gym.make("DobotPickAndPlaceEnv-v1")
```
This is much more convenient as other packages such as [openai/baselines](https://github.com/openai/baselines) can be directly used with these dobot environments.

## Test
```bash
# Basic Test
python -m gym_dobot.run_env

#To see additional options
python -m gym_dobot.run_env --help

#Advanced Usage Example
python -m gym_dobot.run_env --env=DobotClutterPickAndPlaceEnv --steps=200 --clutter=40 
```
