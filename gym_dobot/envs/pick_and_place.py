from gym import utils
from gym_dobot.envs import dobot_env


class DobotPickAndPlaceEnv(dobot_env.DobotEnv, utils.EzPickle):
    def __init__(self, reward_type='sparse',rand_dom=False,unity_remote=False):
        initial_qpos = {
            'dobot:slide0': 0.8,
            'dobot:slide1': 1.2,
            'dobot:slide2': -0.04,
            'object0:joint': [1.25, 0.53, 0.032,  1., 0., 0., 0.],
        }
        dobot_env.DobotEnv.__init__(
            self, 'dobot/pick_and_place.xml', has_object=True, block_gripper=True, n_substeps=20,
            gripper_extra_height=0, target_in_the_air=True, target_offset=.0,
            obj_range=0.13, target_range=0.13, distance_threshold=0.05,
            initial_qpos=initial_qpos, reward_type=reward_type,rand_dom=rand_dom,unity_remote=unity_remote)
        utils.EzPickle.__init__(self)
