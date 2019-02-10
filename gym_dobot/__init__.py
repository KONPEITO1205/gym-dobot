from gym.envs.registration import register

for reward_type in ['sparse']:
    suffix = 'Dense' if reward_type == 'dense' else ''
    kwargs = {
        'reward_type': reward_type,
    }


register(
    id='DobotReach-v1'.format(suffix),
    entry_point='gym_dobot.envs:DobotReachEnv',
    kwargs=kwargs,
    max_episode_steps=50,
)

register(
    id='DobotPush-v1'.format(suffix),
    entry_point='gym_dobot.envs:DobotPushEnv',
    kwargs=kwargs,
    max_episode_steps=50,
)

register(
    id='DobotPickAndPlace-v1'.format(suffix),
    entry_point='gym_dobot.envs:DobotPickAndPlaceEnv',
    kwargs=kwargs,
    max_episode_steps=50,
)

register(
    id='DobotClutterPickAndPlace-v1'.format(suffix),
    entry_point='gym_dobot.envs:DobotClutterPickAndPlaceEnv',
    kwargs=kwargs,
    max_episode_steps=50,
)

register(
    id='DobotClutterPush-v1'.format(suffix),
    entry_point='gym_dobot.envs:DobotClutterPushEnv',
    kwargs=kwargs,
    max_episode_steps=50,
)

register(
    id='DobotHRLPush-v1'.format(suffix),
    entry_point='gym_dobot.envs:DobotHRLPushEnv',
    kwargs=kwargs,
    max_episode_steps=50,
)