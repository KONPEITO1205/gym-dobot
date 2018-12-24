import gym
import click
import gym_dobot.envs as envs
from scipy.misc import imsave


@click.command()
@click.option('--env', default="DobotClutterPickAndPlaceEnv", help='Which environment to run (Eg. - DobotReachEnv)')
@click.option('--render',default=1,help='Whether to render the environment')
@click.option('--steps',default=150,help='Number of timesteps to run the environment each time')
@click.option('--clutter',default=40,help='Number of clutter objects for clutter environments')
@click.option('--rand_dom',default=0,help='Whether to use domain randomization')
@click.option('--unity_remote',default=0,help='Whether to operate in remote rendering mode')
def main(env,render,steps,clutter,rand_dom,unity_remote):
    if 'Clutter' in env:
        env = getattr(envs,env)(clutter_num=clutter,rand_dom=rand_dom,unity_remote=unity_remote)
    else:
        env = getattr(envs,env)(rand_dom=rand_dom,unity_remote=unity_remote)

    while True:
        observation = env.reset()
        for i in range(steps):
            if render:
                env.render()
                # img = env.capture(depth=True)
                # imsave("image.png",img)
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)

if __name__=='__main__':
    main()
