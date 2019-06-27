import numpy as np
from gym_dobot.envs import rotations, robot_env, utils,mjremote
from mujoco_py.generated import const
import datetime
import sys
import os
import cv2

from shapely.geometry import Polygon, Point, MultiPoint

try:
    import mujoco_py
    from mujoco_py.modder import TextureModder,CameraModder,LightModder
except ImportError as e:
    raise error.DependencyNotInstalled("{}. (HINT: you need to install mujoco_py, and also perform the setup instructions here: https://github.com/openai/mujoco-py/.)".format(e))



def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)


class DobotHRLEnv(robot_env.RobotEnv):
    """Superclass for all DobotHRL environments.
    """

    def __init__(
        self, model_path, n_substeps, gripper_extra_height, block_gripper,
        has_object, target_in_the_air, target_offset, obj_range, target_range,
        distance_threshold, initial_qpos, reward_type,rand_dom,unity_remote, randomize=False,
    ):
        """Initializes a new DobotHRL environment.

        Args:
            model_path (string): path to the environments XML file
            n_substeps (int): number of substeps the simulation runs on every call to step
            gripper_extra_height (float): additional height above the table when positioning the gripper
            block_gripper (boolean): whether or not the gripper is blocked (i.e. not movable) or not
            has_object (boolean): whether or not the environment has an object
            target_in_the_air (boolean): whether or not the target should be in the air above the table or on the table surface
            target_offset (float or array with 3 elements): offset of the target
            obj_range (float): range of a uniform distribution for sampling initial object positions
            target_range (float): range of a uniform distribution for sampling a target
            distance_threshold (float): the threshold after which a goal is considered achieved
            initial_qpos (dict): a dictionary of joint names and values that define the initial configuration
            reward_type ('sparse' or 'dense'): the reward type, i.e. sparse or 
            rand_dom ('False' or 'True'): Whether to use domain randomization
        """
        self.gripper_extra_height = gripper_extra_height
        self.block_gripper = block_gripper
        self.has_object = has_object
        self.target_in_the_air = target_in_the_air
        self.target_offset = target_offset
        self.obj_range = obj_range
        self.target_range = target_range
        self.distance_threshold = distance_threshold
        self.reward_type = reward_type
        self.rand_dom = rand_dom
        self.sent_target = False
        self.unity_remote = unity_remote
        self.randomize = randomize
        self.model_path = model_path
        self.count = 0

        # if "Pick" in self.__class__.__name__ or "Clear" in self.__class__.__name__:
        #     self.polygons = [Polygon([(0.604,0.653),(0.604,0.717),(0.768,0.717),(0.768,0.800),
        #         (0.832,0.800),(0.832,0.717),(0.996,0.717),(0.996,0.653),
        #         (0.832,0.653),(0.832,0.572),(0.768,0.572),(0.768,0.653)])]
        # elif "Maze" in self.__class__.__name__:
        #     self.polygons = [Polygon([(0.699,0.803),(0.763,0.803),(0.763,0.627),(0.699,0.627)]),
        #                     Polygon([(0.836,0.567),(0.900,0.567),(0.900,0.743),(0.836,0.743)])]
        # else:
        #     self.polygons = None
        self.polygons = None


        super(DobotHRLEnv, self).__init__(
            model_path=model_path, n_substeps=n_substeps, n_actions=4,
            initial_qpos=initial_qpos)

        if self.unity_remote:
            self.remote = mjremote.mjremote()
            connection_data = self.remote.connect()
            self.nqpos = connection_data[0]
            print('Connected: ', connection_data)
            assert len(self.sim.data.qpos) == self.nqpos, "Remote Renderer and Mujoco Simulation Doesn't Match"
            self.remote.setqpos(self.sim.data.qpos)
            self.startup = True

    # GoalEnv methods
    # ----------------------------

    def compute_reward(self, achieved_goal, goal, info):#, obs=None, params=None):
        # Compute distance between goal and the achieved goal.
        ret = 0
        d = goal_distance(achieved_goal, goal)
        if self.reward_type == 'sparse':
            ret = -(d > self.distance_threshold).astype(np.float32)
            if self.unity_remote:
                self.remote.settargetstatus(int(ret))
        else:
            ret = -d
        return ret

    def set_subgoal(self, name, action):
        site_id = self.sim.model.site_name2id(name)
        sites_offset = (self.sim.data.site_xpos - self.sim.model.site_pos).copy()
        self.sim.model.site_pos[site_id] = action - sites_offset[0]
        self.sim.model.site_rgba[site_id][3] = 1

    # RobotEnv methods
    # ----------------------------

    def _step_callback(self):
        if "Clear" in self.__class__.__name__ and self.if_collision():
            self.collision = True
        if self.block_gripper:
            self.sim.data.set_joint_qpos('dobot:l_gripper_joint', 0.)
            self.sim.data.set_joint_qpos('dobot:r_gripper_joint', 0.)
            self.sim.forward()
        if self.unity_remote:
            if not self.sent_target:
                self.remote.settarget(self.goal)
                self.sent_target = True
            self.remote.setqpos(self.sim.data.qpos)
            self.remote.setmocap(self.sim.data.mocap_pos[0],self.sim.data.mocap_quat[0])

    def _set_action(self, action):
        if self.unity_remote:
            ovr_data = self.remote.getovrinput()
            action = np.array([ovr_data[1],ovr_data[2],ovr_data[3],ovr_data[0]])
        self.episodeAcs.append(action)
        assert action.shape == (4,)
        action = action.copy()  # ensure that we don't change the action outside of this scope
        pos_ctrl, gripper_ctrl = action[:3], action[3]
        pos_ctrl *= 0.05 # limit maximum change in position
        rot_ctrl = [-1, 0, 0, 0]  # fixed rotation of the end effector, expressed as a quaternion
        gripper_ctrl = np.array([gripper_ctrl, -gripper_ctrl])
        assert gripper_ctrl.shape == (2,)
        if self.block_gripper:
            gripper_ctrl = np.zeros_like(gripper_ctrl)
        action = np.concatenate([pos_ctrl, rot_ctrl, gripper_ctrl])

        # Apply action to simulation.
        utils.ctrl_set_action(self.sim, action)
        utils.mocap_set_action(self.sim, action)

    # def _get_obs(self):
    #     # positions
    #     grip_pos = self.sim.data.get_site_xpos('dobot:grip')
    #     dt = self.sim.nsubsteps * self.sim.model.opt.timestep
    #     grip_velp = self.sim.data.get_site_xvelp('dobot:grip') * dt
    #     robot_qpos, robot_qvel = utils.robot_get_obs(self.sim)
    #     if self.has_object:
    #         object_pos = self.sim.data.get_site_xpos('site:object0')
    #         # rotations
    #         object_rot = rotations.mat2euler(self.sim.data.get_site_xmat('site:object0'))
    #         # velocities
    #         object_velp = self.sim.data.get_site_xvelp('site:object0') * dt
    #         object_velr = self.sim.data.get_site_xvelr('site:object0') * dt
    #         # gripper state
    #         object_rel_pos = object_pos - grip_pos
    #         object_velp -= grip_velp
    #     else:
    #         object_pos = object_rot = object_velp = object_velr = object_rel_pos = np.zeros(0)
    #     gripper_state = robot_qpos[-2:]
    #     gripper_vel = robot_qvel[-2:] * dt  # change to a scalar if the gripper is made symmetric

    #     if not self.has_object:
    #         achieved_goal = grip_pos.copy()
    #     else:
    #         achieved_goal = np.squeeze(object_pos.copy())
    #     obs = np.concatenate([
    #         grip_pos, object_pos.ravel(), object_rel_pos.ravel(), gripper_state, object_rot.ravel(),
    #         object_velp.ravel(), object_velr.ravel(), grip_velp, gripper_vel,
    #     ])

    #     return {
    #         'observation': obs.copy(),
    #         'achieved_goal': achieved_goal.copy(),
    #         'desired_goal': self.goal.copy(),
    #     }


    def _get_obs(self):
        grip_pos = self.sim.data.get_site_xpos('dobot:grip')
        dt = self.sim.nsubsteps * self.sim.model.opt.timestep
        grip_velp = self.sim.data.get_site_xvelp('dobot:grip') * dt
        robot_qpos, robot_qvel = utils.robot_get_obs(self.sim)
        if self.has_object:
            object_pos = self.sim.data.get_site_xpos('site:object0')
            # rotations
            object_rot = rotations.mat2euler(self.sim.data.get_site_xmat('site:object0'))
            # velocities
            object_velp = self.sim.data.get_site_xvelp('site:object0') * dt
            object_velr = self.sim.data.get_site_xvelr('site:object0') * dt
            # gripper state
            object_rel_pos = object_pos - grip_pos
            object_velp -= grip_velp
        else:
            object_pos = object_rot = object_velp = object_velr = object_rel_pos = np.zeros(0)
        gripper_state = robot_qpos[-2:]
        gripper_vel = robot_qvel[-2:] * dt  # change to a scalar if the gripper is made symmetric

        if not self.has_object:
            achieved_goal = grip_pos.copy()
        else:
            achieved_goal = np.squeeze(object_pos.copy())

        # self.render()
        # -------------------------------------------------------------------
        # import cv2
        # blackBox = grip_pos
        # redCircle = self.goal
        # blueBox = object_pos
        # height = 273 
        # width = 467
        # img = np.zeros((height,width,3), np.uint8)
        # img[:,:] = (16,163,127)
        # # Updated
        # # x_range = np.random.uniform(0.6, 1.)
        # # y_range = np.random.uniform(0.57, 0.8)
        # half_block_len = int(38/2)
        # gripper_len = 12
        # sphere_rad = 18
        # # Mark the block position
        # mapBlue = self.map_cor(blueBox)
        # xx = int(mapBlue[0])
        # yy = int(mapBlue[1])
        # img[xx-half_block_len:xx+half_block_len,yy-half_block_len:yy+half_block_len] = (84,54,218)

        # # Mark the sphere position
        # mapRed = self.map_cor(redCircle)
        # xx = int(mapRed[0])
        # yy = int(mapRed[1])
        # cv2.circle(img,(yy,xx), 16, (255,0,0), -1)

        # # Mark the gripper position
        # mapBlack = self.map_cor(blackBox)
        # xx = int(mapBlack[0])
        # yy = int(mapBlack[1])
        # img[xx-gripper_len:xx+gripper_len,yy-gripper_len:yy+gripper_len] = (0,0,0)

        # image = cv2.resize(img, (50,50))
        # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)[:,:]
        # -------------------------------------------------------------------
        
        # cv2.imwrite('./images/test'+str(self.count)+'.png',cv2.cvtColor(img, cv2.COLOR_RGB2BGR)[:,:])
        # image = self.capture()
        obs = np.concatenate([
            grip_pos, object_pos.ravel(), object_rel_pos.ravel(), gripper_state, #object_rot.ravel(),
            #object_velp.ravel(), object_velr.ravel(), grip_velp, gripper_vel,
        ])

        # obs = np.concatenate([
        #     grip_pos, object_pos.ravel(), object_rel_pos.ravel(), image.ravel(), #object_rot.ravel(),
        #     # object_velp.ravel(), object_velr.ravel(), grip_velp, gripper_vel,
        # ])

        return {
            'observation': obs.copy(),
            'achieved_goal': achieved_goal.copy(),
            'desired_goal': self.goal.copy(),
        }

    def map_cor(self, pos,X1=0.0,X2=273.0,Y1=0.,Y2=467.0,x1=0.8,x2=0.57,y1=0.6,y2=1.0):
        if pos[0] > 1:
            return np.array([193., 180.])
        y = pos[0]
        x = pos[1]
        X = X1 + ( (x-x1) * (X2-X1) / (x2-x1) )
        Y = Y1 + ( (y-y1) * (Y2-Y1) / (y2-y1) )
        # X=X2+((X1-X2)/x2)*x
        # Y=Y2+((Y1-Y2)/y2)*y
        return(np.array([X,Y]))

    def _viewer_setup(self):
        body_id = self.sim.model.body_name2id('dobot:gripper_link')
        lookat = self.sim.data.body_xpos[body_id]
        for idx, value in enumerate(lookat):
            self.viewer.cam.lookat[idx] = value
        #print(self.viewer.__dict__)

        #print(self.viewer.sim.render())
        self.viewer.cam.distance = 2.2
        self.viewer.cam.azimuth = 145.
        self.viewer.cam.elevation = -25.

        # self.viewer.cam.fixedcamid = 0
        # self.viewer.cam.type = const.CAMERA_FIXED
        # self.viewer.cam.fixedcamid = 0
        # self.viewer.cam.type = 2
        self.viewer._hide_overlay = True

    def _render_callback(self):
        # Visualize target.
        sites_offset = (self.sim.data.site_xpos - self.sim.model.site_pos).copy()
        site_id = self.sim.model.site_name2id('target0')
        self.sim.model.site_pos[site_id] = self.goal - sites_offset[0]
        self.sim.forward()

    def randomize_env(self):
        import xml.etree.cElementTree as ET
        tree = ET.ElementTree(file='/home/vrsystem/gitrep/gym-dobot/gym_dobot/envs/assets/dobot/hrl_maze_temp.xml')
        index = tree.find('worldbody')
        sizeLeft = np.random.randint(6)
        sizeRight = np.random.randint(6)

        angleLeft = 0
        angleRight = 0

        if np.random.randint(2) == 1:
            angleLeft = 1.57

        if np.random.randint(2) == 1:
            angleRight = 1.57

        x1 = np.random.uniform(0.615, 0.98)
        x1_pi = np.random.uniform(0.615,0.854)
        y1 = np.random.uniform(0.718, 0.787)
        y1_pi = np.random.uniform(0.587,0.787)

        HEIGHT = 0.025

        for child in index:
            if child.tag == 'body' and child.attrib['name'] == 'leftwall':
                if angleLeft == 0:
                    child.attrib['pos'] = "{} {} {}".format(x1,y1,HEIGHT)
                else:
                    child.attrib['pos'] = "{} {} {}".format(x1_pi,y1_pi,HEIGHT)
                child.attrib['euler'] = "{} {} {}".format(0,0,angleLeft)
                for key in range(5 - sizeLeft):
                    del child[len(child)-1]
            elif child.tag == 'body' and child.attrib['name'] == 'rightwall':
                if angleRight == 0:
                    child.attrib['pos'] = "{} {} {}".format(x1,y1,HEIGHT)
                else:
                    child.attrib['pos'] = "{} {} {}".format(x1_pi,y1_pi,HEIGHT)
                child.attrib['euler'] = "{} {} {}".format(0,0,angleRight)
                for key in range(5 - sizeRight):
                    del child[len(child)-1]

        tree.write('/home/vrsystem/gitrep/gym-dobot/gym_dobot/envs/assets/dobot/hrl_maze.xml')

    def randomize_environ(self):
        # print(self.sim.data)#.get_joint_qpos('object0:joint')
        # id1 = self.sim.model.body_name2id('rightwall')
        # print(self.sim.data.body_xpos[id1])
        # # self.sim.data.body_xpos[id1][1] = 1
        # print(self.sim.data.qpos)
        x1_left = 0
        x1_right = 0
        y1_left = 0
        y1_right = 0
        while np.abs(x1_left - x1_right) < 0.04:
            x1_left = np.random.uniform(0.615, 0.98)
            y1_left = np.random.uniform(0.65, 0.72)
            x1_right = np.random.uniform(0.615, 0.98)
            y1_right = np.random.uniform(0.65, 0.72)
        self.goal = [x1_left, y1_left, 0.148]

        # x_width = 0.04
        # y_width = 0.04

        self.polygons = [Polygon([(x1_left,y1_left),(x1_left+0.04,y1_left),(x1_left,y1_left+0.04),(x1_left+0.04,y1_left+0.04)]),
                        Polygon([(x1_right,y1_right),(x1_right+0.04,y1_right),(x1_right,y1_right+0.04),(x1_right+0.04,y1_right+0.04)])]

        object_qpos_left = self.sim.data.get_joint_qpos('leftwall')
        object_qpos_left[0] = x1_left
        object_qpos_left[1] = y1_left

        object_qpos_right = self.sim.data.get_joint_qpos('rightwall')
        object_qpos_right[0] = x1_right
        object_qpos_right[1] = y1_right
        self.sim.data.set_joint_qpos('leftwall', object_qpos_left)
        self.sim.data.set_joint_qpos('rightwall', object_qpos_right)



    def _reset_sim(self):
        if self.unity_remote:
            if not self.startup:
                save = self.remote.getsavestatus()
                if save[0]==1:
                    fname = datetime.datetime.now().strftime("Demo_%d%b_%H-%M-%S.npz")
                    dirname, filename = os.path.split(os.path.abspath(__file__))
                    dirpath = os.path.join(dirname,'Demos')
                    path = os.path.join(dirpath,fname)
                    if not os.path.exists(dirpath):
                        try:
                            os.makedirs(dirpath)
                            print("Directory Demos created.")
                        except:
                            print("Failed to create directory. Please create one manually.")
                    try:
                        np.savez_compressed(path, epacs=self.episodeAcs, epobs=self.episodeObs, epinfo=self.episodeInfo)
                    except:
                        sys.exit('ERROR: Could not save demo')
                    print("Saved "+fname)
                elif save[0]==-1:
                    sys.exit('Terminated from Renderer')
            self.startup = False
        self.episodeAcs = []
        self.episodeObs = []
        self.episodeInfo = []
        self.sim.set_state(self.initial_state)
        if "Clear" in self.__class__.__name__:
            self.clutter()
        if self.viewer!= None and self.rand_dom: 
            for name in self.sim.model.geom_names:
                self.modder.rand_all(name)

            #Camera
            pos = np.array([0,-1,1]) + self.np_random.uniform(-0.1,0.1,size=3)
            self.cam_modder.set_pos('camera0',pos)

            #Light
            self.light_modder.set_castshadow('light0',1)
            pos = np.array([0.8,0.9,3])
            pos[:2] = pos[:2] + self.np_random.uniform(-0.85,0.85,size=2)
            self.light_modder.set_pos('light0',pos)

        # Randomize start position of object.
        if self.has_object:
            valid = False
            while not valid:
                low = np.array([0.62,0.586])
                up = np.array([0.98,0.784])
                object_xpos = np.array([self.np_random.uniform(low[0],up[0]),self.np_random.uniform(low[1],up[1])])
                object_qpos = self.sim.data.get_joint_qpos('object0:joint')
                assert object_qpos.shape == (7,)
                object_qpos[:2] = object_xpos
                object_qpos[2] = 0.025
                point = Point(object_xpos)
                valid = True
                if self.polygons:
                    for poly in self.polygons:
                        if poly.contains(point):
                            print("Retrying")
                            valid = False
                            break
            self.sim.data.set_joint_qpos('object0:joint', object_qpos)

        self.sim.forward()
        if self.unity_remote:
            self.sent_target=False

        if self.randomize:
            # Randomize the walls to make it a dynamic environment
            self.randomize_environ()
            # if self.model_path.startswith('/'):
            #     fullpath = self.model_path
            # else:
            #     fullpath = os.path.join(os.path.dirname(__file__), 'assets', self.model_path)
            # if not os.path.exists(fullpath):
            #     raise IOError('File {} does not exist'.format(fullpath))

            # model = mujoco_py.load_model_from_path(fullpath)
            # self.sim = mujoco_py.MjSim(model, nsubsteps=20)

        return True

    def clutter(self):
        x = 0.8
        z = 0.025
        y = 0.784
        dy = [0,-0.033,-0.066,-0.099,-0.132,-0.165,-0.198]
        for i in range(1,8):
            object_name = "object{}:joint".format(i)
            object_xpos = np.array([x,y+dy[i-1],z])
            object_qpos = self.sim.data.get_joint_qpos(object_name)
            assert object_qpos.shape == (7,)
            object_qpos[:3] = object_xpos
            self.sim.data.set_joint_qpos(object_name, object_qpos)
        
        z = 0.025
        y = 0.685
        x = 0.62
        x2 = 0.98
        dx = [0,0.033,0.066,0.099,0.132]
        for i in range(8,13):
            object_name = "object{}:joint".format(i)
            object_xpos = np.array([x+dx[i-8],y,z])
            object_qpos = self.sim.data.get_joint_qpos(object_name)
            assert object_qpos.shape == (7,)
            object_qpos[:3] = object_xpos
            self.sim.data.set_joint_qpos(object_name, object_qpos)

            object_name = "object{}:joint".format(i+5)
            object_xpos = np.array([x2-dx[i-8],y,z])
            object_qpos = self.sim.data.get_joint_qpos(object_name)
            assert object_qpos.shape == (7,)
            object_qpos[:3] = object_xpos
            self.sim.data.set_joint_qpos(object_name, object_qpos)



    def _sample_goal(self):
        valid = False
        while not valid:
            low = np.array([0.62,0.586])
            up = np.array([0.98,0.784])
            goal = np.array([self.np_random.uniform(low[0],up[0]),self.np_random.uniform(low[1],up[1]),0.025])
            point = Point(goal)
            valid = True
            if self.polygons:
                for poly in self.polygons:
                    if poly.contains(point):
                        # print("Retrying Goal")
                        valid = False
                        break


        if self.has_object:
            if self.target_in_the_air and self.np_random.uniform() < 1.0:
                goal[2] = 0.148#self.np_random.uniform(0.028, 0.148)
            goal[:2] = [0.75, 0.68]
            '''
            goal_ranges:
            low = np.array([0.62,0.586])
            up = np.array([0.98,0.784])

            x:[0.62, 0.98], diff=(0.18), offset=(0.8)
            y:[0.586, 0.784], diff=(0.1), offset=(0.685)
            z:[0.025, 0.148],  diff=(0.615), offset=(0.64)

            self.max_u = [0.18, 0.1, 0]
            self.action_offset = [0.8, 0.685, 0.025]
            u = self.action_offset + (self.max_u * u )

            '''
        else:
            goal[2] = self.np_random.uniform(0.028, 0.148)

        return goal.copy()

    def _is_success(self, achieved_goal, desired_goal):
        d = goal_distance(achieved_goal, desired_goal)
        return (d < self.distance_threshold).astype(np.float32)

    def _env_setup(self, initial_qpos):
        for name, value in initial_qpos.items():
            self.sim.data.set_joint_qpos(name, value)
        utils.reset_mocap_welds(self.sim)
        self.sim.forward()

        # Move end effector into position.
        #gripper_target = np.array([0.001, -1.4, -0.431 + self.gripper_extra_height]) + self.sim.data.get_site_xpos('dobot:grip')
        gripper_target = np.array([0.8,1,-0.37])
        gripper_rotation = np.array([-1, 0., 0., 0])
        self.sim.data.set_mocap_pos('dobot:mocap', gripper_target)
        self.sim.data.set_mocap_quat('dobot:mocap', gripper_rotation)
        for _ in range(10):
            self.sim.step()

        # Extract information for sampling goals.
        #self.initial_gripper_xpos = self.sim.data.get_site_xpos('dobot:grip').copy()
        self.initial_gripper_xpos = np.array([0.8, 0.75, 0.2975])
        #print(self.initial_gripper_xpos)
        if self.has_object:
            self.height_offset = self.sim.data.get_site_xpos('site:object0')[2]


    def capture(self,depth=False):
        if self.viewer == None:
            pass
        else:
            # self.viewer.cam.fixedcamid = 0
            # self.viewer.cam.type = const.CAMERA_FIXED
            width, height = 1920, 1080
            img = self._get_viewer().read_pixels(width, height, depth=depth)
            if not depth:
                img = img[350:623,726:1193]
                # img = cv2.resize(img, (500,500))
            # print(img[:].shape)
            # # depth_image = img[:][:][1][::-1] # To visualize the depth image(depth=True)
            # # rgb_image = img[:][:][0][::-1] # To visualize the depth image(depth=True)
            # depth_image = img[:][:][1][::-1]
            # rgb_image = img[:][:][0][::-1]
            if depth:
                rgb_image = img[0][::-1]
                rgb_image = rgb_image[458:730,726:1193]
                rgb_image = cv2.resize(rgb_image, (50,50))
                depth_image = np.expand_dims(img[1][::-1],axis=2)
                depth_image = depth_image[458:730,726:1193]
                depth_image = cv2.resize(depth_image, (50,50))
                depth_image = np.reshape(depth_image, (50,50,1))
                rgbd_image = np.concatenate((rgb_image,depth_image),axis=2)
                return rgbd_image
            else:
                return img[::-1]

    def real2sim(self,pos):
        assert len(pos)==3
        centre = np.array([0.8,0.685,0])
        pos = np.clip(pos,[170,-150,-30],[290,150,30])
        pos[0] -= 230 #Centre X Value
        pos[0] *= -1 #Invert X value
        pos = pos*0.001 # Convert from mm to m
        pos[0],pos[1] = pos[1],pos[0] #Y and X are swapped in Real
        pos[2] += 0.030 # Centre Z Value
        pos[:] *= 2   #Everything in Sim is 2x Scale
        
        return list(pos + centre)

    def sim2real(self,pos):
        assert len(pos)==3
        centre = np.array([0.8,0.685,0])
        pos = np.clip(pos,[0.5,0.565,0],[1.1,0.805,0.12])
        pos = pos - centre
        pos[:] *= 0.5
        pos[2] -= 0.030
        pos[0],pos[1] = pos[1],pos[0]
        pos  = pos*1000
        pos[0] *= -1
        pos[0] += 230
 
        return list(pos)

    def set_goal(self,pos):
        assert len(pos)==3
        self.goal = np.array(pos)

    def set_object(self,pos,posList):
        object_xpos = pos
        object_qpos = self.sim.data.get_joint_qpos('object0:joint')
        assert object_qpos.shape == (7,)
        object_qpos[:2] = object_xpos[:2]
        object_qpos[2] = 0.032
        # print(object_qpos,'pos')
        self.sim.data.set_joint_qpos('object0:joint', object_qpos)
        temp = 1
        for i in range(len(posList)):
            object_xpos = posList[i][:2]
            object_qpos = self.sim.data.get_joint_qpos('object'+str(i+1)+':joint')
            assert object_qpos.shape == (7,)
            object_qpos[:2] = object_xpos
            object_qpos[2] = 0.032
            self.sim.data.set_joint_qpos('object'+str(i+1)+':joint', object_qpos)
            temp = i+1
        for i in range(temp + 1,21):
            object_qpos = self.sim.data.get_joint_qpos('object'+str(i)+':joint')
            assert object_qpos.shape == (7,)
            object_qpos[:2] = [2,2]
            object_qpos[2] = 0.032

    def if_collision(self):
        for i in range(self.sim.data.ncon):
            contact = self.sim.data.contact[i]
            name1 = self.sim.model.geom_id2name(contact.geom1)
            name2 = self.sim.model.geom_id2name(contact.geom2)
            if name1 is None or name2 is None:
                break
            if "object0" in [name1,name2] and name1[:6]==name2[:6]:
                # print('contact', i)
                # print('geom1', name1[:6])
                # print('geom2', name2[:6])
                return True

        return False