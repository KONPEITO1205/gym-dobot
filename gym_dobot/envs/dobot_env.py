import numpy as np
from gym_dobot.envs import rotations, robot_env, utils,mjremote
from mujoco_py.generated import const


def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)


class DobotEnv(robot_env.RobotEnv):
    """Superclass for all Dobot environments.
    """

    def __init__(
        self, model_path, n_substeps, gripper_extra_height, block_gripper,
        has_object, target_in_the_air, target_offset, obj_range, target_range,
        distance_threshold, initial_qpos, reward_type,rand_dom,unity_remote,
    ):
        """Initializes a new Dobot environment.

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
            reward_type ('sparse' or 'dense'): the reward type, i.e. sparse or dense
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


        super(DobotEnv, self).__init__(
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

    # def compute_reward(self, achieved_goal, goal, info):
    #     # Compute distance between goal and the achieved goal.
    #     d = goal_distance(achieved_goal, goal)
    #     if self.reward_type == 'sparse':
    #         return -(d > self.distance_threshold).astype(np.float32)
    #     else:
    #         return -d

    def compute_reward(self, achieved_goal, goal, info, obs, params):
        # Compute distance between goal and the achieved goal.
        ret = 0
        d = goal_distance(achieved_goal, goal)
        if self.reward_type == 'sparse':
            ret = -(d > self.distance_threshold).astype(np.float32)
            if self.unity_remote:
                self.remote.settargetstatus(int(ret))
        else:
            ret = -d
        clutterNumber = 0
        # clutterPos = []
        # if params['clutter_reward'] == 1:
        #     # List of positions of clutter boxes
        #     object0Pos = np.array(obs[3:6])
        #     for i in range(params['clutter_num']):
        #         clutterPos.append(np.array(obs[3*i+25:3*i+28]))
        #     for i in range(params['clutter_num']):
        #         if np.linalg.norm(object0Pos[:2]-clutterPos[i][:2]) < 0.050:
        #             clutterNumber += 1
        # print(clutterPos,'clutter')
        return ret-clutterNumber

    # RobotEnv methods
    # ----------------------------

    def _step_callback(self):
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
        print(gripper_ctrl)
        #pos_ctrl_low = [-0.05,-0.05,-0.05]
        #pos_ctrl_high = [0.05,0.05,0.05]
        #pos_ctrl = np.clip(pos_ctrl,pos_ctrl_low,pos_ctrl_high)
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

    def _get_obs(self):
        # positions
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
        obs = np.concatenate([
            grip_pos, object_pos.ravel(), object_rel_pos.ravel(), gripper_state, object_rot.ravel(),
            object_velp.ravel(), object_velr.ravel(), grip_velp, gripper_vel,
        ])

        return {
            'observation': obs.copy(),
            'achieved_goal': achieved_goal.copy(),
            'desired_goal': self.goal.copy(),
        }

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
        # self.viewer.cam.type = 2
        self.viewer._hide_overlay = True

    def _render_callback(self):
        # Visualize target.
        sites_offset = (self.sim.data.site_xpos - self.sim.model.site_pos).copy()
        site_id = self.sim.model.site_name2id('target0')
        self.sim.model.site_pos[site_id] = self.goal - sites_offset[0]
        self.sim.forward()

    def _reset_sim(self):
        if self.unity_remote:
            if not self.startup:
                save = self.remote.getsavestatus()
                if save[0]==1:
                    fname = datetime.datetime.now().strftime("Demo_%d%b_%H-%M-%S.npz")
                    dirname, filename = os.path.split(os.path.abspath(__file__))
                    path = os.path.join(dirname,'Demos',fname)
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
        # if self.has_object:
        #     object_xpos = self.initial_gripper_xpos[:2]
        #     #while np.linalg.norm(object_xpos - self.initial_gripper_xpos[:2]) < 0.01:
        #     object_xpos = self.initial_gripper_xpos[:2] + self.np_random.uniform(-self.obj_range, self.obj_range, size=2)
        #     #object_xpos[:2] = np.clip(object_xpos,[0.6,0.55],[1.0,0.95,0.47])
        #     object_qpos = self.sim.data.get_joint_qpos('object0:joint')
        #     assert object_qpos.shape == (7,)
        #     object_qpos[:2] = object_xpos
        #     self.sim.data.set_joint_qpos('object0:joint', object_qpos)

        if self.has_object:
            pos = np.array([0.8,0.685])
            size = np.array([0.280,0.100]) - 0.02
            up = pos + size
            low = pos - size
            object_xpos = np.array([self.np_random.uniform(low[0],up[0]),self.np_random.uniform(low[1],up[1])])
            object_qpos = self.sim.data.get_joint_qpos('object0:joint')
            assert object_qpos.shape == (7,)
            object_qpos[:2] = object_xpos
            object_qpos[2] = 0.032
            self.sim.data.set_joint_qpos('object0:joint', object_qpos)
            

        self.sim.forward()
        if self.unity_remote:
            self.sent_target=False
        return True

    def _sample_goal(self):
        # if self.has_object:
        #     goal = self.initial_gripper_xpos[:3] + self.np_random.uniform(-self.target_range, self.target_range, size=3)
        #     goal += self.target_offset
        #     goal[2] = self.height_offset
        #     if self.target_in_the_air and self.np_random.uniform() < 0.5:
        #         goal[2] += self.np_random.uniform(0, 0.25)
        # else:
        #     goal = self.initial_gripper_xpos[:3] + self.np_random.uniform(-0.15, 0.15, size=3)
        # return goal.copy()

        pos = np.array([0.8,0.685])
        size = np.array([0.280,0.100]) - 0.02
        up = pos + size
        low = pos - size
        goal = np.array([self.np_random.uniform(low[0],up[0]),self.np_random.uniform(low[1],up[1]),0.028])

        if self.has_object:
            if self.target_in_the_air and self.np_random.uniform() < 1.0:
                goal[2] = 0.148#self.np_random.uniform(0.028, 0.148)
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
        gripper_target = np.array([0.8,1,0.37])
        gripper_rotation = np.array([-1, 0., 0., 0])
        self.sim.data.set_mocap_pos('dobot:mocap', gripper_target)
        self.sim.data.set_mocap_quat('dobot:mocap', gripper_rotation)
        for _ in range(10):
            self.sim.step()

        # Extract information for sampling goals.
        #self.initial_gripper_xpos = self.sim.data.get_site_xpos('dobot:grip').copy()
        self.initial_gripper_xpos = np.array([0.8, 0.685, 0.2975])
        #print(self.initial_gripper_xpos)
        if self.has_object:
            self.height_offset = self.sim.data.get_site_xpos('site:object0')[2]

    def capture(self,depth=False):
        if self.viewer == None:
            pass
        else:
            self.viewer.cam.fixedcamid = 0
            self.viewer.cam.type = const.CAMERA_FIXED
            width, height = 1920, 1080
            img = self._get_viewer().read_pixels(width, height, depth=depth)
            # print(img[:].shape)
            # # depth_image = img[:][:][1][::-1] # To visualize the depth image(depth=True)
            # # rgb_image = img[:][:][0][::-1] # To visualize the depth image(depth=True)
            # depth_image = img[:][:][1][::-1]
            # rgb_image = img[:][:][0][::-1]
            if depth:
                rgb_image = img[0][::-1]
                depth_image = np.expand_dims(img[1][::-1],axis=2)
                rgbd_image = np.concatenate((rgb_image,depth_image),axis=2)
                return rgbd_image
            else:
                return img[::-1]


    def set_object(self,pos):
        if self.has_object:
            assert len(pos)==3
            object_xpos = pos
            object_qpos = self.sim.data.get_joint_qpos('object0:joint')
            assert object_qpos.shape == (7,)
            object_qpos[:3] = object_xpos
            object_qpos[2] += 0.032
            self.sim.data.set_joint_qpos('object0:joint', object_qpos)
    
    def set_goal(self,pos):
        assert len(pos)==3
        self.goal = np.array(pos)

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



