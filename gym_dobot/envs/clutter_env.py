import numpy as np
from gym_dobot.envs import rotations, robot_env, utils
from mujoco_py.generated import const



def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)


class DobotClutterEnv(robot_env.RobotEnv):
    """Superclass for all DobotClutter environments.
    """

    def __init__(
        self, model_path, n_substeps, gripper_extra_height, block_gripper,
        has_object, target_in_the_air, target_offset, obj_range, target_range,
        distance_threshold, initial_qpos, reward_type,clutter_num,rand_dom,
    ):
        """Initializes a new DobotClutter environment.

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
            clutter_num (int 0-10) : the number of clutter objects to use
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
        assert clutter_num <= 60
        self.clutter_num = clutter_num
        self.rand_dom = rand_dom


        super(DobotClutterEnv, self).__init__(
            model_path=model_path, n_substeps=n_substeps, n_actions=4,
            initial_qpos=initial_qpos)

    # GoalEnv methods
    # ----------------------------

    def compute_reward(self, achieved_goal, goal, info, obs, params):
        # Compute distance between goal and the achieved goal.
        ret = 0
        d = goal_distance(achieved_goal, goal)
        if self.reward_type == 'sparse':
            ret = -(d > self.distance_threshold).astype(np.float32)
        else:
            ret = -d
        clutterNumber = 0
        # clutterPos = []
        # if params['clutter_reward'] == 1:
        #     # List of positions of clutter boxes
        #     object0Pos = np.array(obs[3:6])
        #     for i in range(params['clutter_num']):
        #         clutterPos.append(np.array(obs[3*i+11:3*i+14]))
        #     for i in range(params['clutter_num']):
        #         if np.linalg.norm(object0Pos[:2]-clutterPos[i][:2]) < 0.050:
        #             clutterNumber += 1
        # # print(clutterPos,'clutter')
        return ret-clutterNumber

    # def compute_reward(self, achieved_goal, goal, info, obs, params):
    #     # Compute distance between goal and the achieved goal.
    #     d = goal_distance(achieved_goal, goal)
    #     if self.reward_type == 'sparse':
    #         return -(d > self.distance_threshold).astype(np.float32)
    #     else:
    #         return -d

    # RobotEnv methods
    # ----------------------------

    def _step_callback(self):
        if self.block_gripper:
            self.sim.data.set_joint_qpos('dobot:l_gripper_joint', 0.)
            self.sim.data.set_joint_qpos('dobot:r_gripper_joint', 0.)
            self.sim.forward()

    def _set_action(self, action):
        assert action.shape == (4,)
        action = action.copy()  # ensure that we don't change the action outside of this scope
        pos_ctrl, gripper_ctrl = action[:3], action[3]
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
        objPosList = []
        if self.has_object:
            for i in range(self.clutter_num + 1):
                if i == 0:
                    object_pos = self.sim.data.get_site_xpos('site:object' + str(i))
                    objPosList.append(object_pos)
                    # rotations
                    object_rot = rotations.mat2euler(self.sim.data.get_site_xmat('site:object' + str(i)))
                    # velocities
                    object_velp = self.sim.data.get_site_xvelp('site:object' + str(i)) * dt
                    object_velr = self.sim.data.get_site_xvelr('site:object' + str(i)) * dt
                    # gripper state
                    object_rel_pos = object_pos - grip_pos
                    object_velp -= grip_velp
                else:
                    object_pos = self.sim.data.get_site_xpos('site:object' + str(i))
                    objPosList.append(object_pos)
        else:
            object_pos = object_rot = object_velp = object_velr = object_rel_pos = np.zeros(0)

        gripper_state = robot_qpos[-2:]
        gripper_vel = robot_qvel[-2:] * dt  # change to a scalar if the gripper is made symmetric

        if not self.has_object:
            achieved_goal = grip_pos.copy()
        else:
            achieved_goal = np.squeeze((objPosList[0]).copy())

        obs = np.concatenate([
            grip_pos, objPosList[0].ravel(), object_rel_pos.ravel(), gripper_state,
        ])

        num = 6
        for i in range(1,self.clutter_num+1):
            if np.linalg.norm(objPosList[0][:2]-objPosList[i][:2]) < 0.050 and num:
                obs = np.concatenate([obs,objPosList[i]])
                num -= 1

        for i in range(num):
            obs = np.concatenate([obs,np.zeros(3)])

        
        return {
            'observation': obs.copy(),
            'achieved_goal': achieved_goal.copy(),
            'desired_goal': self.goal.copy(),
        }


    def _get_obs_single_object(self):
        # positions

        grip_pos = self.sim.data.get_site_xpos('dobot:grip')
        dt = self.sim.nsubsteps * self.sim.model.opt.timestep
        grip_velp = self.sim.data.get_site_xvelp('dobot:grip') * dt
        robot_qpos, robot_qvel = utils.robot_get_obs(self.sim)
        objPosList = []
        if self.has_object:
            for i in range(self.clutter_num + 1):
                if i == 0:
                    object_pos = self.sim.data.get_site_xpos('site:object' + str(i))
                    objPosList.append(object_pos)
                    # rotations
                    object_rot = rotations.mat2euler(self.sim.data.get_site_xmat('site:object' + str(i)))
                    # velocities
                    object_velp = self.sim.data.get_site_xvelp('site:object' + str(i)) * dt
                    object_velr = self.sim.data.get_site_xvelr('site:object' + str(i)) * dt
                    # gripper state
                    object_rel_pos = object_pos - grip_pos
                    object_velp -= grip_velp
                else:
                    object_pos = self.sim.data.get_site_xpos('site:object' + str(i))
                    objPosList.append(object_pos)
        else:
            object_pos = object_rot = object_velp = object_velr = object_rel_pos = np.zeros(0)

        gripper_state = robot_qpos[-2:]
        gripper_vel = robot_qvel[-2:] * dt  # change to a scalar if the gripper is made symmetric

        if not self.has_object:
            achieved_goal = grip_pos.copy()
        else:
            achieved_goal = np.squeeze((objPosList[0]).copy())
        obs = np.concatenate([
            grip_pos, objPosList[0].ravel(), object_rel_pos.ravel(), gripper_state, object_rot.ravel(),
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
        self.sim.set_state(self.initial_state)
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
        # if self.has_object:
        #     object_xpos = self.initial_gripper_xpos[:2]
        #     while np.linalg.norm(object_xpos - self.initial_gripper_xpos[:2]) < 0.1:
        #         object_xpos = self.initial_gripper_xpos[:2] + self.np_random.uniform(-self.obj_range, self.obj_range, size=2)
        #         #object_xpos[:2] = np.clip(object_xpos,[0.6,0.55],[1.0,0.95,0.47])
        #     object_qpos = self.sim.data.get_joint_qpos('object0:joint')
        #     assert object_qpos.shape == (7,)
        #     object_qpos[:2] = object_xpos
        #     object_qpos[2] += 0.005
        #     self.sim.data.set_joint_qpos('object0:joint', object_qpos)
            
        if self.has_object:
            pos = np.array([0.8,0.685])
            size = np.array([0.28,0.10]) - 0.020
            up = pos + size
            low = pos - size
            object_xpos = np.array([self.np_random.uniform(low[0],up[0]),self.np_random.uniform(low[1],up[1])])
            object_qpos = self.sim.data.get_joint_qpos('object0:joint')
            assert object_qpos.shape == (7,)
            object_qpos[:2] = object_xpos
            object_qpos[2] = 0.032
            self.sim.data.set_joint_qpos('object0:joint', object_qpos)

        self.sim.forward()
        return True

    def clutter(self):
        count = self.clutter_num
        # nums = list(range(1,61))
        for i in range(1,count+1):
            # choice = self.np_random.choice(nums)
            # del nums[nums.index(choice)]
            choice = i
            object_name = "object{}:joint".format(choice)
            
            pos = np.array([0.8,0.685])
            size = np.array([0.28,0.10]) - 0.020
            up = pos + size
            low = pos - size
            object_xpos = np.array([self.np_random.uniform(low[0],up[0]),self.np_random.uniform(low[1],up[1])])

            object_qpos = self.sim.data.get_joint_qpos(object_name)
            assert object_qpos.shape == (7,)
            object_qpos[:2] = object_xpos
            object_qpos[2] = 0.032
            object_qpos[2] += self.np_random.uniform(0.005, 0.15)
            self.sim.data.set_joint_qpos(object_name, object_qpos)



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
        size = np.array([0.28,0.10]) - 0.020
        up = pos + size
        low = pos - size
        # goal = np.array([self.np_random.uniform(low[0],up[0]),self.np_random.uniform(low[1],up[1]),0.148])
        goal = np.array([0.97, 0.595, 0.148])


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
        self.initial_gripper_xpos = np.array([0.8, 0.75, 0.2975])
        #print(self.initial_gripper_xpos)
        if self.has_object:
            self.height_offset = self.sim.data.get_site_xpos('site:object0')[2]


    def capture(self):
        if self.viewer == None:
            pass
        else:
            self.viewer.cam.fixedcamid = 0
            self.viewer.cam.type = const.CAMERA_FIXED
            img = self.viewer._read_pixels_as_in_window()
            return img

    # def _setPos(self, goal, obs, params):
    #     # Get all positions
    #     gripperPos = obs[:3]
    #     object0Pos = obs[3:6]
    #     gripperState = obs[9:11]
    #     clutterPos = []
    #     for i in range(4):
    #         clutterPos.append(np.array(obs[3*i+11:3*i+14]))
    #     # Set all positions

    #     # Set gripper state
    #     names = [n for n in sim.model.joint_names if n.startswith('dobot')][-2:]
    #     print(names)
    #     import sys
    #     sys.exit()
    #     self.sim.data.set_joint_qpos(object_name,)
    #     # self.sim.data.set_joint_qpos("dobot:grip", np.array(gripperPos.tolist() + [1, 0, 0, 0]))
    #     for i in range(params['clutter_num']+1):
    #         choice = i
    #         object_name = "object{}:joint".format(choice)
    #         object_xpos = []
    #         if i == 0:
    #             object_xpos = object0Pos
    #         elif i < 5 and np.array_equal(np.array(clutterPos[i-1]) , np.array([0., 0., 0.])):
    #             object_xpos = clutterPos[choice-1]
    #         else:
    #             object_xpos = np.array([4,4,0])
    #         self.sim.data.set_joint_qpos(object_name, np.array(object_xpos.tolist() + [1, 0, 0, 0]))
    #     sites_offset = (self.sim.data.site_xpos - self.sim.model.site_pos).copy()
    #     site_id = self.sim.model.site_name2id('target0')
    #     self.sim.model.site_pos[site_id] = goal - sites_offset[0]
    #     # self.sim.forward()

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