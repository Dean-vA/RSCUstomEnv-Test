#Environment for training to move griper to given pos and orientation.
#%%
import gym
from gym import spaces
import numpy as np

import robosuite as suite
from robosuite.wrappers.gym_wrapper import GymWrapper

class RoboSuite(gym.Env):
    """Custom Environment that follows gym interface"""
    def __init__(self, Environment = 'Lift', RenderMode=False, CustomReward=False, RewardShaping=True):
        super(RoboSuite, self).__init__()
        self.CustomReward = CustomReward
        self.RenderMode = RenderMode
        # Define action and observation space
        # They must be gym.spaces objects
        self.action_space = spaces.Box(low=-1, high=1, shape=(8,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(36,),dtype=np.float64)

        #Create RS envp
        self.RoboSuiteEnv = suite.make(env_name= Environment, 
                                        robots="Panda",
                                        has_renderer=RenderMode,
                                        has_offscreen_renderer=False,
                                        horizon=200,                                             
                                        use_camera_obs=False,)

        self.step_no = 0
        self.min_error = 1000
        self.target_pos = [0,0,0]

    #a function to parse the robosuite ordered dictionary and return a gym formatted observation matrix
    def parse_obs(self,obs):
        self.arm_telem = obs['robot0_proprio-state']
        self.gripper_pos = obs['robot0_eef_pos']
        obs = np.hstack((self.arm_telem,self.target_pos))
        return obs 

    def step(self, action):
        self.step_no += 1

        raw_obs, reward, done, info = self.RoboSuiteEnv.step(action)
        observation = self.parse_obs(raw_obs)

        if self.CustomReward:
            error = np.linalg.norm(self.target_pos[:3]-raw_obs['robot0_eef_pos']) #need to add angle error
            if error < 0.9*self.min_error:
                self.min_error = error
                reward = 1
            else:
                reward = 0

        return observation, reward, done, info

    def reset(self):
        observation = self.RoboSuiteEnv.reset()
        self.target_pos = [np.random.uniform(-0.5, 0.5), np.random.uniform(-0.5, 0.5), np.random.uniform(0.8, 1.3), np.random.uniform(-90, 90)]
        observation = self.parse_obs(observation)
        self.step_no = 0
        self.min_error = 1000
        return observation  # reward, done, info can't be included

    def render(self, mode='human'):
        self.RoboSuiteEnv.render()

    def close (self):
        self.RoboSuiteEnv.close()


            