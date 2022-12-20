#%%
# from RoboSuiteCustomEnv2 import RoboSuite
from RSCustomEnv import RoboSuite
#import robosuite as suite
#from robosuite.wrappers.gym_wrapper import GymWrapper
#from robosuite.controllers import load_controller_config
#from robosuite.utils.input_utils import *
import time
from stable_baselines3.common.env_checker import check_env
##%

env = RoboSuite(RenderMode=True, 
                #Horizon=50, 
                CustomReward=True, 
                #ControllerType='standard', 
                #DiscreteActions=False, 
                #ReducedActions=False, 
                #RewardShaping=True,
                #control_freq=20,
                )
# It will check your custom environment and output additional warnings if needed
check_env(env)

obs = env.reset()

for i in range(100):
    start_time = time.time()
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    if env.RenderMode:
        env.render()
    print(action)
    print(obs)
    print(reward)



# %%
