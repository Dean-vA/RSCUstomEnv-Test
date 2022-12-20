from RSCustomEnv import RoboSuite
from stable_baselines3 import PPO
import os
import wandb
from wandb.integration.sb3 import WandbCallback
import argparse
from clearml import Task, OutputModel

# Replace Pendulum-v1/YourName with your own project name (Folder/YourName, e.g. 2022-Y2B-RoboSuite/Michael)
task = Task.init(project_name='2022-Y2B-RoboSuite/Dean', task_name='Experiment6')#, output_uri=True, auto_connect_frameworks={'pytorch': False})
output_model = OutputModel(task=task, framework="PyTorch")
#setting the base docker image
task.set_base_docker('deanis/robosuite:py3.8-2')
#setting the task to run remotely on the default queue
task.execute_remotely(queue_name="default")

parser = argparse.ArgumentParser()
parser.add_argument("--learning_rate", type=float, default=0.0003)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--n_steps", type=int, default=8192)
parser.add_argument("--n_epochs", type=int, default=20)

args = parser.parse_args()

os.environ['WANDB_API_KEY'] = 'e08ae8b88ab980fc97e670c026c961dd757a66f8'

env = RoboSuite(Environment='PickPlace', CustomReward=True)

# initialize wandb project
run = wandb.init(project="rsTest",sync_tensorboard=True)

# add tensorboard logging to the model
model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=f"runs/{run.id}", 
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            n_steps=args.n_steps,
            n_epochs=args.n_epochs,
            device='cpu')

# create wandb callback
wandb_callback = WandbCallback(model_save_freq=10000,
                                model_save_path=f"models/{run.id}",
                                verbose=2,
                                )

# variable for how often to save the model
time_steps = 100000
for i in range(25):
    # add the reset_num_timesteps=False argument to the learn function to prevent the model from resetting the timestep counter
    # add the tb_log_name argument to the learn function to log the tensorboard data to the correct folder
    model.learn(total_timesteps=time_steps, callback=wandb_callback, progress_bar=True, reset_num_timesteps=False,tb_log_name=f"runs/{run.id}")
    # save the model to the models folder with the run id and the current timestep
    model.save(f"models/{run.id}/{time_steps*(i+1)}")