#%%
from clearml import Task

prev_task = Task.get_task(task_id='ce39de021ec74a1db737902ccd638192')
last_snapshot = prev_task.models['output'][-1]
local_weights_path = last_snapshot.get_local_copy()
# %%
