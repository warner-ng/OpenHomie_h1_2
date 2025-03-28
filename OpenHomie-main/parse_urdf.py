# import math
# import numpy as np 
from isaacgym import gymapi, gymutil
import torch
gym = gymapi.acquire_gym()

sim_params = gymapi.SimParams()
sim_params.dt = 1.0 / 60.0
sim_params.physx.solver_type = 1
sim_params.physx.num_position_iterations = 6
sim_params.physx.num_velocity_iterations = 0
sim_params.physx.use_gpu = False
sim_params.use_gpu_pipeline = False

sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)
if sim is None:
    print("Failed to create sim")
    
asset_root = "/home/wubinghuan/projects/OpenHomie_h1_2/OpenHomie-main/HomieRL/legged_gym/resources/robots/"
asset_file = "h1_2_description/h1_2_handless.urdf"
asset_options = gymapi.AssetOptions()
asset_options.fix_base_link = True
asset_options.flip_visual_attachments = False
asset_options.use_mesh_materials = True

asset = gym.load_asset(sim, asset_root, asset_file, asset_options)

dof_names = gym.get_asset_dof_names(asset)
for i, item in enumerate(dof_names):
    print(f"{i} \"{item}\",")
print(len(dof_names))
# link_names = gym.get_asset_rigid_body_names(asset) # link names
# for i, item in enumerate(link_names):
#     print(f"{i} \"{item}\",")
# print("****************************************")
# print(len(link_names))

dof_props_asset = gym.get_asset_dof_properties(asset)
dof_pos_limit = torch.zeros(len(dof_names), 2, dtype=torch.float, device="cuda:0", requires_grad=False)
torque_limits = torch.zeros(len(dof_names), dtype=torch.float, device="cuda:0")
for i in range(len(dof_props_asset)):
    dof_pos_limit[i, 0] = dof_props_asset["lower"][i].item()
    dof_pos_limit[i, 1] = dof_props_asset["upper"][i].item()
    torque_limits[i] = dof_props_asset["effort"][i].item()
print("****************************************")
import numpy as np

print((dof_pos_limit.cpu().numpy().transpose()).tolist())    
print((torque_limits.cpu().numpy().transpose()).tolist())    