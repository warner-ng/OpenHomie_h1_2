# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from legged_gym import LEGGED_GYM_ROOT_DIR
import os

import onnxruntime as ort

import isaacgym
from isaacgym import gymapi
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger

import numpy as np
import torch


def load_policy():
    body = torch.jit.load("", map_location="cuda:0")
    def policy(obs):
        action = body.forward(obs)
        return action
    return policy

def load_onnx_policy():
    model = ort.InferenceSession("")
    def run_inference(input_tensor):
        ort_inputs = {model.get_inputs()[0].name: input_tensor.cpu().numpy()}
        ort_outs = model.run(None, ort_inputs)
        return torch.tensor(ort_outs[0], device="cuda:0")
    return run_inference

def play(args, x_vel=0.0, y_vel=0.0, yaw_vel=0.0, height=0.74):

    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 50)
    env_cfg.terrain.num_rows = 10
    env_cfg.terrain.num_cols = 8
    env_cfg.terrain.curriculum = True
    env_cfg.terrain.max_init_terrain_level = 9
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.disturbance = False
    env_cfg.domain_rand.randomize_payload_mass = False
    env_cfg.domain_rand.randomize_body_displacement = False
    env_cfg.commands.heading_command = False
    env_cfg.commands.use_random = False
    env_cfg.terrain.mesh_type = 'plane'
    env_cfg.asset.self_collision = 0
    env_cfg.env.upper_teleop = False
    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    
    # Add keyboard event subscriptions for height control
    if hasattr(env, 'gym') and hasattr(env, 'viewer') and env.viewer is not None:
        # Height control
        env.gym.subscribe_viewer_keyboard_event(env.viewer, gymapi.KEY_Q, "height_increase")
        env.gym.subscribe_viewer_keyboard_event(env.viewer, gymapi.KEY_E, "height_decrease")
        env.gym.subscribe_viewer_keyboard_event(env.viewer, gymapi.KEY_R, "reset_height")
        env.gym.subscribe_viewer_keyboard_event(env.viewer, gymapi.KEY_1, "height_increase_fine")
        env.gym.subscribe_viewer_keyboard_event(env.viewer, gymapi.KEY_2, "height_decrease_fine")
        
        # Movement control
        env.gym.subscribe_viewer_keyboard_event(env.viewer, gymapi.KEY_W, "move_forward")
        env.gym.subscribe_viewer_keyboard_event(env.viewer, gymapi.KEY_S, "move_backward")
        env.gym.subscribe_viewer_keyboard_event(env.viewer, gymapi.KEY_A, "move_left")
        env.gym.subscribe_viewer_keyboard_event(env.viewer, gymapi.KEY_D, "move_right")
        env.gym.subscribe_viewer_keyboard_event(env.viewer, gymapi.KEY_Z, "turn_left")
        env.gym.subscribe_viewer_keyboard_event(env.viewer, gymapi.KEY_X, "turn_right")
        env.gym.subscribe_viewer_keyboard_event(env.viewer, gymapi.KEY_SPACE, "stop_all")
        
        print("Keyboard controls added:")
        print("  === HEIGHT CONTROL ===")
        print("  Q: Increase height (+0.02)")
        print("  E: Decrease height (-0.02)")
        print("  1: Fine increase height (+0.005)")
        print("  2: Fine decrease height (-0.005)")
        print("  R: Reset height to default (0.74)")
        print("  === MOVEMENT CONTROL ===")
        print("  W: Move forward")
        print("  S: Move backward") 
        print("  A: Move left")
        print("  D: Move right")
        print("  Z: Turn left")
        print("  X: Turn right")
        print("  SPACE: Stop all movement")
    
    # Height control parameters
    current_height = height
    height_step = 0.02
    height_step_fine = 0.005
    min_height = 0.1  # Safety minimum
    max_height = 1.2  # Safety maximum
    default_height = 0.74
    
    # Movement control parameters  
    current_x_vel = x_vel
    current_y_vel = y_vel
    current_yaw_vel = yaw_vel
    vel_step = 0.1  # Speed increment per keypress
    max_forward_vel = 1.2  # Based on training range
    max_backward_vel = -0.8
    max_lateral_vel = 0.5
    max_yaw_vel = 0.8
    env.commands[:, 0] = x_vel
    env.commands[:, 1] = y_vel
    env.commands[:, 2] = yaw_vel
    env.commands[:, 4] = height
    env.action_curriculum_ratio = 1.0
    obs = env.get_observations()
    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device) # Use this to load from trained pt file
    
    # policy = load_onnx_policy() # Use this to load from exported onnx file
    
    if EXPORT_POLICY:
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
        export_policy_as_jit(ppo_runner.alg.actor_critic, path)
        print('Exported policy as jit script to: ', path)
    print(policy)
    camera_position = np.array(env_cfg.viewer.pos, dtype=np.float64)
    camera_vel = np.array([1., 1., 0.])
    camera_direction = np.array(env_cfg.viewer.lookat) - np.array(env_cfg.viewer.pos)
    env.reset_idx(torch.arange(env.num_envs).to("cuda:0"))
    step_count = 0
    for _ in range(10*int(env.max_episode_length)):
        env.action_curriculum_ratio = 1.0
        
        # Check for keyboard events for height and movement control
        if hasattr(env, 'gym') and hasattr(env, 'viewer') and env.viewer is not None:
            for evt in env.gym.query_viewer_action_events(env.viewer):
                # Height controls
                if evt.action == "height_increase" and evt.value > 0:
                    current_height = min(current_height + height_step, max_height)
                    print(f"Height increased to: {current_height:.3f}")
                elif evt.action == "height_decrease" and evt.value > 0:
                    current_height = max(current_height - height_step, min_height)
                    print(f"Height decreased to: {current_height:.3f}")
                elif evt.action == "height_increase_fine" and evt.value > 0:
                    current_height = min(current_height + height_step_fine, max_height)
                    print(f"Height fine increased to: {current_height:.3f}")
                elif evt.action == "height_decrease_fine" and evt.value > 0:
                    current_height = max(current_height - height_step_fine, min_height)
                    print(f"Height fine decreased to: {current_height:.3f}")
                elif evt.action == "reset_height" and evt.value > 0:
                    current_height = default_height
                    print(f"Height reset to default: {current_height:.3f}")
                
                # Movement controls
                elif evt.action == "move_forward" and evt.value > 0:
                    current_x_vel = min(current_x_vel + vel_step, max_forward_vel)
                    print(f"Forward velocity: {current_x_vel:.2f} m/s")
                elif evt.action == "move_backward" and evt.value > 0:
                    current_x_vel = max(current_x_vel - vel_step, max_backward_vel)
                    print(f"Backward velocity: {current_x_vel:.2f} m/s")
                elif evt.action == "move_left" and evt.value > 0:
                    current_y_vel = min(current_y_vel + vel_step, max_lateral_vel)
                    print(f"Left velocity: {current_y_vel:.2f} m/s")
                elif evt.action == "move_right" and evt.value > 0:
                    current_y_vel = max(current_y_vel - vel_step, -max_lateral_vel)
                    print(f"Right velocity: {current_y_vel:.2f} m/s")
                elif evt.action == "turn_left" and evt.value > 0:
                    current_yaw_vel = min(current_yaw_vel + vel_step, max_yaw_vel)
                    print(f"Turn left velocity: {current_yaw_vel:.2f} rad/s")
                elif evt.action == "turn_right" and evt.value > 0:
                    current_yaw_vel = max(current_yaw_vel - vel_step, -max_yaw_vel)
                    print(f"Turn right velocity: {current_yaw_vel:.2f} rad/s")
                elif evt.action == "stop_all" and evt.value > 0:
                    current_x_vel = 0.0
                    current_y_vel = 0.0
                    current_yaw_vel = 0.0
                    print("All movement stopped!")
        
        # Display current status every 200 steps to avoid spam
        if step_count % 200 == 0:
            print(f"Status - Height: {current_height:.3f}m, Vel: [{current_x_vel:.2f}, {current_y_vel:.2f}, {current_yaw_vel:.2f}]")
        
        actions = policy(obs.detach())
        env.commands[:, 0] = current_x_vel   # Use dynamic x velocity
        env.commands[:, 1] = current_y_vel   # Use dynamic y velocity
        env.commands[:, 2] = current_yaw_vel # Use dynamic yaw velocity
        env.commands[:, 4] = current_height  # Use dynamic height
        obs, _, _, _, _, _, _ = env.step(actions.detach())
        step_count += 1
        
        if MOVE_CAMERA:
            camera_position += camera_vel * env.dt
            env.set_camera(camera_position, camera_position + camera_direction)

if __name__ == '__main__':
    EXPORT_POLICY = True
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    args = get_args()
    print("\n=== Interactive Robot Control ===")
    print("HEIGHT CONTROL:")
    print("  Q/E: Increase/Decrease height (±0.02)")
    print("  1/2: Fine height control (±0.005)")
    print("  R: Reset height to default (0.74)")
    print("\nMOVEMENT CONTROL:")
    print("  W/S: Move forward/backward")
    print("  A/D: Move left/right")
    print("  Z/X: Turn left/right")
    print("  SPACE: Stop all movement")
    print("\nOTHER:")
    print("  ESC: Quit simulation")
    print("=====================================")
    print("Velocity ranges: X[-0.8,1.2] Y[-0.5,0.5] Yaw[-0.8,0.8]")
    print("Height range: [0.1,1.2] meters")
    print("=====================================\n")
    play(args, x_vel=0., y_vel=0., yaw_vel=0., height=0.74)