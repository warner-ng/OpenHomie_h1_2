import time

import lcm
import numpy as np
import torch
from utils.cheetah_state_estimator import StateEstimator
from lcm_types.pd_tau_targets_lcmt import pd_tau_targets_lcmt
from lcm_types.arm_action_lcmt import arm_action_lcmt
from utils.command_profile import RCControllerProfile
from lcm_types.body_record_lcmt import body_record_lcmt
lc = lcm.LCM("udpm://239.255.76.67:7667?ttl=255")


class LCMAgent():
    def __init__(self, se:StateEstimator, command_profile: RCControllerProfile):
        self.se = se
        self.command_profile = command_profile

        self.dt = 1/50
        self.timestep = 0

        self.num_envs = 1
        self.num_dofs = 27
        self.num_obs = 2 * self.num_dofs + 10 + 12 # 91
        self.num_history_length = 6
        self.num_lower_dofs = 12
        self.num_commands = 4
        self.device = 'cuda:0'

        self.default_dof_pos = np.array([-0.1000,  0.0000,  0.0000,  0.3000, -0.2000,  0.0000, -0.1000,  0.0000,
         0.0000,  0.3000, -0.2000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
         0.0000,  0.0000,  0.0000,  0.0000,  0.0000, -0.0000,  0.0000,  0.0000,
         0.0000,  0.0000,  0.0000], dtype=np.float)
        self.p_gains = np.array([150., 150., 150., 300.,  40.,  40., 150., 150., 150., 300.,  40.,  40., 300., 200., 200., 200., 100.,  20.,  20.,  20., 200., 200., 200., 100., 20.,  20.,  20.], dtype=np.float)
        self.d_gains = np.array([2.0000, 2.0000, 2.0000, 4.0000, 4.0000, 4.0000, 2.0000, 2.0000, 2.0000, 4.0000, 4.0000, 4.0000, 5.0000, 4.0000, 4.0000, 4.0000, 1.0000, 0.5000,0.5000, 0.5000, 4.0000, 4.0000, 4.0000, 1.0000, 0.5000, 0.5000, 0.5000], dtype=np.float)

        self.torque_limit = np.array([ 88.,  88.,  88., 139.,  50.,  50.,  88.,  88.,  88., 139.,  50.,  50.,
         88.,  25.,  25.,  25.,  25.,  25.,   5.,   5.,  25.,  25.,  25.,  25.,
         25.,   5.,   5.])
        self.commands = np.zeros((1, self.num_commands))
        self.actions = torch.zeros(self.num_lower_dofs)
        self.last_actions = torch.zeros(self.num_lower_dofs) # TODO
        self.gravity_vector = np.zeros(3)
        self.dof_pos = np.zeros(self.num_dofs)
        self.dof_vel = np.zeros(self.num_dofs)
        self.body_angular_vel = np.zeros(3)
        self.joint_pos_target = np.zeros(self.num_dofs + 2)
        self.torques = np.zeros(self.num_dofs + 2)

        self.joint_idxs = self.se.joint_idxs


    def get_obs(self):
        self.gravity_vector = self.se.get_gravity_vector()
        cmds = self.command_profile.get_command(self.timestep * self.dt)
        self.commands[:, :] = cmds[:self.num_commands]
        # self.commands[:, :] = self.se.get_command()
        self.dof_pos = self.se.get_dof_pos()
        # body_record = body_record_lcmt()
        # body_record.q = self.dof_pos
        # lc.publish("body_data_record", body_record.encode())
        self.dof_vel = self.se.get_dof_vel()
        self.body_angular_vel = self.se.get_body_angular_vel()
        actions =  torch.cat((self.actions.reshape(1, -1).to("cuda:0"), torch.zeros(1, 15).to("cuda:0")), dim=-1).to("cuda:0")
        print("commands: ", self.commands[:, :])
        print("ang_vel: ", self.body_angular_vel)
        print("grav: ", self.gravity_vector)
        print("dof_pos: ", self.dof_pos)
        print("dof_vel: ", self.dof_vel)
        print("action: ", actions)
        ob = np.concatenate((self.commands[:, :] * np.array([2.0, 2.0, 0.25, 1.0]), # 4
                             self.body_angular_vel.reshape(1, -1) * 0.5, # 3
                             self.gravity_vector.reshape(1, -1), # 3
                             (self.dof_pos - self.default_dof_pos).reshape(1, -1),
                             self.dof_vel.reshape(1, -1) * 0.05,
                             actions.cpu().detach().numpy().reshape(1, -1)[:, :12]
                             ), axis=1)
        return torch.tensor(ob, device=self.device).float()

    def publish_action(self, action, hard_reset=False):
        action = action.cpu().numpy()
        command_for_robot = pd_tau_targets_lcmt()
        scaled_pos_target = action * 0.25 + self.default_dof_pos[:12]
        # torques = (scaled_pos_target - self.dof_pos[:12]) * self.p_gains[:12]  - self.dof_vel[:12] * self.d_gains[:12]   
        # torques = np.clip(torques[:12], -self.torque_limit[:12], self.torque_limit[:12])
        self.joint_pos_target[:12] = scaled_pos_target[:12]
        # arm_actions = self.se.get_arm_action()
        # self.joint_pos_target[15:] = 0.#arm_actions
        # self.joint_pos_target[12] = scaled_pos_target[12] # waist
        # self.joint_pos_target[15:] = scaled_pos_target[13:]
        # self.torques[:12] = torques[:12]
        # print("torques: ", torques)
        # print("==============================================================================")
        # self.torques[15:] = torques[13:] 
        command_for_robot.q_des = self.joint_pos_target
        command_for_robot.tau_ff = self.torques
        command_for_robot.timestamp_us = int(time.time() * 10 ** 6)

        lc.publish("pd_plustau_targets", command_for_robot.encode())

        # arm_action = arm_action_lcmt()
        # arm_action.act = arm_actions
        # lc.publish("new_arm_action", arm_action.encode())

    def reset(self):
        self.actions = torch.zeros(12)
        self.time = time.time()
        self.timestep = 0
        return self.get_obs()


    def step(self, actions, hard_reset=False):
        clip_actions = 100.
        self.last_actions = self.actions[:]
        self.actions = torch.clip(actions[0:1, :], -clip_actions, clip_actions)
        self.publish_action(self.actions, hard_reset=hard_reset)
        time.sleep(max(self.dt - (time.time() - self.time), 0))
        if self.timestep % 100 == 0: print(f'frq: {1 / (time.time() - self.time)} Hz');
        self.time = time.time()
        obs = self.get_obs()


        self.timestep += 1
        return obs
