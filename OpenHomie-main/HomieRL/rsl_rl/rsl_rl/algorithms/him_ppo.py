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

import torch
import torch.nn as nn
import torch.optim as optim

from rsl_rl.modules import HIMActorCritic
from rsl_rl.storage import HIMRolloutStorage

class HIMPPO:
    actor_critic: HIMActorCritic
    def __init__(self,
                 actor_critic,
                 use_flip = True,
                 num_learning_epochs=1,
                 num_mini_batches=1,
                 clip_param=0.2,
                 gamma=0.998,
                 lam=0.95,
                 value_loss_coef=1.0,
                 entropy_coef=0.0,
                 learning_rate=1e-3,
                 max_grad_norm=1.0,
                 use_clipped_value_loss=True,
                 schedule="fixed",
                 desired_kl=0.01,
                 device='cpu',
                 symmetry_scale=1e-3,
                 ):

        self.device = device
        self.use_flip = use_flip

        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate

        # PPO components
        self.actor_critic = actor_critic
        self.actor_critic.to(self.device)
        self.storage = None # initialized later
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate)
        self.transition = HIMRolloutStorage.Transition()
        self.transition_sym = HIMRolloutStorage.Transition()
        self.symmetry_scale = symmetry_scale
        # PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

    def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape):
        self.storage = HIMRolloutStorage(num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape, self.device)

    def test_mode(self):
        self.actor_critic.test()
    
    def train_mode(self):
        self.actor_critic.train()

    def act(self, obs, critic_obs):
        # Compute the actions and values
        self.transition.actions = self.actor_critic.act(obs).detach()
        self.transition.values = self.actor_critic.evaluate(critic_obs).detach()
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()
        # need to record obs and critic_obs before env.step()
        self.transition.observations = obs
        self.transition.critic_observations = critic_obs
        
        obs_sym = self.flip_g1_actor_obs(obs)
        critic_obs_sym = self.flip_g1_critic_obs(critic_obs)
        self.transition_sym.actions = self.actor_critic.act(obs_sym).detach()
        self.transition_sym.values = self.actor_critic.evaluate(critic_obs_sym).detach()
        self.transition_sym.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition_sym.actions).detach()
        self.transition_sym.action_mean = self.actor_critic.action_mean.detach()
        self.transition_sym.action_sigma = self.actor_critic.action_std.detach()
        # need to record obs and critic_obs before env.step()
        self.transition_sym.observations = obs_sym
        self.transition_sym.critic_observations = critic_obs_sym
        return self.transition.actions
    
    def process_env_step(self, rewards, dones, infos, next_critic_obs):
        self.transition.next_critic_observations = next_critic_obs.clone()
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        
        next_critic_obs_sym = self.flip_g1_critic_obs(next_critic_obs)
        self.transition_sym.next_critic_observations = next_critic_obs_sym.clone()
        self.transition_sym.rewards = rewards.clone()
        self.transition_sym.dones = dones
        # Bootstrapping on time outs
        if 'time_outs' in infos:
            self.transition.rewards += self.gamma * torch.squeeze(self.transition.values * infos['time_outs'].unsqueeze(1).to(self.device), 1)
            self.transition_sym.rewards += self.gamma * torch.squeeze(self.transition_sym.values * infos['time_outs'].unsqueeze(1).to(self.device), 1)
        # Record the transition
        self.storage.add_transitions(self.transition)
        self.storage.add_transitions(self.transition_sym)
        self.transition.clear()
        self.transition_sym.clear()
        self.actor_critic.reset(dones)
    
    def compute_returns(self, last_critic_obs):
        last_values= self.actor_critic.evaluate(last_critic_obs).detach()
        self.storage.compute_returns(last_values, self.gamma, self.lam)

    def update(self):
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_estimation_loss = 0
        mean_swap_loss = 0
        mean_actor_sym_loss = 0
        mean_critic_sym_loss = 0
        
        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        for obs_batch, critic_obs_batch, actions_batch, next_critic_obs_batch, target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch, \
            old_mu_batch, old_sigma_batch in generator:
                self.actor_critic.act(obs_batch)
                actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
                value_batch = self.actor_critic.evaluate(critic_obs_batch)
                mu_batch = self.actor_critic.action_mean
                sigma_batch = self.actor_critic.action_std
                entropy_batch = self.actor_critic.entropy

                # KL
                if self.desired_kl != None and self.schedule == 'adaptive':
                    with torch.inference_mode():
                        kl = torch.sum(
                            torch.log(sigma_batch / old_sigma_batch + 1.e-5) + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch)) / (2.0 * torch.square(sigma_batch)) - 0.5, axis=-1)
                        kl_mean = torch.mean(kl)

                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                        
                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] = self.learning_rate

                #Estimator Update
                if self.use_flip:
                    if self.actor_critic.num_actions == 19:
                        flipped_obs_batch = self.filp_actor_obs(obs_batch)
                        flipped_next_critic_obs_batch = self.flip_critic_obs(next_critic_obs_batch)
                    if self.actor_critic.num_actions == 12:
                        flipped_obs_batch = self.flip_g1_actor_obs(obs_batch)
                        flipped_next_critic_obs_batch = self.flip_g1_critic_obs(next_critic_obs_batch)
                    estimator_update_obs_batch =  torch.cat((obs_batch, flipped_obs_batch), dim=0)
                    estimator_update_next_critic_obs_batch = torch.cat((next_critic_obs_batch, flipped_next_critic_obs_batch), dim=0)
                else:
                    estimator_update_obs_batch = obs_batch
                    estimator_update_next_critic_obs_batch = next_critic_obs_batch
                estimation_loss, swap_loss = self.actor_critic.update_estimator(estimator_update_obs_batch, estimator_update_next_critic_obs_batch, lr=self.learning_rate)
                
                # Surrogate loss
                ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
                surrogate = -torch.squeeze(advantages_batch) * ratio
                surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
                surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

                # Value function loss
                if self.use_clipped_value_loss:
                    value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(-self.clip_param, self.clip_param)
                    value_losses = (value_batch - returns_batch).pow(2)
                    value_losses_clipped = (value_clipped - returns_batch).pow(2)
                    value_loss = torch.max(value_losses, value_losses_clipped).mean()
                else:
                    value_loss = (returns_batch - value_batch).pow(2).mean()
                    
                if self.use_flip:
                    if self.actor_critic.num_actions == 19:
                        flipped_obs_batch = self.filp_actor_obs(obs_batch)
                        flipped_critic_obs_batch = self.flip_critic_obs(critic_obs_batch)
                        actor_sym_loss = self.symmetry_scale * torch.mean(torch.sum(torch.square(self.actor_critic.act_inference(flipped_obs_batch) - self.flip_actions(self.actor_critic.act_inference(obs_batch))), dim=-1))
                    if self.actor_critic.num_actions == 12:
                        flipped_critic_obs_batch = self.flip_g1_critic_obs(critic_obs_batch)
                        actor_sym_loss = self.symmetry_scale * torch.mean(torch.sum(torch.square(self.actor_critic.act_inference(flipped_obs_batch) - self.flip_g1_actions(self.actor_critic.act_inference(obs_batch))), dim=-1))
                    
                    critic_sym_loss = self.symmetry_scale * torch.mean(torch.square(self.actor_critic.evaluate(flipped_critic_obs_batch) - self.actor_critic.evaluate(critic_obs_batch).detach()))
                    loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean() + actor_sym_loss + critic_sym_loss
                else:
                    loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()

                # Gradient step
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()

                mean_value_loss += value_loss.item()
                mean_surrogate_loss += surrogate_loss.item()
                mean_estimation_loss += estimation_loss
                mean_swap_loss += swap_loss
                if self.use_flip:
                    mean_actor_sym_loss += actor_sym_loss.item()
                    mean_critic_sym_loss += critic_sym_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_estimation_loss /= num_updates
        mean_swap_loss /= num_updates
        if self.use_flip:
            mean_actor_sym_loss /= num_updates
            mean_critic_sym_loss /= num_updates
        self.storage.clear()

        if self.use_flip:
            return mean_value_loss, mean_surrogate_loss, mean_estimation_loss, mean_swap_loss, mean_actor_sym_loss, mean_critic_sym_loss
        else:
            return mean_value_loss, mean_surrogate_loss, estimation_loss, swap_loss, 0, 0
    
    
    def filp_actor_obs(self, obs):
        proprioceptive_obs = torch.clone(obs[:, :self.actor_critic.num_one_step_obs * self.actor_critic.actor_history_length])
        proprioceptive_obs = proprioceptive_obs.view(-1, self.actor_critic.actor_history_length, self.actor_critic.num_one_step_obs)
        
        if self.actor_critic.actor_use_height:
            height_obs = torch.clone(obs[:, self.actor_critic.num_one_step_obs * self.actor_critic.actor_history_length:])
            height_obs = height_obs.view(-1, 12, 8)
            flipped_height_obs = torch.flip(height_obs, dims=[2])
            
        flipped_proprioceptive_obs = torch.zeros_like(proprioceptive_obs)
        flipped_proprioceptive_obs[:, :, 0] =  proprioceptive_obs[:, :, 0] # x command
        flipped_proprioceptive_obs[:, :, 1] = -proprioceptive_obs[:, :, 1] # y command
        flipped_proprioceptive_obs[:, :, 2] = -proprioceptive_obs[:, :, 2] # yaw command
        flipped_proprioceptive_obs[:, :, 3] = -proprioceptive_obs[:, :, 3] # base ang vel roll
        flipped_proprioceptive_obs[:, :, 4] =  proprioceptive_obs[:, :, 4] # base ang vel pitch
        flipped_proprioceptive_obs[:, :, 5] = -proprioceptive_obs[:, :, 5] # base ang vel yaw
        flipped_proprioceptive_obs[:, :, 6] =  proprioceptive_obs[:, :, 6] # projected gravity x
        flipped_proprioceptive_obs[:, :, 7] = -proprioceptive_obs[:, :, 7] # projected gravity y
        flipped_proprioceptive_obs[:, :, 8] =  proprioceptive_obs[:, :, 8] # projected gravity z
        
        # Joint Pos
        flipped_proprioceptive_obs[:, :,  9] = -proprioceptive_obs[:, :, 14] # left_hip_yaw_joint pos
        flipped_proprioceptive_obs[:, :, 10] = -proprioceptive_obs[:, :, 15] # left_hip_roll_joint pos
        flipped_proprioceptive_obs[:, :, 11] =  proprioceptive_obs[:, :, 16] # left_hip_pitch_joint pos
        flipped_proprioceptive_obs[:, :, 12] =  proprioceptive_obs[:, :, 17] # left_knee_joint pos
        flipped_proprioceptive_obs[:, :, 13] =  proprioceptive_obs[:, :, 18] # left_ankle_joint pos
        
        flipped_proprioceptive_obs[:, :, 14] = -proprioceptive_obs[:, :,  9] # right_hip_yaw_joint pos
        flipped_proprioceptive_obs[:, :, 15] = -proprioceptive_obs[:, :, 10] # right_hip_roll_joint pos
        flipped_proprioceptive_obs[:, :, 16] =  proprioceptive_obs[:, :, 11] # right_hip_pitch_joint pos
        flipped_proprioceptive_obs[:, :, 17] =  proprioceptive_obs[:, :, 12] # right_knee_joint pos
        flipped_proprioceptive_obs[:, :, 18] =  proprioceptive_obs[:, :, 13] # right_ankle_joint pos
        
        flipped_proprioceptive_obs[:, :, 19] = -proprioceptive_obs[:, :, 19] # torso_joint pos
        
        flipped_proprioceptive_obs[:, :, 20] =  proprioceptive_obs[:, :, 24] # left_shoulder_pitch_joint pos
        flipped_proprioceptive_obs[:, :, 21] = -proprioceptive_obs[:, :, 25] # left_shoulder_roll_joint pos
        flipped_proprioceptive_obs[:, :, 22] = -proprioceptive_obs[:, :, 26] # left_shoulder_yaw_joint pos
        flipped_proprioceptive_obs[:, :, 23] =  proprioceptive_obs[:, :, 27] # left_elbow_joint pos
        
        flipped_proprioceptive_obs[:, :, 24] =  proprioceptive_obs[:, :, 20] # right_shoulder_pitch_joint pos
        flipped_proprioceptive_obs[:, :, 25] = -proprioceptive_obs[:, :, 21] # right_shoulder_roll_joint pos
        flipped_proprioceptive_obs[:, :, 26] = -proprioceptive_obs[:, :, 22] # right_shoulder_yaw_joint pos
        flipped_proprioceptive_obs[:, :, 27] =  proprioceptive_obs[:, :, 23] # right_elbow_joint pos
        
        # Joint Vel
        flipped_proprioceptive_obs[:, :, 28] = -proprioceptive_obs[:, :, 33] # left_hip_yaw_joint vel
        flipped_proprioceptive_obs[:, :, 29] = -proprioceptive_obs[:, :, 34] # left_hip_roll_joint vel
        flipped_proprioceptive_obs[:, :, 30] =  proprioceptive_obs[:, :, 35] # left_hip_pitch_joint vel
        flipped_proprioceptive_obs[:, :, 31] =  proprioceptive_obs[:, :, 36] # left_knee_joint vel
        flipped_proprioceptive_obs[:, :, 32] =  proprioceptive_obs[:, :, 37] # left_ankle_joint vel
        
        flipped_proprioceptive_obs[:, :, 33] = -proprioceptive_obs[:, :, 28] # right_hip_yaw_joint vel
        flipped_proprioceptive_obs[:, :, 34] = -proprioceptive_obs[:, :, 29] # right_hip_roll_joint vel
        flipped_proprioceptive_obs[:, :, 35] =  proprioceptive_obs[:, :, 30] # right_hip_pitch_joint vel
        flipped_proprioceptive_obs[:, :, 36] =  proprioceptive_obs[:, :, 31] # right_knee_joint vel
        flipped_proprioceptive_obs[:, :, 37] =  proprioceptive_obs[:, :, 32] # right_ankle_joint vel
        
        flipped_proprioceptive_obs[:, :, 38] = -proprioceptive_obs[:, :, 38] # torso_joint vel
        
        flipped_proprioceptive_obs[:, :, 39] =  proprioceptive_obs[:, :, 43] # left_shoulder_pitch_joint vel
        flipped_proprioceptive_obs[:, :, 40] = -proprioceptive_obs[:, :, 44] # left_shoulder_roll_joint vel
        flipped_proprioceptive_obs[:, :, 41] = -proprioceptive_obs[:, :, 45] # left_shoulder_yaw_joint vel
        flipped_proprioceptive_obs[:, :, 42] =  proprioceptive_obs[:, :, 46] # left_elbow_joint vel
        
        flipped_proprioceptive_obs[:, :, 43] =  proprioceptive_obs[:, :, 39] # right_shoulder_pitch_joint vel
        flipped_proprioceptive_obs[:, :, 44] = -proprioceptive_obs[:, :, 40] # right_shoulder_roll_joint vel
        flipped_proprioceptive_obs[:, :, 45] = -proprioceptive_obs[:, :, 41] # right_shoulder_yaw_joint vel
        flipped_proprioceptive_obs[:, :, 46] =  proprioceptive_obs[:, :, 42] # right_elbow_joint vel
        
        # Last Action
        flipped_proprioceptive_obs[:, :, 47] = -proprioceptive_obs[:, :, 52] # left_hip_yaw last action
        flipped_proprioceptive_obs[:, :, 48] = -proprioceptive_obs[:, :, 53] # left_hip_roll last action
        flipped_proprioceptive_obs[:, :, 49] =  proprioceptive_obs[:, :, 54] # left_hip_pitch last action
        flipped_proprioceptive_obs[:, :, 50] =  proprioceptive_obs[:, :, 55] # left_knee last action
        flipped_proprioceptive_obs[:, :, 51] =  proprioceptive_obs[:, :, 56] # left_ankle last action
        
        flipped_proprioceptive_obs[:, :, 52] = -proprioceptive_obs[:, :, 47] # right_hip_yaw last action
        flipped_proprioceptive_obs[:, :, 53] = -proprioceptive_obs[:, :, 48] # right_hip_roll last action
        flipped_proprioceptive_obs[:, :, 54] =  proprioceptive_obs[:, :, 49] # right_hip_pitch last action
        flipped_proprioceptive_obs[:, :, 55] =  proprioceptive_obs[:, :, 50] # right_knee last action
        flipped_proprioceptive_obs[:, :, 56] =  proprioceptive_obs[:, :, 51] # right_ankle last action
        
        flipped_proprioceptive_obs[:, :, 57] = -proprioceptive_obs[:, :, 57] # torso last action
        
        flipped_proprioceptive_obs[:, :, 58] =  proprioceptive_obs[:, :, 62] # left_shoulder_pitch last action
        flipped_proprioceptive_obs[:, :, 59] = -proprioceptive_obs[:, :, 63] # left_shoulder_roll last action
        flipped_proprioceptive_obs[:, :, 60] = -proprioceptive_obs[:, :, 64] # left_shoulder_yaw last action
        flipped_proprioceptive_obs[:, :, 61] =  proprioceptive_obs[:, :, 65] # left_elbow last action
        
        flipped_proprioceptive_obs[:, :, 62] =  proprioceptive_obs[:, :, 58] # right_shoulder_pitch last action
        flipped_proprioceptive_obs[:, :, 63] = -proprioceptive_obs[:, :, 59] # right_shoulder_roll last action
        flipped_proprioceptive_obs[:, :, 64] = -proprioceptive_obs[:, :, 60] # right_shoulder_yaw last action
        flipped_proprioceptive_obs[:, :, 65] =  proprioceptive_obs[:, :, 61] # right_elbow last action
        
        if self.actor_critic.actor_use_height:
            return torch.cat((flipped_proprioceptive_obs.view(-1, self.actor_critic.num_one_step_obs * self.actor_critic.actor_history_length), flipped_height_obs.view(-1, self.actor_critic.num_height_points)), dim=-1).detach()
        else:
            return flipped_proprioceptive_obs.view(-1, self.actor_critic.num_one_step_obs * self.actor_critic.actor_history_length).detach()                                                                                                                                                                                                                                             
    
    def flip_critic_obs(self, critic_obs):
        proprioceptive_obs = torch.clone(critic_obs[:, :self.actor_critic.num_one_step_critic_obs * self.actor_critic.critic_history_length])
        height_obs = torch.clone(critic_obs[:, self.actor_critic.num_one_step_critic_obs * self.actor_critic.critic_history_length:])
        proprioceptive_obs = proprioceptive_obs.view(-1, self.actor_critic.critic_history_length, self.actor_critic.num_one_step_critic_obs)
        height_obs = height_obs.view(-1, 17, 11)
        flipped_proprioceptive_obs = torch.zeros_like(proprioceptive_obs)
        flipped_height_obs = torch.zeros_like(height_obs)
        
        flipped_proprioceptive_obs[:, :, 0] = proprioceptive_obs[:, :, 0] # x command
        flipped_proprioceptive_obs[:, :, 1] = -proprioceptive_obs[:, :, 1] # y command
        flipped_proprioceptive_obs[:, :, 2] = -proprioceptive_obs[:, :, 2] # yaw command
        flipped_proprioceptive_obs[:, :, 3] = -proprioceptive_obs[:, :, 3] # base ang vel roll
        flipped_proprioceptive_obs[:, :, 4] = proprioceptive_obs[:, :, 4] # base ang vel pitch
        flipped_proprioceptive_obs[:, :, 5] = -proprioceptive_obs[:, :, 5] # base ang vel yaw
        flipped_proprioceptive_obs[:, :, 6] = proprioceptive_obs[:, :, 6] # projected gravity x
        flipped_proprioceptive_obs[:, :, 7] = -proprioceptive_obs[:, :, 7] # projected gravity y
        flipped_proprioceptive_obs[:, :, 8] = proprioceptive_obs[:, :, 8] # projected gravity z
        
        flipped_proprioceptive_obs[:, :,  9] = -proprioceptive_obs[:, :, 14] # left_hip_yaw_joint pos
        flipped_proprioceptive_obs[:, :, 10] = -proprioceptive_obs[:, :, 15] # left_hip_roll_joint pos
        flipped_proprioceptive_obs[:, :, 11] = proprioceptive_obs[:, :, 16] # left_hip_pitch_joint pos
        flipped_proprioceptive_obs[:, :, 12] = proprioceptive_obs[:, :, 17] # left_knee_joint pos
        flipped_proprioceptive_obs[:, :, 13] = proprioceptive_obs[:, :, 18] # left_ankle_joint pos
        
        flipped_proprioceptive_obs[:, :, 14] = -proprioceptive_obs[:, :, 9] # right_hip_yaw_joint pos
        flipped_proprioceptive_obs[:, :, 15] = -proprioceptive_obs[:, :, 10] # right_hip_roll_joint pos
        flipped_proprioceptive_obs[:, :, 16] = proprioceptive_obs[:, :, 11] # right_hip_pitch_joint pos
        flipped_proprioceptive_obs[:, :, 17] = proprioceptive_obs[:, :, 12] # right_knee_joint pos
        flipped_proprioceptive_obs[:, :, 18] = proprioceptive_obs[:, :, 13] # right_ankle_joint pos
        
        flipped_proprioceptive_obs[:, :, 19] = -proprioceptive_obs[:, :, 19] # torso_joint pos
        
        flipped_proprioceptive_obs[:, :, 20] = proprioceptive_obs[:, :, 24] # left_shoulder_pitch_joint pos
        flipped_proprioceptive_obs[:, :, 21] = -proprioceptive_obs[:, :, 25] # left_shoulder_roll_joint pos
        flipped_proprioceptive_obs[:, :, 22] = -proprioceptive_obs[:, :, 26] # left_shoulder_yaw_joint pos
        flipped_proprioceptive_obs[:, :, 23] = proprioceptive_obs[:, :, 27] # left_elbow_joint pos
        
        flipped_proprioceptive_obs[:, :, 24] = proprioceptive_obs[:, :, 20] # right_shoulder_pitch_joint pos
        flipped_proprioceptive_obs[:, :, 25] = -proprioceptive_obs[:, :, 21] # right_shoulder_roll_joint pos
        flipped_proprioceptive_obs[:, :, 26] = -proprioceptive_obs[:, :, 22] # right_shoulder_yaw_joint pos
        flipped_proprioceptive_obs[:, :, 27] = proprioceptive_obs[:, :, 23] # right_elbow_joint pos
        
        flipped_proprioceptive_obs[:, :, 28] = -proprioceptive_obs[:, :, 33] # left_hip_yaw_joint vel
        flipped_proprioceptive_obs[:, :, 29] = -proprioceptive_obs[:, :, 34] # left_hip_roll_joint vel
        flipped_proprioceptive_obs[:, :, 30] = proprioceptive_obs[:, :, 35] # left_hip_pitch_joint vel
        flipped_proprioceptive_obs[:, :, 31] = proprioceptive_obs[:, :, 36] # left_knee_joint vel
        flipped_proprioceptive_obs[:, :, 32] = proprioceptive_obs[:, :, 37] # left_ankle_joint vel
        
        flipped_proprioceptive_obs[:, :, 33] = -proprioceptive_obs[:, :, 28] # right_hip_yaw_joint vel
        flipped_proprioceptive_obs[:, :, 34] = -proprioceptive_obs[:, :, 29] # right_hip_roll_joint vel
        flipped_proprioceptive_obs[:, :, 35] = proprioceptive_obs[:, :, 30] # right_hip_pitch_joint vel
        flipped_proprioceptive_obs[:, :, 36] = proprioceptive_obs[:, :, 31] # right_knee_joint vel
        flipped_proprioceptive_obs[:, :, 37] = proprioceptive_obs[:, :, 32] # right_ankle_joint vel
        
        flipped_proprioceptive_obs[:, :, 38] = -proprioceptive_obs[:, :, 38] # torso_joint vel
        
        flipped_proprioceptive_obs[:, :, 39] = proprioceptive_obs[:, :, 43] # left_shoulder_pitch_joint vel
        flipped_proprioceptive_obs[:, :, 40] = -proprioceptive_obs[:, :, 44] # left_shoulder_roll_joint vel
        flipped_proprioceptive_obs[:, :, 41] = -proprioceptive_obs[:, :, 45] # left_shoulder_yaw_joint vel
        flipped_proprioceptive_obs[:, :, 42] = proprioceptive_obs[:, :, 46] # left_elbow_joint vel
        
        flipped_proprioceptive_obs[:, :, 43] = proprioceptive_obs[:, :, 39] # right_shoulder_pitch_joint vel
        flipped_proprioceptive_obs[:, :, 44] = -proprioceptive_obs[:, :, 40] # right_shoulder_roll_joint vel
        flipped_proprioceptive_obs[:, :, 45] = -proprioceptive_obs[:, :, 41] # right_shoulder_yaw_joint vel
        flipped_proprioceptive_obs[:, :, 46] = proprioceptive_obs[:, :, 42] # right_elbow_joint vel
        
        flipped_proprioceptive_obs[:, :, 47] = -proprioceptive_obs[:, :, 52] # left_hip_yaw last action
        flipped_proprioceptive_obs[:, :, 48] = -proprioceptive_obs[:, :, 53] # left_hip_roll last action
        flipped_proprioceptive_obs[:, :, 49] = proprioceptive_obs[:, :, 54] # left_hip_pitch last action
        flipped_proprioceptive_obs[:, :, 50] = proprioceptive_obs[:, :, 55] # left_knee last action
        flipped_proprioceptive_obs[:, :, 51] = proprioceptive_obs[:, :, 56] # left_ankle last action
        
        flipped_proprioceptive_obs[:, :, 52] = -proprioceptive_obs[:, :, 47] # right_hip_yaw last action
        flipped_proprioceptive_obs[:, :, 53] = -proprioceptive_obs[:, :, 48] # right_hip_roll last action
        flipped_proprioceptive_obs[:, :, 54] = proprioceptive_obs[:, :, 49] # right_hip_pitch last action
        flipped_proprioceptive_obs[:, :, 55] = proprioceptive_obs[:, :, 50] # right_knee last action
        flipped_proprioceptive_obs[:, :, 56] = proprioceptive_obs[:, :, 51] # right_ankle last action
        
        flipped_proprioceptive_obs[:, :, 57] = -proprioceptive_obs[:, :, 57] # torso last action
        
        flipped_proprioceptive_obs[:, :, 58] = proprioceptive_obs[:, :, 62] # left_shoulder_pitch last action
        flipped_proprioceptive_obs[:, :, 59] = -proprioceptive_obs[:, :, 63] # left_shoulder_roll last action
        flipped_proprioceptive_obs[:, :, 60] = -proprioceptive_obs[:, :, 64] # left_shoulder_yaw last action
        flipped_proprioceptive_obs[:, :, 61] = proprioceptive_obs[:, :, 65] # left_elbow last action
        
        flipped_proprioceptive_obs[:, :, 62] = proprioceptive_obs[:, :, 58] # right_shoulder_pitch last action
        flipped_proprioceptive_obs[:, :, 63] = -proprioceptive_obs[:, :, 59] # right_shoulder_roll last action
        flipped_proprioceptive_obs[:, :, 64] = -proprioceptive_obs[:, :, 60] # right_shoulder_yaw last action
        flipped_proprioceptive_obs[:, :, 65] = proprioceptive_obs[:, :, 61] # right_elbow last action
        
        flipped_proprioceptive_obs[:, :, 66] = proprioceptive_obs[:, :, 66] # base lin vel x
        flipped_proprioceptive_obs[:, :, 67] = -proprioceptive_obs[:, :, 67] # base lin vel y
        flipped_proprioceptive_obs[:, :, 68] = proprioceptive_obs[:, :, 68] # base lin vel z
        
        flipped_height_obs = torch.flip(height_obs, dims=[2])
        
        return torch.cat((flipped_proprioceptive_obs.view(-1, self.actor_critic.num_one_step_critic_obs * self.actor_critic.critic_history_length), flipped_height_obs.view(-1, self.actor_critic.num_critic_height_points)), dim=-1).detach()
    
    def flip_actions(self, actions):
        flipped_actions = torch.zeros_like(actions)
        flipped_actions[:, 0] = -actions[:, 5] # left_hip_yaw
        flipped_actions[:, 1] = -actions[:, 6] # left_hip_roll
        flipped_actions[:, 2] = actions[:, 7] # left_hip_pitch
        flipped_actions[:, 3] = actions[:, 8] # left_knee
        flipped_actions[:, 4] = actions[:, 9] # left_ankle
        flipped_actions[:, 5] = -actions[:, 0] # right_hip_yaw
        flipped_actions[:, 6] = -actions[:, 1] # right_hip_roll
        flipped_actions[:, 7] = actions[:, 2] # right_hip_pitch
        flipped_actions[:, 8] = actions[:, 3] # right_knee
        flipped_actions[:, 9] = actions[:, 4] # right_ankle
        flipped_actions[:, 10] = -actions[:, 10] # torso_yaw
        flipped_actions[:, 11] = actions[:, 15] # left_shoulder_pitch
        flipped_actions[:, 12] = -actions[:, 16] # left_shoulder_roll
        flipped_actions[:, 13] = -actions[:, 17] # left_shoulder_yaw
        flipped_actions[:, 14] = actions[:, 18] # left_elbow
        flipped_actions[:, 15] = actions[:, 11] # right_shoulder_pitch
        flipped_actions[:, 16] = -actions[:, 12] # right_shoulder_roll
        flipped_actions[:, 17] = -actions[:, 13] # right_shoulder_yaw
        flipped_actions[:, 18] = actions[:, 14] # right_elbow
        
        return flipped_actions.detach()
    
    
    
    def flip_g1_actor_obs(self, obs):
        proprioceptive_obs = torch.clone(obs[:, :self.actor_critic.num_one_step_obs * self.actor_critic.actor_history_length])
        proprioceptive_obs = proprioceptive_obs.view(-1, self.actor_critic.actor_history_length, self.actor_critic.num_one_step_obs)
        
        if self.actor_critic.actor_use_height:
            height_obs = torch.clone(obs[:, self.actor_critic.num_one_step_obs * self.actor_critic.actor_history_length:])
            height_obs = height_obs.view(-1, 12, 8)
            flipped_height_obs = torch.flip(height_obs, dims=[2])
        
        flipped_proprioceptive_obs = torch.zeros_like(proprioceptive_obs)
        flipped_proprioceptive_obs[:, :, 0] =  proprioceptive_obs[:, :, 0] # x command
        flipped_proprioceptive_obs[:, :, 1] = -proprioceptive_obs[:, :, 1] # y command
        flipped_proprioceptive_obs[:, :, 2] = -proprioceptive_obs[:, :, 2] # yaw command
        flipped_proprioceptive_obs[:, :, 3] =  proprioceptive_obs[:, :, 3] # height command
        flipped_proprioceptive_obs[:, :, 5-1] = -proprioceptive_obs[:, :, 5-1] # base ang vel roll
        flipped_proprioceptive_obs[:, :, 6-1] =  proprioceptive_obs[:, :, 6-1] # base ang vel pitch
        flipped_proprioceptive_obs[:, :, 7-1] = -proprioceptive_obs[:, :, 7-1] # base ang vel yaw
        flipped_proprioceptive_obs[:, :, 8-1] =  proprioceptive_obs[:, :, 8-1] # projected gravity x
        flipped_proprioceptive_obs[:, :, 9-1] = -proprioceptive_obs[:, :, 9-1] # projected gravity y
        flipped_proprioceptive_obs[:, :, 10-1] =  proprioceptive_obs[:, :, 10-1] # projected gravity z
        
        # joint pos
        flipped_proprioceptive_obs[:, :, 11-1] = -proprioceptive_obs[:, :, 17-1] # lower
        flipped_proprioceptive_obs[:, :, 12-1] =  proprioceptive_obs[:, :, 18-1]
        flipped_proprioceptive_obs[:, :, 13-1] = -proprioceptive_obs[:, :, 19-1]
        flipped_proprioceptive_obs[:, :, 14-1] =  proprioceptive_obs[:, :, 20-1]
        flipped_proprioceptive_obs[:, :, 15-1] =  proprioceptive_obs[:, :, 21-1]
        flipped_proprioceptive_obs[:, :, 16-1] = -proprioceptive_obs[:, :, 22-1]
        flipped_proprioceptive_obs[:, :, 17-1] = -proprioceptive_obs[:, :, 11-1]
        flipped_proprioceptive_obs[:, :, 18-1] =  proprioceptive_obs[:, :, 12-1]
        flipped_proprioceptive_obs[:, :, 19-1] = -proprioceptive_obs[:, :, 13-1]
        flipped_proprioceptive_obs[:, :, 20-1] =  proprioceptive_obs[:, :, 14-1]
        flipped_proprioceptive_obs[:, :, 21-1] =  proprioceptive_obs[:, :, 15-1]
        flipped_proprioceptive_obs[:, :, 22-1] = -proprioceptive_obs[:, :, 16-1]
        
        flipped_proprioceptive_obs[:, :, 23-1] =  -proprioceptive_obs[:, :, 23-1] # waist
        
        flipped_proprioceptive_obs[:, :, 24-1] =  proprioceptive_obs[:, :, 31-1] # left shoulder
        flipped_proprioceptive_obs[:, :, 25-1] = -proprioceptive_obs[:, :, 32-1]
        flipped_proprioceptive_obs[:, :, 26-1] = -proprioceptive_obs[:, :, 33-1]
        flipped_proprioceptive_obs[:, :, 27-1] =  proprioceptive_obs[:, :, 34-1] # elbow
        flipped_proprioceptive_obs[:, :, 28-1] = -proprioceptive_obs[:, :, 35-1] # wrist
        flipped_proprioceptive_obs[:, :, 29-1] =  proprioceptive_obs[:, :, 36-1]
        flipped_proprioceptive_obs[:, :, 30-1] = -proprioceptive_obs[:, :, 37-1]

        
        flipped_proprioceptive_obs[:, :, 31-1] =  proprioceptive_obs[:, :, 24-1] # right shoulder
        flipped_proprioceptive_obs[:, :, 32-1] = -proprioceptive_obs[:, :, 25-1]
        flipped_proprioceptive_obs[:, :, 33-1] = -proprioceptive_obs[:, :, 26-1]
        flipped_proprioceptive_obs[:, :, 34-1] =  proprioceptive_obs[:, :, 27-1] # elbow
        flipped_proprioceptive_obs[:, :, 35-1] = -proprioceptive_obs[:, :, 28-1] # wrist
        flipped_proprioceptive_obs[:, :, 36-1] =  proprioceptive_obs[:, :, 29-1]
        flipped_proprioceptive_obs[:, :, 37-1] = -proprioceptive_obs[:, :, 30-1] #36
        
        # joint vel
        flipped_proprioceptive_obs[:, :, 27+11-1] = -proprioceptive_obs[:, :, 27+17-1] # lower
        flipped_proprioceptive_obs[:, :, 27+12-1] =  proprioceptive_obs[:, :, 27+18-1]
        flipped_proprioceptive_obs[:, :, 27+13-1] = -proprioceptive_obs[:, :, 27+19-1]
        flipped_proprioceptive_obs[:, :, 27+14-1] =  proprioceptive_obs[:, :, 27+20-1]
        flipped_proprioceptive_obs[:, :, 27+15-1] =  proprioceptive_obs[:, :, 27+21-1]
        flipped_proprioceptive_obs[:, :, 27+16-1] = -proprioceptive_obs[:, :, 27+22-1]
        flipped_proprioceptive_obs[:, :, 27+17-1] = -proprioceptive_obs[:, :, 27+11-1]
        flipped_proprioceptive_obs[:, :, 27+18-1] =  proprioceptive_obs[:, :, 27+12-1]
        flipped_proprioceptive_obs[:, :, 27+19-1] = -proprioceptive_obs[:, :, 27+13-1]
        flipped_proprioceptive_obs[:, :, 27+20-1] =  proprioceptive_obs[:, :, 27+14-1]
        flipped_proprioceptive_obs[:, :, 27+21-1] =  proprioceptive_obs[:, :, 27+15-1]
        flipped_proprioceptive_obs[:, :, 27+22-1] = -proprioceptive_obs[:, :, 27+16-1]
        
        flipped_proprioceptive_obs[:, :, 27+23-1] =  -proprioceptive_obs[:, :, 27+23-1] # waist
        
        flipped_proprioceptive_obs[:, :, 27+24-1] =  proprioceptive_obs[:, :, 27+31-1] # left shoulder
        flipped_proprioceptive_obs[:, :, 27+25-1] = -proprioceptive_obs[:, :, 27+32-1]
        flipped_proprioceptive_obs[:, :, 27+26-1] = -proprioceptive_obs[:, :, 27+33-1]
        flipped_proprioceptive_obs[:, :, 27+27-1] =  proprioceptive_obs[:, :, 27+34-1] # elbow
        flipped_proprioceptive_obs[:, :, 27+28-1] = -proprioceptive_obs[:, :, 27+35-1] # wrist
        flipped_proprioceptive_obs[:, :, 27+29-1] =  proprioceptive_obs[:, :, 27+36-1]
        flipped_proprioceptive_obs[:, :, 27+30-1] = -proprioceptive_obs[:, :, 27+37-1]

        
        flipped_proprioceptive_obs[:, :, 27+31-1] =  proprioceptive_obs[:, :, 27+24-1] # right shoulder
        flipped_proprioceptive_obs[:, :, 27+32-1] = -proprioceptive_obs[:, :, 27+25-1]
        flipped_proprioceptive_obs[:, :, 27+33-1] = -proprioceptive_obs[:, :, 27+26-1]
        flipped_proprioceptive_obs[:, :, 27+34-1] =  proprioceptive_obs[:, :, 27+27-1] # elbow
        flipped_proprioceptive_obs[:, :, 27+35-1] = -proprioceptive_obs[:, :, 27+28-1] # wrist
        flipped_proprioceptive_obs[:, :, 27+36-1] =  proprioceptive_obs[:, :, 27+29-1]
        flipped_proprioceptive_obs[:, :, 27+37-1] = -proprioceptive_obs[:, :, 27+30-1] #36
        
        # joint target
        flipped_proprioceptive_obs[:, :, 11+54-1] = -proprioceptive_obs[:, :, 17+54-1] # lower
        flipped_proprioceptive_obs[:, :, 12+54-1] =  proprioceptive_obs[:, :, 18+54-1]
        flipped_proprioceptive_obs[:, :, 13+54-1] = -proprioceptive_obs[:, :, 19+54-1]
        flipped_proprioceptive_obs[:, :, 14+54-1] =  proprioceptive_obs[:, :, 20+54-1]
        flipped_proprioceptive_obs[:, :, 15+54-1] =  proprioceptive_obs[:, :, 21+54-1]
        flipped_proprioceptive_obs[:, :, 16+54-1] = -proprioceptive_obs[:, :, 22+54-1]
        flipped_proprioceptive_obs[:, :, 17+54-1] = -proprioceptive_obs[:, :, 11+54-1]
        flipped_proprioceptive_obs[:, :, 18+54-1] =  proprioceptive_obs[:, :, 12+54-1]
        flipped_proprioceptive_obs[:, :, 19+54-1] = -proprioceptive_obs[:, :, 13+54-1]
        flipped_proprioceptive_obs[:, :, 20+54-1] =  proprioceptive_obs[:, :, 14+54-1]
        flipped_proprioceptive_obs[:, :, 21+54-1] =  proprioceptive_obs[:, :, 15+54-1]
        flipped_proprioceptive_obs[:, :, 22+54-1] = -proprioceptive_obs[:, :, 16+54-1]

        
        if self.actor_critic.actor_use_height:
            return torch.cat((flipped_proprioceptive_obs.view(-1, self.actor_critic.num_one_step_obs * self.actor_critic.actor_history_length), flipped_height_obs.view(-1, self.actor_critic.num_height_points)), dim=-1).detach()
        else:
            return flipped_proprioceptive_obs.view(-1, self.actor_critic.num_one_step_obs * self.actor_critic.actor_history_length).detach()                                                                                                                                                                                                                                             
    
    def flip_g1_critic_obs(self, critic_obs):
        proprioceptive_obs = torch.clone(critic_obs[:, :self.actor_critic.num_one_step_critic_obs * self.actor_critic.critic_history_length])
        # height_obs = torch.clone(critic_obs[:, self.actor_critic.num_one_step_critic_obs * self.actor_critic.critic_history_length:])
        proprioceptive_obs = proprioceptive_obs.view(-1, self.actor_critic.critic_history_length, self.actor_critic.num_one_step_critic_obs)
        # height_obs = height_obs.view(-1, 17, 11)
        # flipped_proprioceptive_obs = torch.zeros_like(proprioceptive_obs)
        # flipped_height_obs = torch.zeros_like(height_obs)
        
        flipped_proprioceptive_obs = torch.zeros_like(proprioceptive_obs)
        flipped_proprioceptive_obs[:, :, 0] =  proprioceptive_obs[:, :, 0] # x command
        flipped_proprioceptive_obs[:, :, 1] = -proprioceptive_obs[:, :, 1] # y command
        flipped_proprioceptive_obs[:, :, 2] = -proprioceptive_obs[:, :, 2] # yaw command
        flipped_proprioceptive_obs[:, :, 3] =  proprioceptive_obs[:, :, 3] # height command
        flipped_proprioceptive_obs[:, :, 5-1] = -proprioceptive_obs[:, :, 5-1] # base ang vel roll
        flipped_proprioceptive_obs[:, :, 6-1] =  proprioceptive_obs[:, :, 6-1] # base ang vel pitch
        flipped_proprioceptive_obs[:, :, 7-1] = -proprioceptive_obs[:, :, 7-1] # base ang vel yaw
        flipped_proprioceptive_obs[:, :, 8-1] =  proprioceptive_obs[:, :, 8-1] # projected gravity x
        flipped_proprioceptive_obs[:, :, 9-1] = -proprioceptive_obs[:, :, 9-1] # projected gravity y
        flipped_proprioceptive_obs[:, :, 10-1] =  proprioceptive_obs[:, :, 10-1] # projected gravity z
        
        # joint pos
        flipped_proprioceptive_obs[:, :, 11-1] = -proprioceptive_obs[:, :, 17-1] # lower
        flipped_proprioceptive_obs[:, :, 12-1] =  proprioceptive_obs[:, :, 18-1]
        flipped_proprioceptive_obs[:, :, 13-1] = -proprioceptive_obs[:, :, 19-1]
        flipped_proprioceptive_obs[:, :, 14-1] =  proprioceptive_obs[:, :, 20-1]
        flipped_proprioceptive_obs[:, :, 15-1] =  proprioceptive_obs[:, :, 21-1]
        flipped_proprioceptive_obs[:, :, 16-1] = -proprioceptive_obs[:, :, 22-1]
        flipped_proprioceptive_obs[:, :, 17-1] = -proprioceptive_obs[:, :, 11-1]
        flipped_proprioceptive_obs[:, :, 18-1] =  proprioceptive_obs[:, :, 12-1]
        flipped_proprioceptive_obs[:, :, 19-1] = -proprioceptive_obs[:, :, 13-1]
        flipped_proprioceptive_obs[:, :, 20-1] =  proprioceptive_obs[:, :, 14-1]
        flipped_proprioceptive_obs[:, :, 21-1] =  proprioceptive_obs[:, :, 15-1]
        flipped_proprioceptive_obs[:, :, 22-1] = -proprioceptive_obs[:, :, 16-1]
        
        flipped_proprioceptive_obs[:, :, 23-1] =  -proprioceptive_obs[:, :, 23-1] # waist
        
        flipped_proprioceptive_obs[:, :, 24-1] =  proprioceptive_obs[:, :, 31-1] # left shoulder
        flipped_proprioceptive_obs[:, :, 25-1] = -proprioceptive_obs[:, :, 32-1]
        flipped_proprioceptive_obs[:, :, 26-1] = -proprioceptive_obs[:, :, 33-1]
        flipped_proprioceptive_obs[:, :, 27-1] =  proprioceptive_obs[:, :, 34-1] # elbow
        flipped_proprioceptive_obs[:, :, 28-1] = -proprioceptive_obs[:, :, 35-1] # wrist
        flipped_proprioceptive_obs[:, :, 29-1] =  proprioceptive_obs[:, :, 36-1]
        flipped_proprioceptive_obs[:, :, 30-1] = -proprioceptive_obs[:, :, 37-1]

        
        flipped_proprioceptive_obs[:, :, 31-1] =  proprioceptive_obs[:, :, 24-1] # right shoulder
        flipped_proprioceptive_obs[:, :, 32-1] = -proprioceptive_obs[:, :, 25-1]
        flipped_proprioceptive_obs[:, :, 33-1] = -proprioceptive_obs[:, :, 26-1]
        flipped_proprioceptive_obs[:, :, 34-1] =  proprioceptive_obs[:, :, 27-1] # elbow
        flipped_proprioceptive_obs[:, :, 35-1] = -proprioceptive_obs[:, :, 28-1] # wrist
        flipped_proprioceptive_obs[:, :, 36-1] =  proprioceptive_obs[:, :, 29-1]
        flipped_proprioceptive_obs[:, :, 37-1] = -proprioceptive_obs[:, :, 30-1] #36
        
        # joint vel
        flipped_proprioceptive_obs[:, :, 27+11-1] = -proprioceptive_obs[:, :, 27+17-1] # lower
        flipped_proprioceptive_obs[:, :, 27+12-1] =  proprioceptive_obs[:, :, 27+18-1]
        flipped_proprioceptive_obs[:, :, 27+13-1] = -proprioceptive_obs[:, :, 27+19-1]
        flipped_proprioceptive_obs[:, :, 27+14-1] =  proprioceptive_obs[:, :, 27+20-1]
        flipped_proprioceptive_obs[:, :, 27+15-1] =  proprioceptive_obs[:, :, 27+21-1]
        flipped_proprioceptive_obs[:, :, 27+16-1] = -proprioceptive_obs[:, :, 27+22-1]
        flipped_proprioceptive_obs[:, :, 27+17-1] = -proprioceptive_obs[:, :, 27+11-1]
        flipped_proprioceptive_obs[:, :, 27+18-1] =  proprioceptive_obs[:, :, 27+12-1]
        flipped_proprioceptive_obs[:, :, 27+19-1] = -proprioceptive_obs[:, :, 27+13-1]
        flipped_proprioceptive_obs[:, :, 27+20-1] =  proprioceptive_obs[:, :, 27+14-1]
        flipped_proprioceptive_obs[:, :, 27+21-1] =  proprioceptive_obs[:, :, 27+15-1]
        flipped_proprioceptive_obs[:, :, 27+22-1] = -proprioceptive_obs[:, :, 27+16-1]
        
        flipped_proprioceptive_obs[:, :, 27+23-1] =  -proprioceptive_obs[:, :, 27+23-1] # waist
        
        flipped_proprioceptive_obs[:, :, 27+24-1] =  proprioceptive_obs[:, :, 27+31-1] # left shoulder
        flipped_proprioceptive_obs[:, :, 27+25-1] = -proprioceptive_obs[:, :, 27+32-1]
        flipped_proprioceptive_obs[:, :, 27+26-1] = -proprioceptive_obs[:, :, 27+33-1]
        flipped_proprioceptive_obs[:, :, 27+27-1] =  proprioceptive_obs[:, :, 27+34-1] # elbow
        flipped_proprioceptive_obs[:, :, 27+28-1] = -proprioceptive_obs[:, :, 27+35-1] # wrist
        flipped_proprioceptive_obs[:, :, 27+29-1] =  proprioceptive_obs[:, :, 27+36-1]
        flipped_proprioceptive_obs[:, :, 27+30-1] = -proprioceptive_obs[:, :, 27+37-1]

        
        flipped_proprioceptive_obs[:, :, 27+31-1] =  proprioceptive_obs[:, :, 27+24-1] # right shoulder
        flipped_proprioceptive_obs[:, :, 27+32-1] = -proprioceptive_obs[:, :, 27+25-1]
        flipped_proprioceptive_obs[:, :, 27+33-1] = -proprioceptive_obs[:, :, 27+26-1]
        flipped_proprioceptive_obs[:, :, 27+34-1] =  proprioceptive_obs[:, :, 27+27-1] # elbow
        flipped_proprioceptive_obs[:, :, 27+35-1] = -proprioceptive_obs[:, :, 27+28-1] # wrist
        flipped_proprioceptive_obs[:, :, 27+36-1] =  proprioceptive_obs[:, :, 27+29-1]
        flipped_proprioceptive_obs[:, :, 27+37-1] = -proprioceptive_obs[:, :, 27+30-1] #36
        
        # joint target
        flipped_proprioceptive_obs[:, :, 11+54-1] = -proprioceptive_obs[:, :, 17+54-1] # lower
        flipped_proprioceptive_obs[:, :, 12+54-1] =  proprioceptive_obs[:, :, 18+54-1]
        flipped_proprioceptive_obs[:, :, 13+54-1] = -proprioceptive_obs[:, :, 19+54-1]
        flipped_proprioceptive_obs[:, :, 14+54-1] =  proprioceptive_obs[:, :, 20+54-1]
        flipped_proprioceptive_obs[:, :, 15+54-1] =  proprioceptive_obs[:, :, 21+54-1]
        flipped_proprioceptive_obs[:, :, 16+54-1] = -proprioceptive_obs[:, :, 22+54-1]
        flipped_proprioceptive_obs[:, :, 17+54-1] = -proprioceptive_obs[:, :, 11+54-1]
        flipped_proprioceptive_obs[:, :, 18+54-1] =  proprioceptive_obs[:, :, 12+54-1]
        flipped_proprioceptive_obs[:, :, 19+54-1] = -proprioceptive_obs[:, :, 13+54-1]
        flipped_proprioceptive_obs[:, :, 20+54-1] =  proprioceptive_obs[:, :, 14+54-1]
        flipped_proprioceptive_obs[:, :, 21+54-1] =  proprioceptive_obs[:, :, 15+54-1]
        flipped_proprioceptive_obs[:, :, 22+54-1] = -proprioceptive_obs[:, :, 16+54-1]
        
        flipped_proprioceptive_obs[:, :, 23+54-1] = proprioceptive_obs[:, :, 23+54-1] # base lin vel x
        flipped_proprioceptive_obs[:, :, 24+54-1] = -proprioceptive_obs[:, :, 24+54-1] # base lin vel y
        flipped_proprioceptive_obs[:, :, 25+54-1] = proprioceptive_obs[:, :, 25+54-1] # base lin vel z
        
        # flipped_height_obs = torch.flip(height_obs, dims=[2])
        # return torch.cat((flipped_proprioceptive_obs.view(-1, self.actor_critic.num_one_step_critic_obs * self.actor_critic.critic_history_length), flipped_height_obs.view(-1, self.actor_critic.num_critic_height_points)), dim=-1).detach()
        
        return flipped_proprioceptive_obs.view(-1, self.actor_critic.num_one_step_critic_obs * self.actor_critic.critic_history_length).detach()
    
    def flip_g1_actions(self, actions):
        flipped_actions = torch.zeros_like(actions)
        # flipped_actions[:, :] = actions[:, :]
        flipped_actions[:, 0] = -actions[:, 6]       
        flipped_actions[:, 1] =  actions[:, 7]       
        flipped_actions[:, 2] = -actions[:, 8]       
        flipped_actions[:, 3] =  actions[:, 9]       
        flipped_actions[:, 4] =  actions[:, 10]      
        flipped_actions[:, 5] = -actions[:, 11]      
        flipped_actions[:, 6] = -actions[:, 0]       
        flipped_actions[:, 7] =  actions[:, 1]       
        flipped_actions[:, 8] = -actions[:, 2]       
        flipped_actions[:, 9] =  actions[:, 3]       
        flipped_actions[:, 10] = actions[:, 4]       
        flipped_actions[:, 11] =-actions[:, 5]       
        return flipped_actions.detach()