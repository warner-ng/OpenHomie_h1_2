import copy
import time
import os

import numpy as np
import torch



class DeploymentRunner:
    def __init__(self, experiment_name="unnamed", se=None):
        self.agents = {}
        self.policy = None
        self.command_profile = None
        self.se = se

        self.control_agent_name = None
        self.command_agent_name = None



    def add_open_loop_agent(self, agent, name):
        self.agents[name] = agent
        self.logger.add_robot(name, agent.env.cfg)

    def add_control_agent(self, agent, name):
        self.control_agent_name = name
        self.agents[name] = agent

    def set_command_agents(self, name):
        self.command_agent = name

    def add_policy(self, policy):
        self.policy = policy

    def add_command_profile(self, command_profile):
        self.command_profile = command_profile


    def calibrate(self, wait=True):
        # first, if the robot is not in nominal pose, move slowly to the nominal pose
        for agent_name in self.agents.keys():
            if hasattr(self.agents[agent_name], "get_obs"):
                agent = self.agents[agent_name]
                agent.get_obs()
                joint_pos = agent.dof_pos[:12]
                final_goal = np.array([-0.1000,  0.0000,  0.0000,  0.3000, -0.2000,  0.0000, -0.1000,  0.0000,
                    0.0000,  0.3000, -0.2000,  0.0000], dtype=np.float)

                print(f"About to calibrate; the robot will stand [Press R2 to calibrate]")
                while wait:
                    if self.command_profile.state_estimator.right_lower_right_switch_pressed:
                        self.command_profile.state_estimator.right_lower_right_switch_pressed = False
                        break
                target = joint_pos
                cal_action = np.zeros((agent.num_envs, agent.num_lower_dofs))
                target_sequence = []
                while np.max(np.abs(target - final_goal)) > 0.01:
                    target -= np.clip((target - final_goal), -0.05, 0.05)
                    target_sequence += [copy.deepcopy(target)]
                for target in target_sequence:
                    next_target = target
                    action_scale = 0.25

                    next_target = next_target / action_scale
                    cal_action[:, 0:12] = next_target
                    agent.step(torch.from_numpy(cal_action))
                    agent.get_obs()
                    time.sleep(0.05)

                print("Starting pose calibrated [Press R2 to start controller]")
                while True:
                    if self.command_profile.state_estimator.right_lower_right_switch_pressed:
                        self.command_profile.state_estimator.right_lower_right_switch_pressed = False
                        break

                for agent_name in self.agents.keys():
                    obs = self.agents[agent_name].reset()
                    if agent_name == self.control_agent_name:
                        control_obs = obs

        return control_obs


    def run(self, num_log_steps=1000000000, max_steps=100000000):
        assert self.control_agent_name is not None, "cannot deploy, runner has no control agent!"
        # assert self.policy is not None, "cannot deploy, runner has no policy!"
        assert self.command_profile is not None, "cannot deploy, runner has no command profile!"

        # TODO: add basic test for comms

        for agent_name in self.agents.keys():
            obs = self.agents[agent_name].reset()
            if agent_name == self.control_agent_name:
                control_obs = obs
        control_obs = self.calibrate(wait=True)['obs_history']

        # now, run control loop

        try:
            for i in range(max_steps):

                action = self.policy(control_obs)
                for agent_name in self.agents.keys():
                    obs = self.agents[agent_name].step(action)

                    if agent_name == self.control_agent_name:
                        control_obs = obs['obs_history']

                # bad orientation emergency stop
                rpy = self.agents[self.control_agent_name].se.get_rpy()
                if abs(rpy[0]) > 1.6 or abs(rpy[1]) > 1.6:
                    self.calibrate(wait=False, low=True)

                if self.command_profile.state_estimator.right_lower_right_switch_pressed:
                    control_obs = self.calibrate(wait=False)['obs_history']
                    time.sleep(1)
                    self.command_profile.state_estimator.right_lower_right_switch_pressed = False
                    while not self.command_profile.state_estimator.right_lower_right_switch_pressed:
                        time.sleep(0.01)
                    self.command_profile.state_estimator.right_lower_right_switch_pressed = False

            # finally, return to the nominal pose
            control_obs = self.calibrate(wait=False)

        except KeyboardInterrupt:
            pass
