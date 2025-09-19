import torch


class CommandProfile:
    def __init__(self, dt, max_time_s=10.):
        self.dt = dt
        self.max_timestep = int(max_time_s / self.dt)
        self.commands = torch.zeros((self.max_timestep, 9))
        self.start_time = 0

    def get_command(self, t):
        timestep = int((t - self.start_time) / self.dt)
        timestep = min(timestep, self.max_timestep - 1)
        return self.commands[timestep, :]

    def get_buttons(self):
        return 0

    def reset(self, reset_time):
        self.start_time = reset_time


class RCControllerProfile(CommandProfile):
    def __init__(self, dt, state_estimator, x_scale=1.0, y_scale=1.0, yaw_scale=1.0, probe_vel_multiplier=1.0):
        super().__init__(dt)
        self.state_estimator = state_estimator
        self.x_scale = x_scale
        self.y_scale = y_scale
        self.yaw_scale = yaw_scale

        self.probe_vel_multiplier = probe_vel_multiplier

        self.triggered_commands = {i: None for i in range(4)}  # command profiles for each action button on the controller
        self.currently_triggered = [0, 0, 0, 0]
        self.button_states = [0, 0, 0, 0]

    def get_command(self, t):

        command = self.state_estimator.get_command()
        # command[0] = command[0] * self.x_scale
        # command[1] = command[1] * self.y_scale
        # command[2] = command[2] * self.yaw_scale
        
        return command

    def get_buttons(self):
        return self.state_estimator.get_buttons()

