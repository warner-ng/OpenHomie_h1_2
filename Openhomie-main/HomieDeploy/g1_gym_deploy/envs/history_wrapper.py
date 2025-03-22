# import isaacgym

# assert isaacgym, "import isaacgym before pytorch"
import torch


class HistoryWrapper:
    def __init__(self, env):
        self.env = env

        self.obs_history_length = self.env.num_history_length
        self.num_obs_history = self.obs_history_length * self.env.num_obs
        self.obs_history = torch.zeros(self.env.num_envs, self.num_obs_history, dtype=torch.float,
                                       device=self.env.device, requires_grad=False)

    def step(self, action):
        obs = self.env.step(action)
        self.obs_history = torch.cat((self.obs_history[:, self.env.num_obs:], obs), dim=-1)
        return {'obs': obs, 'obs_history': self.obs_history}

    def get_observations(self):
        obs = self.env.get_observations()
        self.obs_history = torch.cat((self.obs_history[:, self.env.num_obs:], obs), dim=-1)
        return {'obs': obs, 'obs_history': self.obs_history}

    def get_obs(self):
        obs = self.env.get_obs()
        self.obs_history = torch.cat((self.obs_history[:, self.env.num_obs:], obs), dim=-1)
        return {'obs': obs, 'obs_history': self.obs_history}

    def reset(self):
        ret = self.env.reset()
        self.obs_history[:, :] = 0
        return {"obs": ret, "obs_history": self.obs_history}

    def __getattr__(self, name):
        return getattr(self.env, name)
