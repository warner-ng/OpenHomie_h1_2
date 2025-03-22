import glob
import pickle as pkl
import lcm
import sys

from utils.deployment_runner import DeploymentRunner
from envs.lcm_agent import LCMAgent
from utils.cheetah_state_estimator import StateEstimator
from utils.command_profile import *
import onnxruntime as ort

import pathlib
import os

# os.environ["LCM_DEFAULT_URL"] = "eth0"

lc = lcm.LCM("udpm://239.255.76.67:7667?ttl=255")

def load_and_run_policy():
    # load trained policy
    ckpt_path = "/home/unitree/deploy/deploy.onnx"

    se = StateEstimator(lc)

    control_dt = 1/50
    command_profile = RCControllerProfile(dt=control_dt, state_estimator=se)

    hardware_agent = LCMAgent(se, command_profile)
    se.spin()

    from envs.history_wrapper import HistoryWrapper
    hardware_agent = HistoryWrapper(hardware_agent)

    policy = load_onnx_policy(ckpt_path)

    deployment_runner = DeploymentRunner(se=None)
    deployment_runner.add_control_agent(hardware_agent, "hardware_closed_loop")
    deployment_runner.add_policy(policy)
    deployment_runner.add_command_profile(command_profile)

    max_steps = 10000000
    print(f'max steps {max_steps}')

    deployment_runner.run(max_steps=max_steps)


def load_onnx_policy(path):
    model = ort.InferenceSession(path)
    def run_inference(input_tensor):
        ort_inputs = {model.get_inputs()[0].name: input_tensor.cpu().numpy()}
        ort_outs = model.run(None, ort_inputs)
        return torch.tensor(ort_outs[0], device="cuda:0")
    return run_inference


if __name__ == '__main__':
    load_and_run_policy()
