import math
import select
import threading
import time

import numpy as np

from lcm_types.body_control_data_lcmt import body_control_data_lcmt
from lcm_types.rc_command_lcmt import rc_command_lcmt
from lcm_types.state_estimator_lcmt import state_estimator_lcmt
from lcm_types.arm_action_lcmt import arm_action_lcmt
from lcm_types.command_lcmt import command_lcmt
import lcm
import os
def get_rpy_from_quaternion(q):
    w, x, y, z = q
    r = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x ** 2 + y ** 2))
    p = np.arcsin(2 * (w * y - z * x))
    y = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y ** 2 + z ** 2))
    return np.array([r, p, y])


def get_rotation_matrix_from_rpy(rpy):
    """
    Get rotation matrix from the given quaternion.
    Args:
        q (np.array[float[4]]): quaternion [w,x,y,z]
    Returns:
        np.array[float[3,3]]: rotation matrix.
    """
    r, p, y = rpy
    R_x = np.array([[1, 0, 0],
                    [0, math.cos(r), -math.sin(r)],
                    [0, math.sin(r), math.cos(r)]
                    ])

    R_y = np.array([[math.cos(p), 0, math.sin(p)],
                    [0, 1, 0],
                    [-math.sin(p), 0, math.cos(p)]
                    ])

    R_z = np.array([[math.cos(y), -math.sin(y), 0],
                    [math.sin(y), math.cos(y), 0],
                    [0, 0, 1]
                    ])

    rot = np.dot(R_z, np.dot(R_y, R_x))
    return rot


class StateEstimator:
    def __init__(self, lc):

        # reverse legs, from cpp order to isaacgym order
        self.joint_idxs = [0,1,2,3,4,5,6,7,8,9,10,11,12,15,16,17,18,19,20,21,22,23,24,25,26,27,28]


        self.lc = lc
        # os.environ["LCM_DEFAULT_URL"] = "wlan0"
        # self.new_lcm = lcm.LCM("udpm://239.255.76.67:7667?ttl=255")
        self.num_dofs = 27
        self.joint_pos = np.zeros(self.num_dofs+2)
        self.joint_vel = np.zeros(self.num_dofs+2)
        self.arm_actions = np.zeros(14)
        self.euler = np.zeros(3)
        self.R = np.eye(3)
        self.buf_idx = 0
        self.imu_ang_vel = np.zeros(3)
        
        self.left_stick = [0, 0]
        self.right_stick = [0, 0]
        self.right_lower_right_switch = 0
        self.right_lower_right_switch_pressed = 0


        self.init_time = time.time()
        self.received_first_bodydate = False

        self.imu_subscription = self.lc.subscribe("state_estimator_data", self._imu_cb)
        self.bodydate_state_subscription = self.lc.subscribe("body_control_data", self._bodydata_cb)
        self.rc_command_subscription = self.lc.subscribe("rc_command", self._rc_command_cb)
        self.pedal_command_subscription = self.lc.subscribe("pedal_command", self._pedal_command_cb)
        # self.arm_action_subscrition = self.new_lcm.subscribe("arm_action_lcmt", self._arm_action_cb)

        self.body_quat = np.array([0, 0, 0, 1])
        self.smoothing_ratio = 0.2
        self.body_ang_vel = np.zeros(3)
        self.smoothing_length = 12
        self.dt_history = np.zeros((self.smoothing_length, 1))
        self.euler_prev = np.zeros(3)
        self.deuler_history = np.zeros((self.smoothing_length, 3))
        self.timeuprev = time.time()
        self.command = np.zeros(4)
        self.command[3] = 0.74
        

    def get_gravity_vector(self):
        grav = np.dot(self.R.T, np.array([0, 0, -1]))
        return grav


    def get_rpy(self):
        return self.euler

    def get_command(self):

        # cmd_x = 0.6 * self.left_stick[1]
        # cmd_y = -0.5 * self.left_stick[0]

        # cmd_yaw = -0.8 * self.right_stick[0]
        # cmd_height = 0.74 - 0.54 * self.right_stick[1]

        # # if np.abs(cmd_x) < 0.1 and np.abs(cmd_y) < 0.1 and np.abs(cmd_yaw) < 0.1:
        # #     return np.array([0.0, 0.0, 0.0, 0.68])


        return self.command
        # return np.array([cmd_x, cmd_y, cmd_yaw, cmd_height])

    def get_buttons(self):
        return self.right_lower_right_switch

    def get_dof_pos(self):
        return self.joint_pos[self.joint_idxs]

    def get_dof_vel(self):
        return self.joint_vel[self.joint_idxs]

    def get_yaw(self):
        return self.euler[2]

    def get_body_angular_vel(self):
        # self.body_ang_vel = self.smoothing_ratio * np.mean(self.deuler_history / self.dt_history, axis=0) + (1 - self.smoothing_ratio) * self.body_ang_vel
        return self.body_ang_vel

    def get_arm_action(self):
        return self.arm_actions

    def _bodydata_cb(self, channel, data):
        if not self.received_first_bodydate:
            self.received_first_bodydate = True
            print(f"First body data: {time.time() - self.init_time}")

        msg = body_control_data_lcmt.decode(data)
        self.joint_pos = np.array(msg.q)
        self.joint_vel = np.array(msg.qd)

    def _arm_action_cb(self, channel, data):
        msg = arm_action_lcmt.decode(data)
        self.arm_actions = np.array(msg.act)

    def _imu_cb(self, channel, data):
        msg = state_estimator_lcmt.decode(data)

        self.euler = np.array(msg.rpy)

        self.R = get_rotation_matrix_from_rpy(self.euler)
        self.body_ang_vel = np.array(msg.omegaBody)
        # self.deuler_history[self.buf_idx % self.smoothing_length, :] = msg.rpy - self.euler_prev
        # self.dt_history[self.buf_idx % self.smoothing_length] = time.time() - self.timeuprev
        self.timeuprev = time.time()
        self.buf_idx += 1
        self.euler_prev = np.array(msg.rpy)
        
    def _rc_command_cb(self, channel, data):

        msg = rc_command_lcmt.decode(data)

        self.right_lower_right_switch_pressed = ((msg.right_lower_right_switch and not self.right_lower_right_switch) or self.right_lower_right_switch_pressed)

        self.right_stick = msg.right_stick
        self.left_stick = msg.left_stick
        self.right_lower_right_switch = msg.right_lower_right_switch

    def _pedal_command_cb(self, channel, data):
        msg = command_lcmt.decode(data)
        self.command = msg.command

    def poll(self, cb=None):
        t = time.time()
        try:
            while True:
                timeout = 0.01
                rfds, wfds, efds = select.select([self.lc.fileno()], [], [], timeout)
                # nrfds, nwfds, nefds = select.select([self.new_lcm.fileno()], [], [], timeout)
                if rfds:
                    # print("message received!")
                    self.lc.handle()
                    # print(f'Freq {1. / (time.time() - t)} Hz'); t = time.time()
                else:
                    continue
                # if nrfds:
                    # self.new_lcm.handle()
                # else:
                #     continue

        except KeyboardInterrupt:
            pass

    def spin(self):
        self.run_thread = threading.Thread(target=self.poll, daemon=False)
        self.run_thread.start()

    def close(self):
        self.lc.unsubscribe(self.bodydate_state_subscription)


if __name__ == "__main__":
    import lcm

    lc = lcm.LCM("udpm://239.255.76.67:7667?ttl=255")
    se = StateEstimator(lc)
    se.poll()
