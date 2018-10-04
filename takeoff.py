import numpy as np
from physics_sim import PhysicsSim


class TakeOff():
    """Task TakeOff that defines the goal and provides feedback to the agent."""

    def __init__(self,
                 init_pose=None,
                 init_velocities=None,
                 init_angle_velocities=None,
                 runtime=5.,
                 target_position=None):
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime)
        self.action_repeat = 3

        self.state_size = self.action_repeat * 6
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4

        # Goal
        if target_position is None:
            print("Setting default init pose")
        self.target_position = target_position if target_position is not None else np.array([0., 0., 10.])

    def get_reward(self):
        """Uses current pose of sim to return reward."""
        reward = np.tanh(1 - 0.003 * (abs(self.sim.pose[:3] - self.target_position))).sum()
        return reward

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds)  # update the sim pose and velocities
            reward += self.get_reward()
            pose_all.append(self.sim.pose)
            if done:
                reward += 10
        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat)
        return state
