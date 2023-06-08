import os
from typing import Any, SupportsFloat
import numpy as np
import pybullet as p
import pybullet_data
import gymnasium as gym
from gymnasium import spaces


class iiwaEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self):
        self.state = self.init_state()
        self.step_count = 0

    def init_state(self):
        p.conect(p.GUI)
        p.resetSimulation()
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0,0,-9.81)
        self.iiwaId = p.loadURDF(os.path.join('assets', 'urdf', 'iiwa14_rs_scanner_v1.urdf'),
                                basePosition=[0,0,0],
                                baseOrientation=[0,0,0,1],
                                useFixedBase=True,
                                flags=p.URDF_USE_SELF_COLLISION)
        p.loadURDF("plane.urdf",[0,0,0],[0,0,0,1])
        eef_pose = p.getLinkState(self.iiwaId, 10)[0]
        obs = np.array([eef_pose]).flatten
        return obs
    
    def reset(self):
        p.disconnect()
        self.state=self.init_state()
        self.step_count=0

    def step(self,action):
        self.step_count +=1
        p.setJointMotorControlArray(self.iiwaId, [4], p.POSITION_CONTROL, [action])
        p.stepSimulation()
        eef_pose = p.getLinkState(self.iiwaId,10)[0]
        if (self.step_count >= 50):
            self.reset()
            eef_pose = p.getLinkState(self.iiwaId,10)[0]
            obs = np.array([eef_pose]).flatten
            self.state = obs
            reward = -1 #reward arbitrary
            done = True
            return reward, done
        
        obs = np.array([eef_pose]).flatten
        self.state = obs
        done = False
        reward = -1 #reward arbitrary

        return reward, done
    
# env = iiwaEnv()
# for step in range(500):
#     action = np.random.uniform(0,1)
#     a,b = env.step(action)
#     print(env.state)
#     p.stepSimulation()


        

