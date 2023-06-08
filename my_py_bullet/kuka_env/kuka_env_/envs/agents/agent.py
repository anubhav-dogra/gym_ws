import numpy as np
import pybullet as p

class Agent:
    def __init__(self):
        self.base=-1 #base index is -1 in pybullet
        self.body = None
        self.upper_limits = None
        self.ik_lower_limits = None
        self.ik_upper_limits = None
        self.ik_joint_names = None


    # env is gym environment, body ? 
    def init_env(self, body, env, indices=None):
        self.init(body, env.id, env.np_random, indices)


    def control(self, indices, target_angles, gains, forces):
        if type(gains) in [int, float]:
            gains = [gains]*len(indices)
        if type(forces) in [int,float]:
            forces = [forces]*len(indices)
        p.setJointMotorControlArray(self.body, jointIndices=indices,controlMode=p.POSITION_CONTROL, targetPositions=target_angles, positionGains=gains, forces=forces, physicsClientId=self.id)
