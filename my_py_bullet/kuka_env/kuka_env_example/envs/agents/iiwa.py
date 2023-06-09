import os
import numpy as np
import pybullet as p
from .robot import Robot


class iiwa(Robot):
    def __init__(self):
        joint_indices = [0,1,2,3,4,5,6]
        tool_link_ee = 9
        # base_pose_offset = [0, 0, 0]
        # ee_orient_rpy = [0, 0, 0]   

        super(iiwa, self).__init__(joint_indices,tool_link_ee)
                                #    base_pose_offset, ee_orient_rpy)
    
    def init(self, directory, id, np_random, fixed_base=True):
        self.body = p.loadURDF(os.path.join(directory, 'urdf', 'iiwa14_rs_scanner_v1.urdf'),
                                useFixedBase = fixed_base,
                                basePosition=[0,0,0],
                                flags=p.URDF_USE_SELF_COLLISION,
                                physicsClientId=id)
        super(iiwa,self).init(self.body, id, np_random)