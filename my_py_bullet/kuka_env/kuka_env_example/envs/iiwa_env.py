import os
import math
import time
from typing import Any, SupportsFloat
import numpy as np
import pybullet as p
import pybullet_data
import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding


largeValObservation = 100
RENDER_HEIGHT = 720
RENDER_WIDTH = 960

class iiwaEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self):
        self.useSimulation = 1
        self.useInverseKinematics = 1
        self.useNullSpace = 1
        self.useOrientation = 1
        self.eef_index = 10
        self.observations = []
        self.cam_dist = 1.3
        self.cam_yaw = 180
        self.cam_pitch = -40
        self.maxForce = 200
        self.maxVelocity=0.35
        actionRepeat=1
        renders=True
        maxSteps=100000
        self._maxSteps = maxSteps
        self._actionRepeat = actionRepeat
        self._renders = renders
        self.terminated = 0
        self._timestep = 1. / 240.
        self.target_pos = np.array([-0.62, 0.2, 0.15])
        #lower limits for null space
        self.lowerlimits = [-.967, -2, -2.96, 0.19, -2.96, -2.09, -3.05]
        #upper limits for null space
        self.upperlimits = [.967, 2, 2.96, 2.29, 2.96, 2.09, 3.05]
        #joint ranges for null space
        self.jointranges = [5.8, 4, 5.8, 4, 5.8, 4, 6]
        #restposes for null space
        self.resetposes = [0, 0, 0, 0.5 * math.pi, 0, -math.pi * 0.5 * 0.66, 0]
        #joint damping coefficents
        self.jointdampings = [
            0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001,
            0.00001]
        self.step_count = 0
        # if self._renders:
        #     cid = p.connect(p.SHARED_MEMORY)
        # if (cid < 0):
        #     cid = p.connect(p.GUI)
        #     p.resetDebugVisualizerCamera(1.5, 180, -40, [-1, -0.0, 0.33])
        # else:
        p.connect(p.DIRECT)
        # p.connect(p.DIRECT)
        self.state = self.init_state()



    def init_state(self):

        p.resetSimulation()
        path = "/home/anubhav/gym_ws/my_py_bullet/kuka_env/kuka_env_example/envs"
        self.iiwaId = p.loadURDF(os.path.join(path,'agents','assets','iiwa','urdf','iiwa14_rs_scanner_v1.urdf'),
                                basePosition=[0,0,0],
                                baseOrientation=[0,0,0,1],
                                useFixedBase=True,
                                flags=p.URDF_USE_SELF_COLLISION)
        p.resetBasePositionAndOrientation(self.iiwaId, [0.00000, 0.000000, 0.00000],
                                      [0.000000, 0.000000, 0.000000, 1.000000])
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.loadURDF("plane.urdf",[0,0,0],[0,0,0,1])
        p.setTimeStep(self._timestep)
        p.setGravity(0,0,-9.81)
        self.jointPositions = [
        0.00, -2.464, 1.486, 1.405, 1.393, 1.515, -1.747, 0.842, 0.0, 0.000000, -0.00000 ]
        self.numJoints = p.getNumJoints(self.iiwaId)
        maxForce = 200
        for jointIndex in range(self.numJoints):
            p.resetJointState(self.iiwaId, jointIndex, self.jointPositions[jointIndex])
            p.setJointMotorControl2(self.iiwaId,
                                    jointIndex,
                                    p.POSITION_CONTROL,
                                    targetPosition=self.jointPositions[jointIndex],
                                    force=maxForce)
        self.endEffectorPos = [-0.6203, -0.000, 0.1505]
        self.endEffectorAngle = 0
        self.motorNames = []
        self.motorIndices = []
        for i in range(self.numJoints):
            jointInfo = p.getJointInfo(self.iiwaId, i)
            qIndex = jointInfo[3]
            if qIndex > -1:
                # print("motorname")
                # print(jointInfo[1])
                self.motorNames.append(str(jointInfo[1]))
                self.motorIndices.append(i)

        observationDim = self.getObservationDimension()
        observation_high = np.array([largeValObservation]*observationDim)
        action_dim = 3
        self._action_bound = 1
        action_high = np.array([self._action_bound]*action_dim)
        self.action_space = spaces.Box(-action_high, action_high)
        self.observation_space = spaces.Box(-observation_high, observation_high, dtype=np.float64)
        p.stepSimulation()
        self.observation = self.getObservation()
        return np.array(self.observation)

        # eef_pose = p.getLinkState(self.iiwaId, 10)[0]
        # obs = np.array([eef_pose]).flatten
        # return observations
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.terminated = 0
        # p.disconnect()
        self.state=self.init_state()
        self.step_count=0
        self.observation = self.getObservation()
        info = self._get_info()
        return np.array(self.observation), info
    

    def _get_info(self):
        return {"distance": np.linalg.norm(self.target_pos, ord=1)}
    
    def seed(self, seed = None):
        self.np_random, seed  = seeding.np_random(seed)
        return [seed]
    
    def step(self,action):
        action_ = (self.target_pos - self.getObservation()[0])
        action_ = action_/np.linalg.norm(action_) #normalizing 
        # print(action_)
        # dv = 0.005
        # dx = action[0]*dv
        # dy = action[1]*dv
        # dz = action[2]*dv
        # for i in range(self._actionRepeat):
        self.applyAction(action)
        p.stepSimulation()
        # if self._termination():  #defined function
            # break
        self.step_count+=1
        if self._renders: #defined function
            time.sleep(self._timestep)
        self.observation = self.getObservation()
        # print("self.step_count")
        # print(self.step_count)

        terminated  = self._termination()
        # print(terminated)
        # npaction = np.array([action[3]])  #only penalize rotation until learning works well [action[0],action[1],action[3]])
        # actioncost = np.linalg.norm(action)
        # print("action_cost")
        # print(actioncost)
    
        reward = self._reward()
        # print("reward")
        # print(reward)
        
        truncated = False
        info = {"is_success": terminated}
        return np.array(self.observation), reward, terminated, truncated, info
    
        ## simple example of action (random)
        # self.step_count +=1
        # p.setJointMotorControlArray(self.iiwaId, [4], p.POSITION_CONTROL, [action])
        # p.stepSimulation()
        # eef_pose = p.getLinkState(self.iiwaId,self.eef_index)[0]
        # if (self.step_count >= 50):
        #     self.reset()
        #     eef_pose = p.getLinkState(self.iiwaId,self.eef_index)[0]
        #     obs = np.array([eef_pose]).flatten
        #     self.state = obs
        #     reward = -1 #reward arbitrary
        #     done = True
        #     return reward, done
        
        # obs = np.array([eef_pose]).flatten
        # self.state = obs
        # done = False
        # reward = -1 #reward arbitrary

        # return reward, done
    
    def render(self, mode="rgb_array", close=False):
        if mode!="rgb_array":
            return np.array([])
        
        base_pos, orn = p.getBasePositionAndOrientation(self.iiwaId)
        view_matrix = p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=base_pos,
                                                          distance=self.cam_dist,
                                                          yaw = self.cam_yaw,
                                                          pitch = self.cam_pitch,
                                                          roll=0,
                                                          upAxisIndex=2)
        proj_matrix = p.computeProjectionMatrixFOV(fov=60,
                                                   aspect=float(RENDER_WIDTH)/RENDER_HEIGHT,
                                                   nearVal=0.1,
                                                   farVal=100.0)
        
        (_, _, px, _, _) = p.getCameraImage(width=RENDER_WIDTH,
                                                height=RENDER_HEIGHT,
                                                viewMatrix=view_matrix,
                                                projectionMatrix=proj_matrix,
                                                renderer=p.ER_BULLET_HARDWARE_OPENGL)
        
        rgb_array = np.array(px, dtype=np.uint8)
        rbg_array = np.reshape(rgb_array, (RENDER_HEIGHT, RENDER_WIDTH, 4))

        rgb_array = rgb_array[:,:,3]
        return rgb_array

    ####################################################################################
    ######################## user Defined functions ####################################
    ####################################################################################
    
    def getActionDimension(self):
        if (self.useInverseKinematics):
            return len(self.motorIndices)
        return 6 #position and rpy of eef

    def getObservationDimension(self):
        return len(self.getObservation())
    
    def getObservation(self):
        self.observation = []
        state = p.getLinkState(self.iiwaId, self.eef_index)
        pos = state[0]
        orn = state[1]
        euler = p.getEulerFromQuaternion(orn)

        self.observation.extend(list(pos))
        self.observation.extend(list(euler))

        return self.observation
    
    def applyAction(self,motorCommands):
        if(self.useInverseKinematics):
            dx= motorCommands[0]
            dy= motorCommands[1]
            dz= motorCommands[2]
            # da= motorCommands[3]

            state = p.getLinkState(self.iiwaId, self.eef_index)
            actualEndEffectorPos = state[0]
            # print("actualEndEffectorPos",actualEndEffectorPos)

            ## target positions I guess
            self.endEffectorPos[0]= actualEndEffectorPos[0] + dx
            # self.endEffectorPos[0]=self.endEffectorPos[0] + dx
            # if (self.endEffectorPos[0] > -0.75):
            #     self.endEffectorPos[0] = -0.75
            # if (self.endEffectorPos[0] < -0.75):
            #     self.endEffectorPos[0] = -0.75
            # self.endEffectorPos[1] = self.endEffectorPos[1] + dy
            self.endEffectorPos[1] = actualEndEffectorPos[1] + dy
            # if (self.endEffectorPos[1] < -0.01):
            #     self.endEffectorPos[1] = -0.01
            # if (self.endEffectorPos[1] > -0.01):
            #     self.endEffectorPos[1] = -0.01
            # self.endEffectorPos[2] = self.endEffectorPos[2] + dz
            self.endEffectorPos[2] = actualEndEffectorPos[2] + dz
            # if (self.endEffectorPos[2] < 0.04):
            #     self.endEffectorPos[2] = 0.04
            # if (self.endEffectorPos[2] > 0.04):
            #     self.endEffectorPos[2] = 0.04
            # self.endEffectorAngle = self.endEffectorAngle + da
            pos =  self.endEffectorPos
            # print(pos)
            # print(pos)
            orn = p.getQuaternionFromEuler([math.pi, 0, -math.pi/2])  # -math.pi,yaw])
            if(self.useNullSpace==1):
                if(self.useOrientation==1):

                    jointPoses = p.calculateInverseKinematics(self.iiwaId,
                                                              self.eef_index,
                                                              pos,
                                                              orn,
                                                              lowerLimits=self.lowerlimits,
                                                              upperLimits=self.upperlimits,
                                                              jointRanges=self.jointranges,
                                                              restPoses=self.resetposes)    
                else:
                    jointPoses = p.calculateInverseKinematics(self.iiwaId,
                                                              self.eef_index,
                                                              pos,
                                                              lowerLimits=self.lowerlimits,
                                                              upperLimits=self.upperlimits,
                                                              jointRanges=self.jointranges,
                                                              restPoses=self.resetposes)
            else:
                if(self.useOrientation==1):
                    jointPoses = p.calculateInverseKinematics(self.iiwaId,
                                                              self.eef_index,
                                                              pos,
                                                              orn,
                                                              jointDamping=self.jointdampings)
                else:
                    jointPoses = p.calculateInverseKinematics(self.iiwaId,
                                                              self.eef_index,
                                                              pos)
            # print("jointPoses")
            # print(jointPoses)
            # print("self.eef_index")
            # print(self.eef_index)
            # print(self.numJoints)
            if (self.useSimulation):
                for i in range(7):
                #print(i)
                    p.setJointMotorControl2(bodyUniqueId=self.iiwaId,
                                            jointIndex=i,
                                            controlMode=p.POSITION_CONTROL,
                                            targetPosition=jointPoses[i],
                                            targetVelocity=0,
                                            force=self.maxForce,
                                            maxVelocity=self.maxVelocity,
                                            positionGain=0.3,
                                            velocityGain=1)
            else:
                #idk this (Anubhav)
                #reset the joint state (ignoring all dynamics, not recommended to use during simulation)
                for i in range(self.numJoints):
                    p.resetJointState(self.iiwaId, i, jointPoses[i])
                    #fingers
                    p.setJointMotorControl2(self.iiwaId,
                                            7,
                                            p.POSITION_CONTROL,
                                            targetPosition=self.endEffectorAngle,
                                            force=self.maxForce)


        else:
            for action in range(len(motorCommands)):
                    motor = self.motorIndices[action]
                    p.setJointMotorControl2(self.iiwaId,
                                            motor,
                                            p.POSITION_CONTROL,
                                            targetPosition=motorCommands[action],
                                            force=self.maxForce)
  
    def _termination(self):
        #print (self._kuka.endEffectorPos[2])
        state = p.getLinkState(self.iiwaId, self.eef_index)
        actualEndEffectorPos = state[0]

        #print("self._envStepCounter")
        #print(self._envStepCounter)
        if (self.step_count > self._maxSteps):
            self.observation = self.getObservation()
            print("*********************terminated observation*********************")
            return True
        
        if np.linalg.norm(actualEndEffectorPos - self.target_pos) < 0.01:
            self.terminated=1
            print("pos error")
            print(np.linalg.norm(actualEndEffectorPos - self.target_pos))
            self._observation = self.getObservation()
            return True
        
        if (actualEndEffectorPos[2] < 0.05):
            return True
        
        if (actualEndEffectorPos[2] > 0.3 or actualEndEffectorPos[1] < -0.1):
            return True
        return False

    def _reward(self):
        state = p.getLinkState(self.iiwaId, self.eef_index)
        tool_pos = state[0]
        reward1=0
        reward_distance = -np.linalg.norm(self.target_pos - tool_pos) # Penalize distances away from target
        if (np.linalg.norm(self.target_pos - tool_pos)) < 0.1:
            reward1 = 100
        if (np.linalg.norm(self.target_pos - tool_pos)) < 0.01:
            reward_distance = 1000 + reward_distance
        # print("reward_distance")
        # print(reward_distance)
        return reward_distance+reward1

# env = iiwaEnv()
# for step in range(500):
#     action = np.random.uniform(0,1)
#     a,b = env.step(action)
#     print(env.state)
#     p.stepSimulation()


        

