import gymnasium
import numpy as np
import pybullet as p

class KukaEnv(gymnasium.Env):
    metadata = {'render.modes':['human']}

    def __init__(self):
        pass

    def step(self, action):
        eef_pose = self.body.id

    def reset(self):
        p.resetSimulation(physicsClientId=self.id)
        if not self.gui:
            # Reconnect the physics engine to forcefully clear memory when running long training scripts
            self.disconnect()
            self.id = p.connect(p.DIRECT)
            # self.util = Util(self.id, self.np_random)
        if self.gpu:
            self.util.enable_gpu()
        # Configure camera position
        p.resetDebugVisualizerCamera(cameraDistance=1.75, cameraYaw=-25, cameraPitch=-45, cameraTargetPosition=[-0.2, 0, 0.4], physicsClientId=self.id)
        p.configureDebugVisualizer(p.COV_ENABLE_MOUSE_PICKING, 0, physicsClientId=self.id)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=self.id)
        p.setTimeStep(self.time_step, physicsClientId=self.id)
        # Disable real time simulation so that the simulation only advances when we call stepSimulation
        p.setRealTimeSimulation(0, physicsClientId=self.id)
        p.setGravity(0, 0, self.gravity, physicsClientId=self.id)
        # self.agents = []
        # self.last_sim_time = None
        # self.iteration = 0
        # self.forces = []
        # self.task_success = 0


    # def render(self):
    #     if not self.gui:
    #         self.gui = True
    #         if self.id is not None:
    #             self.disconnect()
    #         try:
    #              self.width = get_monitors()[0].width
    #              self.height = get_monitors()[0].height
    #         except Exception as e:
    #             self.width = 1920
    #             self.height = 1080
    #         self.id = p.connect(p.GUI, options='--background_color_red=0.8 --background_color_green=0.9 --background_color_blue=1.0 --width=%d --height=%d' % (self.width, self.height))
    #         self.util = Util(self.id, self.np_random)

    def close(self):
        pass

    def seed(self, seed=None): 
        pass   