import pybullet as p
import pybullet_data
import time

client = p.connect(p.GUI)
p.setGravity(0,0,-9.81)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
# p.setAdditionalSearchPath("/home/terabotics/gym_ws/my_py_bullet/robots/urdf")
planeId = p.loadURDF("plane.urdf")
robotId = p.loadURDF("/home/terabotics/gym_ws/my_py_bullet/kuka_env/kuka_env_/envs/agents/assets/iiwa/urdf/iiwa14_rs_scanner_v1.urdf", basePosition=[0,0,0], baseOrientation=[0,0,0,1], useFixedBase=True)
position, orientation = p.getBasePositionAndOrientation(robotId)
number_joints = p.getNumJoints(robotId)
eef_pose = p.getLinkState(robotId,10)[0]
print(eef_pose)
print(number_joints)
for joint_number in range(number_joints):
    info = p.getJointInfo(robotId,joint_number)
    print("joint",joint_number)
    print(info,end="\n")

jointPositions = [
        0.00, -2.464, 1.486, 1.405, 1.393, 1.515, -1.747, 0.842, 0.0, 0.000000, -0.00000 ]
numJoints = p.getNumJoints(robotId)
maxForce = 200
for jointIndex in range(numJoints):
    p.resetJointState(robotId, jointIndex, jointPositions[jointIndex])
    p.setJointMotorControl2(robotId,
                            jointIndex,
                            p.POSITION_CONTROL,
                            targetPosition=jointPositions[jointIndex],
                            force=maxForce)
eef_pose = p.getLinkState(robotId,10)[0]
print(eef_pose)
time.sleep(10)
for _ in range(1000):
    p.stepSimulation()




