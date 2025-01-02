import rospy
import cv2
import rosbag
from cv_bridge import CvBridge
import numpy as np
import os, sys, json


bridge = CvBridge() 

bag = rosbag.Bag("/home/terabotics/gym_ws/skill_learning/demos/demo2.bag")

synced_data = []
current_pose = None
current_image = None

for topic, msg, t in bag.read_messages(topics=['/camera/color/image_raw', '/tool_link_ee_pose']):
    if topic == "/tool_link_ee_pose":
        # if isinstance(msg, TransformStamped):
        current_pose = [msg.transform.translation.x, msg.transform.translation.y, msg.transform.translation.z,
                      msg.transform.rotation.x, msg.transform.rotation.y, msg.transform.rotation.z, msg.transform.rotation.w]

    if topic == "/camera/color/image_raw":
        # if isinstance(msg, Image):
        current_image = bridge.imgmsg_to_cv2(msg, "bgr8")

    if current_pose is not None and current_image is not None:
        synced_data.append((current_image, current_pose))
        current_pose = None  # Reset to ensure the next message pair is fresh
        current_image = None  # Reset after appending the pair

bag.close()

print(f"Loaded {len(synced_data)} synchronized image-pose pairs.")

#############################################################################
#calculate action pose difference
def calculate_actions(synced_data):
    dataset = []

    for i in range(1, len(synced_data)):
        img, pose = synced_data[i]
        _, prev_pose = synced_data[i - 1]
        
        # Calculate pose difference (action)
        delta_pose = [pose[j] - prev_pose[j] for j in range(7)]
        
        # Store the image, current pose, and action (pose difference)
        dataset.append((img, prev_pose, delta_pose))
    
    return dataset

# Create the dataset
dataset = calculate_actions(synced_data)
print(f"Generated dataset with {len(dataset)} samples.")

#############################################################################
# Save the dataset
# Create directory to store images
os.makedirs("/home/terabotics/gym_ws/skill_learning/extracted_data/demo2_dataset/images", exist_ok=True)

# Initialize data list
data_list = []

for idx, (img, pose, action) in enumerate(dataset):
    # Save image
    image_path = f"/home/terabotics/gym_ws/skill_learning/extracted_data/demo2_dataset/images/image_{idx}.png"
    cv2.imwrite(image_path, img)
    
    # Prepare data entry
    data_entry = {
        "image_path": image_path,
        "pose": pose,
        "action": action
    }
    
    data_list.append(data_entry)

# Save data as JSON
with open("/home/terabotics/gym_ws/skill_learning/extracted_data/demo2_dataset/data.json", "w") as json_file:
    json.dump(data_list, json_file, indent=4)

print("Dataset saved successfully.")
