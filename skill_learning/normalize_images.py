import cv2
import numpy as np
import os
import glob
from sklearn.model_selection import train_test_split

image_dir = "/home/terabotics/gym_ws/skill_learning/extracted_data/demo1/"
processed_images = []

pose_dir = "/home/terabotics/gym_ws/skill_learning/extracted_data/demo1/"
processed_poses = []

for img_file in sorted(glob.glob(os.path.join(image_dir, '*.png'))):
    img_path = os.path.join(image_dir, img_file)
    img = cv2.imread(img_path)
    img_resized = cv2.resize(img, (64, 64))
    img_normalized = img_resized / 255.0
    processed_images.append(img_normalized)

processed_images = np.array(processed_images)

for pose_file in sorted(glob.glob(os.path.join(pose_dir, '*.npy'))):
    pose_path = os.path.join(pose_dir, pose_file)
    pose = np.load(pose_path)
    processed_poses.append(pose)

processed_poses = np.array(processed_poses)
# for i, pose in enumerate(processed_poses):
#     print(f"Pose {i}: Shape = {np.array(pose).shape}")

# np.save("/home/terabotics/gym_ws/skill_learning/extracted_data/demo1/processed_images.npy", np.array(processed_images))
# np.save("/home/terabotics/gym_ws/skill_learning/extracted_data/demo1/processed_poses.npy", np.array(processed_poses))

paired_data = list(zip(processed_images, processed_poses))

train_data, test_data = train_test_split(paired_data, test_size=0.2, random_state=42)
train_data, val_data = train_test_split(train_data, test_size=0.25, random_state=42)  # 0.25 * 0.8 = 0.2

# Unpack data into images and poses
train_images, train_poses = zip(*train_data)
val_images, val_poses = zip(*val_data)
test_images, test_poses = zip(*test_data)

# Convert back to numpy arrays
train_images = np.array(train_images)
train_poses = np.array(train_poses)
val_images = np.array(val_images)
val_poses = np.array(val_poses)
test_images = np.array(test_images)
test_poses = np.array(test_poses)


# Assuming 'train_images', 'train_poses', 'val_images', 'val_poses', 'test_images', 'test_poses' are numpy arrays

# Define the path where you want to save the dataset
dataset_path = '/home/terabotics/gym_ws/skill_learning/extracted_data/dataset1.npz'
np.savez(
    dataset_path,
    train_images=train_images, train_poses=train_poses,
    val_images=val_images, val_poses=val_poses,
    test_images=test_images, test_poses=test_poses
)

print(f"Dataset saved at {dataset_path}")