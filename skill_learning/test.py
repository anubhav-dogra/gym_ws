import numpy as np
import tensorflow as tf

print("TensorFlow version:", tf.__version__)

# Check GPU availability
print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))


# # Load the dataset
# data = np.load('/home/terabotics/gym_ws/skill_learning/extracted_data/dataset1.npz')

# # Access the data
# train_images = data['train_images']
# train_poses = data['train_poses']
# val_images = data['val_images']
# val_poses = data['val_poses']
# test_images = data['test_images']
# test_poses = data['test_poses']

# print("Dataset loaded successfully")
# print(f"Training images shape: {train_images.shape}")
# print(f"Training poses shape: {train_poses.shape}")