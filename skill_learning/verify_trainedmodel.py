import torch
import json
import matplotlib.pyplot as plt
import numpy as np
from network_ import MarkerNet
import cv2


# Load the model
model = MarkerNet()
model.load_state_dict(torch.load('/home/terabotics/gym_ws/skill_learning/imitation_learning_model.pth', weights_only=True))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()  # Set the model to evaluation mode

# Load the test/validation data
with open('/home/terabotics/gym_ws/skill_learning/extracted_data/demo2_dataset/data.json', 'r') as f:
    test_data = json.load(f)

# Extract pose data
poses = np.array([data['pose'] for data in test_data])
actions = np.array([data['action'] for data in test_data])

# Compute min and max of pose and action data for normalization
# pose_min = np.min(poses, axis=0)
# pose_max = np.max(poses, axis=0)

# action_min = np.min(actions, axis=0)
# action_max = np.max(actions, axis=0)

pose_mean = np.mean(poses, axis=0)
pose_std = np.std(poses, axis=0)

action_mean = np.mean(actions, axis=0)
action_std = np.std(actions, axis=0)
# Normalization functions
# def normalize_pose(pose, pose_min, pose_max):
#     return (pose - pose_min) / (pose_max - pose_min)

# def denormalize_pose(pose_normalized, pose_min, pose_max):
#     """ Denormalizes the pose using min-max scaling. """
#     return pose_normalized * (pose_max - pose_min) + pose_min

# def normalize_action(action, action_min, action_max):
#     return (action - action_min) / (action_max - action_min)

# def denormalize_action(action_normalized, action_min, action_max):
#     """ Denormalizes the action using min-max scaling. """
#     return action_normalized * (action_max - action_min) + action_min

def standardize_pose(pose, pose_mean, pose_std):
    """Standardizes the pose using mean and standard deviation."""
    return (pose - pose_mean) / pose_std

def standardize_action(action, action_mean, action_std):
    """Standardizes the action using mean and standard deviation."""
    return (action - action_mean) / action_std

def destandardize_pose(pose_standardized, pose_mean, pose_std):
    """Reverts pose from standardized form to the original scale."""
    return pose_standardized * pose_std + pose_mean

def destandardize_action(action_standardized, action_mean, action_std):
    """Reverts action from standardized form to the original scale."""
    return action_standardized * action_std + action_mean
# Image preprocessing
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))  # Resizing image to fit the model input
    image_tensor = torch.Tensor(img).permute(2, 0, 1).unsqueeze(0)  # Convert to CxHxW and add batch dimension
    return image_tensor

# Pose preprocessing
def preprocess_pose(pose):
    pose_tensor = torch.Tensor(pose).unsqueeze(0)  # Add batch dimension
    pose_normalized = torch.tensor(standardize_pose(pose_tensor.numpy(), pose_mean, pose_std)).float()
    return pose_normalized

# Action preprocessing
def preprocess_action(action):
    action_tensor = torch.Tensor(action).unsqueeze(0)  # Add batch dimension
    action_normalized = torch.tensor(standardize_action(action_tensor.numpy(), action_mean,action_std)).float()
    return action_normalized

# Marker visibility preprocessing
def preprocess_marker_visible(marker_visible):
    return torch.tensor(marker_visible).float().unsqueeze(0)

# Initialize arrays to store predicted actions and poses
predicted_actions = []
predicted_poses = []

# Run predictions on the test data
for item in test_data:
    image = preprocess_image(item['image_path']).to(device)
    pose = preprocess_pose(item['pose']).to(device)
    action = preprocess_action(item['action']).to(device)
    marker_visible = preprocess_marker_visible(item['marker_visible']).to(device)

    # Reshape marker_visible if necessary to match the batch dimension
    marker_visible = marker_visible.unsqueeze(1)  # Ensure marker_visible is 2D with batch dimension

    with torch.no_grad():
        # Get the predicted action from the model
        predicted_action = model(image, pose, marker_visible)
        
        # Move the predicted action to the CPU and append to the list
        # predicted_actions.append(predicted_action.cpu().numpy())

        # Denormalize predicted action and pose for accurate calculations
        pred_action_denorm = destandardize_action(predicted_action.cpu().numpy(), action_mean, action_std)
        pred_action_denorm = np.squeeze(pred_action_denorm)

        pose_denorm = destandardize_pose(pose.cpu().numpy(), pose_mean, pose_std)
        pose_denorm = np.squeeze(pose_denorm)

        # Apply the predicted action to the pose
        new_pose = pose_denorm + pred_action_denorm
        
        # Append the new pose (already a NumPy array) to the predicted_poses list
        predicted_poses.append(new_pose)
        predicted_actions.append(pred_action_denorm)

# Convert predicted poses and actions to NumPy arrays for visualization
predicted_poses = np.array(predicted_poses).squeeze()
predicted_actions = np.array(predicted_actions).squeeze()

# Visualize the predicted end-effector positions (x, y, z)
plt.plot(predicted_actions[:, 0], label='Predicted X')
plt.plot(predicted_actions[:, 1], label='Predicted Y')
plt.plot(predicted_actions[:, 2], label='Predicted Z')
plt.title('Predicted End-Effector Positions')
plt.xlabel('Time Step')
plt.ylabel('Position')
plt.legend()
plt.show()

# Compare with ground truth poses
ground_truth_actions = np.array([item['pose'] for item in test_data])

plt.plot(predicted_poses[:, 0], label='Predicted X')
plt.plot(ground_truth_actions[:, 0], label='Ground Truth X', linestyle='--')
plt.plot(predicted_poses[:, 1], label='Predicted Y')
plt.plot(ground_truth_actions[:, 1], label='Ground Truth Y', linestyle='--')
plt.plot(predicted_poses[:, 2], label='Predicted Z')
plt.plot(ground_truth_actions[:, 2], label='Ground Truth Z', linestyle='--')
plt.title('Predicted vs Ground Truth End-Effector Positions')
plt.xlabel('Time Step')
plt.ylabel('Position')
plt.legend()
plt.grid()
# plt.rcParams.update({'font.size': 32})
plt.show()
