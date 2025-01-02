import matplotlib.pyplot as plt
import torch
import json
from network_ import SequenceMarkerNet
import numpy as np
import cv2

model = SequenceMarkerNet()
model.load_state_dict(torch.load('/home/terabotics/gym_ws/skill_learning/sq_model.pth', weights_only=True))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

# Load the test/validation data
with open('/home/terabotics/gym_ws/skill_learning/extracted_data/combined_dataset/train_data_sq.json', 'r') as f:
    test_data = json.load(f)

poses = np.array([data['pose'] for data in test_data])
actions = np.array([data['action'] for data in test_data])


pose_mean = np.mean(poses, axis=0)
pose_std = np.std(poses, axis=0)

action_mean = np.mean(actions, axis=0)
action_std = np.std(actions, axis=0)

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

predicted_actions = []
predicted_poses = []

for item in test_data:
    image = preprocess_image(item['image_path']).unsqueeze(0).to(device)  # Add batch dimension
    pose = preprocess_pose(item['pose']).unsqueeze(0).to(device)  # Add batch dimension
    action = preprocess_action(item['action']).unsqueeze(0).to(device)  # Add batch dimension
    marker_visible = preprocess_marker_visible(item['marker_visible']).unsqueeze(0).to(device)  # Add batch dimension

    # Ensure the inputs are shaped correctly
    image = image.unsqueeze(1)  # Add sequence dimension if necessary
    pose = pose.unsqueeze(1)  # Add sequence dimension if necessary
    marker_visible = marker_visible.unsqueeze(1)  # Add sequence dimension if necessary

    # Remove the extra dimension if present
    if image.dim() == 6:
        image = image.squeeze(1)  # Remove the extra sequence dimension if it's not needed
    if pose.dim() == 3:
        pose = pose.unsqueeze(1)  # Add the sequence dimension if it's missing
    if marker_visible.dim() == 2:
        marker_visible = marker_visible.unsqueeze(1)  # Add the sequence dimension if it's missing

    with torch.no_grad():
        outputs = model(image, pose, marker_visible)
        # predicted_actions = outputs.cpu().numpy()


        pred_action_denorm = destandardize_action(outputs.cpu().numpy(), action_mean, action_std)
        pred_action_denorm = np.squeeze(pred_action_denorm)

        pose_denorm = destandardize_pose(pose.cpu().numpy(), pose_mean, pose_std)
        pose_denorm = np.squeeze(pose_denorm)

        # Apply the predicted action to the pose
        new_pose = pose_denorm + pred_action_denorm
        
        # Append the new pose (already a NumPy array) to the predicted_poses list
        predicted_poses.append(new_pose)
        predicted_actions.append(pred_action_denorm)

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