import json 
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim
from network_ import MarkerNet, SequentialMarkerNet
# from torch.utils.tensorboard import SummaryWriter

class MarkerDataset(Dataset):
    def __init__(self, json_file, transform=None):
        self.data_list = json.load(open(json_file, 'r'))
        self.transform = transform

    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        data_entry = self.data_list[idx]

        img = cv2.imread(data_entry['image_path'])
        img = cv2.resize(img, (224, 224))
        img = torch.tensor(img).permute(2, 0, 1).float()/255.0  # Normalize and reorder to PyTorch format (C, H, W)

        pose= torch.tensor(data_entry['pose']).float()
        pose_normalized = torch.tensor(standardize_pose(pose.numpy(), pose_mean, pose_std)).float()


        action = torch.tensor(data_entry['action']).float()
        action_normalized = torch.tensor(standardize_action(action.numpy(), action_mean, action_std)).float()

        marker_visible = torch.tensor(data_entry['marker_visible']).float() # Marker visibility as a float (0 or 1)

        return img, pose_normalized, action_normalized, marker_visible
    

with open('/home/terabotics/gym_ws/skill_learning/extracted_data/combined_dataset/train_data.json', 'r') as f:
    dataset = json.load(f)

# Extract pose data
poses = np.array([data['pose'] for data in dataset])
actions = np.array([data['action'] for data in dataset])

pose_mean = np.mean(poses, axis=0)
pose_std = np.std(poses, axis=0)

action_mean = np.mean(actions, axis=0)
action_std = np.std(actions, axis=0)
# 1. Compute min and max of pose data for normalization
# pose_min = np.min(poses, axis=0)
# pose_max = np.max(poses, axis=0)

# action_min = np.min(actions, axis=0)
# action_max = np.max(actions, axis=0)

# def normalize_pose(pose, pose_min, pose_max):
#     """ Normalizes the pose using min-max scaling. """
#     return (pose - pose_min) / (pose_max - pose_min)

# def normalize_action(action, action_min, action_max):
#     """ Normalizes the action using min-max scaling. """
#     return (action - action_min) / (action_max - action_min)
def standardize_pose(pose, pose_mean, pose_std):
    """Standardizes the pose using mean and standard deviation."""
    return (pose - pose_mean) / pose_std

def standardize_action(action, action_mean, action_std):
    """Standardizes the action using mean and standard deviation."""
    return (action - action_mean) / action_std


if __name__ == '__main__':

    # Load the dataset
    # data_set = MarkerDataset("/home/terabotics/gym_ws/skill_learning/extracted_data/demo1_dataset/data.json")
    # data_loader = DataLoader(data_set, batch_size=32, shuffle=True)

    train_dataset = MarkerDataset('/home/terabotics/gym_ws/skill_learning/extracted_data/combined_dataset/train_data.json')
    val_dataset = MarkerDataset('/home/terabotics/gym_ws/skill_learning/extracted_data/combined_dataset/val_data.json')

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Create the model instance
    model = MarkerNet()
    # model = SequentialMarkerNet()
    print(model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Define optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    # Training loop
    num_epochs = 1000
    # writer = SummaryWriter('/home/terabotics/gym_ws/skill_learning/experiment_1')
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for i, (img, pose_normalized, action_normalized, marker_visible) in enumerate(train_loader):
            img, pose_normalized, action_normalized, marker_visible = img.to(device), pose_normalized.to(device), action_normalized.to(device), marker_visible.to(device)
            optimizer.zero_grad()  # Zero the gradients
            
            marker_visible = marker_visible.float().unsqueeze(1)
            # Forward pass

            action_pred = model(img, pose_normalized, marker_visible)
            
            # Compute the loss (MSE between predicted and actual actions)
            loss = criterion(action_pred, action_normalized)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        # writer.add_scalar('Training Loss', running_loss / len(dataset), epoch)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}")
    # writer.close()
    # Save the trained model
    torch.save(model.state_dict(), 'imitation_learning_model.pth')
    print("Model training complete and saved.")

    # Evaluation mode
    model.eval()

    with open('/home/terabotics/gym_ws/skill_learning/extracted_data/combined_dataset/val_data.json', 'r') as f:
        dataset = json.load(f)

    # Extract pose data
    poses = np.array([data['pose'] for data in dataset])

    # Evaluate on the training set
    total_loss = 0.0

    with torch.no_grad():  # No need to compute gradients during evaluation
        for img, pose_normalized, action_normalized, marker_visible in val_loader:
            img = img.to(device)
            pose_normalized = pose_normalized.to(device)
            action_normalized = action_normalized.to(device)
            marker_visible = marker_visible.to(device)

            marker_visible = marker_visible.float().unsqueeze(1)

            action_pred = model(img, pose_normalized, marker_visible)
            loss = criterion(action_pred, action_normalized)
            total_loss += loss.item()
            new_pose = pose_normalized + action_pred

    print(f"Model evaluation loss: {total_loss / len(val_loader):.4f}")