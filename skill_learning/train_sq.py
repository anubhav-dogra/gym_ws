import json 
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim
from network_ import SequenceMarkerNet

class RobotSeqTrainingDataset(Dataset):
    def __init__(self, data_file, transform=None):
        self.data_list = json.load(open(data_file, 'r'))
        self.transform = transform
        
        #pre compute normalization parameters
        self.pose_mean = np.mean([data['pose'] for data in self.data_list], axis=0)
        self.pose_std = np.std([data['pose'] for data in self.data_list], axis=0)
        self.action_mean = np.mean([data['action'] for data in self.data_list], axis=0)
        self.action_std = np.std([data['action'] for data in self.data_list], axis=0)

    def standardize_(self, value, value_mean, value_std):
        """Standardizes the pose using mean and standard deviation."""
        return (value - value_mean) / value_std
    
    def destandardize_(self, value, value_mean, value_std):
        """Reverts pose from standardized form to the original scale."""
        return value * value_std + value_mean
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        data_entry = self.data_list[idx]

        img = cv2.imread(data_entry['image_path'])
        img = cv2.resize(img, (224, 224))
        img = torch.tensor(img).permute(2, 0, 1).float()/255.0  # Normalize and reorder to PyTorch format (C, H, W)

        pose= torch.tensor(data_entry['pose']).float()
        pose_normalized = torch.tensor(self.standardize_(pose.numpy(), self.pose_mean, self.pose_std)).float()


        action = torch.tensor(data_entry['action']).float()
        action_normalized = torch.tensor(self.standardize_(action.numpy(), self.action_mean, self.action_std)).float()

        marker_visible = torch.tensor(data_entry['marker_visible']).float() # Marker visibility as a float (0 or 1)

        return img, pose_normalized, action_normalized, marker_visible


def create_sequences(data, seq_length):
    sequences = []
    for i in range(len(data) - seq_length + 1):
        seq = [data[j] for j in range(i, i + seq_length)]
        images, poses, actions, markers = zip(*seq)
        sequences.append((
            torch.stack(images),
            torch.stack(poses),
            torch.stack(actions),
            torch.stack(markers)
            ))
    return sequences

seq_length = 10  # Define sequence length
batch_size = 16  # Define batch size
training_rate = 1e-3  # Define training rate
num_epochs = 100  # Define number of epochs

train_dataset = RobotSeqTrainingDataset('/home/terabotics/gym_ws/skill_learning/extracted_data/combined_dataset/train_data_sq.json')
val_dataset = RobotSeqTrainingDataset('/home/terabotics/gym_ws/skill_learning/extracted_data/combined_dataset/val_data_sq.json')

train_sequences = create_sequences(train_dataset, seq_length)
val_sequences = create_sequences(val_dataset, seq_length)

train_dataloader = DataLoader(train_sequences, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_sequences, batch_size=batch_size, shuffle=False)

# Model, loss function, and optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = SequenceMarkerNet().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=training_rate)

# Training loop
for epoch in range(num_epochs):
    model.train()
    running_loss =  0.0
    for images, poses, actions, markers in train_dataloader:
        images = images.to(device)
        poses = poses.to(device)
        actions = actions.to(device)
        markers = markers.to(device)

        optimizer.zero_grad()
        outputs = model(images, poses, markers)
        loss = criterion(outputs, actions)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()*images.size(0)

    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_sequences):.4f}")

# Validation loop
model.eval()
val_loss = 0.0
with torch.no_grad():
    for images, poses, actions, markers in val_dataloader:
        images = images.to(device)
        poses = poses.to(device)
        actions = actions.to(device)
        markers = markers.to(device)

        outputs = model(images, poses, markers)
        loss = criterion(outputs, actions)
        val_loss += loss.item()*images.size(0)

print(f"Validation Loss: {val_loss / len(val_sequences):.4f}")

torch.save(model.state_dict(), 'sq_model.pth')
print("Model training complete and saved.")