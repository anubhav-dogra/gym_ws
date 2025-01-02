import torch
import torch.nn as nn
class MarkerNet(nn.Module):
    def __init__(self):
        super(MarkerNet, self).__init__()

        self.cnn_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # fully connected layer for combining image and pose inputs
        self.fc_layers = nn.Sequential(
            nn.Linear(128 * 28 * 28 + 7 + 1, 128),  # Image features + pose input + marker visibility
            nn.ReLU(),
            # nn.Dropout(0.5),  # Dropout to reduce overfitting
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 7) # Output is the predicted pose change (delta pose)
        )

    def forward(self, img, pose, marker_visible):
        # Process image through CNN layers
        img_features = self.cnn_layers(img)
        img_features = img_features.view(img_features.size(0), -1)

        # Concatenate image features and pose input
        combined = torch.cat((img_features, pose, marker_visible), dim=1)

        # Pass through fully connected layers
        action_output = self.fc_layers(combined)
        return action_output



class SequenceMarkerNet(nn.Module):
    def __init__(self):
        super(SequenceMarkerNet, self).__init__()
        self.cnn_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.lstm = nn.LSTM(input_size=128*28*28 + 7 + 1, hidden_size=64, batch_first=True)
        self.fc_layers = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 7)
        )
    
    def forward(self, img, pose, marker_visible):
        # Check input dimensions
        if img.dim() == 6:
            # img: (batch_size, seq_length, channels, height, width)
            batch_size, seq_length, channels, height, width = img.size()
            img = img.view(batch_size * seq_length, channels, height, width)  # Flatten sequence
            img_features = self.cnn_layers(img)  # Apply CNN
            img_features = img_features.view(batch_size, seq_length, -1)  # Reshape back to (batch_size, seq_length, features)
        elif img.dim() == 5:
            # img: (batch_size, seq_length, channels, height, width)
            batch_size, seq_length, channels, height, width = img.size()
            img_features = self.cnn_layers(img.view(-1, channels, height, width))  # Flatten sequence and apply CNN
            img_features = img_features.view(batch_size, seq_length, -1)  # Reshape back to (batch_size, seq_length, features)
        elif img.dim() == 4:
            # img: (batch_size, channels, height, width)
            batch_size, channels, height, width = img.size()
            seq_length = 1  # No sequence length
            img_features = self.cnn_layers(img)  # Apply CNN
            img_features = img_features.view(batch_size, seq_length, -1)  # Reshape to (batch_size, seq_length, features)

        # Process pose and marker_visible
        pose = pose.view(batch_size, seq_length, -1)  # Ensure pose is shaped correctly
        marker_visible = marker_visible.view(batch_size, seq_length, -1)  # Ensure marker_visible is shaped correctly

        combined = torch.cat((img_features, pose, marker_visible), dim=2)
        lstm_out, _ = self.lstm(combined)
        output = self.fc_layers(lstm_out)

        return output
