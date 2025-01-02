import json
import numpy as np
from sklearn.model_selection import train_test_split

def load_data_from_json(json_path):
    """Load data from a JSON file."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

def save_data_to_json(data, json_path):
    """Save data to a JSON file."""
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=4)

# Load data from demo1 and demo2 datasets
demo1_data = load_data_from_json('/home/terabotics/gym_ws/skill_learning/extracted_data/demo1_dataset/data.json')
demo2_data = load_data_from_json('/home/terabotics/gym_ws/skill_learning/extracted_data/demo2_dataset/data.json')

# Combine datasets
combined_data = demo1_data + demo2_data



# Split the data into training and validation sets with random_state=42
# train_data, val_data = train_test_split(combined_data, test_size=0.2, random_state=42)

# # Save the split datasets
# save_data_to_json(train_data, '/home/terabotics/gym_ws/skill_learning/extracted_data/combined_dataset/train_data.json')
# save_data_to_json(val_data, '/home/terabotics/gym_ws/skill_learning/extracted_data/combined_dataset/val_data.json')



def split_sequences(data, train_ratio=0.8):
    # Split indices
    train_size = int(len(data) * train_ratio)
    
    train_data = data[:train_size]
    val_data = data[train_size:]
    
    return train_data, val_data

train_data_sq, val_data_sq = split_sequences(combined_data)
save_data_to_json(train_data_sq, '/home/terabotics/gym_ws/skill_learning/extracted_data/combined_dataset/train_data_sq.json')
save_data_to_json(val_data_sq, '/home/terabotics/gym_ws/skill_learning/extracted_data/combined_dataset/val_data_sq.json')

print("Data combined and split successfully. Training and validation data saved.")
