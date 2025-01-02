import json
import os
import pandas as pd

def load_pose_data(pose_file):
    return pd.read_csv(pose_file)

def create_image_pose_mapping(image_dir, pose_file):
    pose_data = load_pose_data(pose_file)
    image_files = sorted(os.listdir(image_dir))  # Ensure image files are sorted
    image_pose_mapping = []

    for i, image_file in enumerate(image_files):
        if i >= len(pose_data):
            break
        pose = pose_data.iloc[i]
        image_pose_mapping.append({
            'image_file': os.path.join(image_dir, image_file),
            'pose': pose.tolist()
        })
    return image_pose_mapping

def create_training_dataset(mapping, output_file):
    dataset = []
    for entry in mapping:
        data = {
            'image_file': entry['image_file'],
            'pose': entry['pose']
        }
        dataset.append(data)
    with open(output_file, 'w') as f:
        json.dump(dataset, f, indent=4)
    print(f"Training dataset created at {output_file}")

if __name__ == "__main__":
    import sys


    # if len(sys.argv) != 4:
    #     print("Usage: python create_dataset.py <image_dir> <pose_file> <output_file>")
    #     sys.exit(1)

    # image_dir = sys.argv[1]
    # pose_file = sys.argv[2]
    # output_file = sys.argv[3]

    image_dir = "/home/terabotics/gym_ws/skill_learning/extracted_data/demo1_images/"
    pose_file = "/home/terabotics/gym_ws/skill_learning/extracted_data/demo1_pose.csv"
    output_file = "/home/terabotics/gym_ws/skill_learning/extracted_data/demo1_dataset.json"

    mapping = create_image_pose_mapping(image_dir, pose_file)
    create_training_dataset(mapping, output_file)
