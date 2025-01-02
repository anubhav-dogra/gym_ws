import json
import matplotlib.pyplot as plt
import cv2

# Load the dataset
# dataset file 
dataset_file = "/home/terabotics/gym_ws/skill_learning/extracted_data/demo1_dataset.json"
with open(dataset_file, 'r') as f:
    dataset = json.load(f)

# Inspect the first entry
first_entry = dataset[0]
print("First entry in the dataset:")
print(f"Image file: {first_entry['image_file']}")
print(f"Pose: {first_entry['pose']}")

# Display the first image
image = cv2.imread(first_entry['image_file'])
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("First Image in Dataset")
plt.show()
