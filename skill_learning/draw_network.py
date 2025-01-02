from torchviz import make_dot
from torch.autograd import Variable
import torch
from network_ import SequenceMarkerNet

model = SequenceMarkerNet()

# Create a sample input to pass through the model
# Adjust the size according to your input format (image, pose, marker visibility)
# Assuming an image input size (1, 3, 224, 224), pose input (1, 7), and marker visibility (1, 1)
image_input = Variable(torch.randn(1, 3, 224, 224))  # Example input for image
pose_input = Variable(torch.randn(1, 7))  # Example input for pose
marker_visibility = Variable(torch.randn(1, 1))  # Example marker visibility input

# Pass through the model (assuming model takes multiple inputs)
output = model(image_input, pose_input, marker_visibility)

# Generate the visualization
dot = make_dot(output, params=dict(model.named_parameters()))
dot.render("SequenceMarkerNet", format="png")  # Saves the diagram as a PNG file