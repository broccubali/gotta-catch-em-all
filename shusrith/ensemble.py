import torch
import torch.nn as nn
from torchvision import datasets
from torchvision import transforms, models
from PIL import Image
import os
import pandas as pd
from tqdm import tqdm

# Load the pre-trained models
model1 = models.inception_v3(pretrained=True)
model2 = models.inception_v3(pretrained=True)

# Modify the final layers to match the number of classes
num_classes = 143
model1.AuxLogits.fc = nn.Sequential(
    nn.Dropout(p=0.3), nn.Linear(model1.AuxLogits.fc.in_features, num_classes)
)
model1.fc = nn.Sequential(
    nn.Dropout(p=0.3), nn.Linear(model1.fc.in_features, num_classes)
)

model2.AuxLogits.fc = nn.Sequential(
    nn.Dropout(p=0.3), nn.Linear(model2.AuxLogits.fc.in_features, num_classes)
)
model2.fc = nn.Sequential(
    nn.Dropout(p=0.3), nn.Linear(model2.fc.in_features, num_classes)
)

# Load the state dictionaries
state_dict1 = torch.load("inception_v3_0.2986385660218047.pth")
state_dict2 = torch.load("inception_v3_0.2949068700959039.pth")


# Update the keys to match the new model structure
def update_state_dict(state_dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        if key == "AuxLogits.fc.weight":
            new_state_dict["AuxLogits.fc.1.weight"] = value
        elif key == "AuxLogits.fc.bias":
            new_state_dict["AuxLogits.fc.1.bias"] = value
        elif key == "fc.weight":
            new_state_dict["fc.1.weight"] = value
        elif key == "fc.bias":
            new_state_dict["fc.1.bias"] = value
        else:
            new_state_dict[key] = value
    return new_state_dict


model1.load_state_dict(update_state_dict(state_dict1))
model2.load_state_dict(update_state_dict(state_dict2))

# Move the models to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model1.to(device)
model2.to(device)

# Set the models to evaluation mode
model1.eval()
model2.eval()

# Define the transformations
transform = transforms.Compose(
    [
        transforms.Resize((299, 299)),  # Resize images to 299x299 for Inception v3
        transforms.ToTensor(),  # Convert image to tensor
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),  # Normalize with dataset stats
    ]
)

# Define the test image folder
test_image_folder = "/home/shusrith/Downloads/aoml-hackathon-1/dataset/test"
test_image_paths = [
    os.path.join(test_image_folder, fname) for fname in os.listdir(test_image_folder)
]

predicted_class = []
file_paths = []

# Perform ensemble inference on the test dataset
with torch.no_grad():
    for image_path in tqdm(test_image_paths):
        image = Image.open(image_path)
        image = (
            transform(image).unsqueeze(0).to(device)
        )  # Add batch dimension and move to GPU

        # Get predictions from both models
        outputs1 = model1(image)
        outputs2 = model2(image)

        # Average the predictions
        avg_outputs = (outputs1 + outputs2) / 2

        # Get the predicted class
        _, predicted = torch.max(avg_outputs, 1)
        predicted_class.append(predicted.item())
        file_paths.append(image_path)

# Create a DataFrame
class_labels = datasets.ImageFolder(
    root="/home/shusrith/Downloads/aoml-hackathon-1/dataset/train"
).classes
df = pd.DataFrame(
    {"Image_Name": file_paths, "Class": [class_labels[i] for i in predicted_class]}
)
df["Image_Name"] = df["Image_Name"].apply(lambda x: x.split("/")[-1])
df.to_csv("ensemble_submission.csv", index=False)
