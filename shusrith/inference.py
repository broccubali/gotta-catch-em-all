from tqdm import tqdm
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
import torch.optim as optim
import pandas as pd
import os
from PIL import Image

model = models.inception_v3(pretrained=True)

num_classes = 143  # Set to your dataset's number of classes
model.AuxLogits.fc = nn.Sequential(
    nn.Dropout(p=0.3), nn.Linear(model.AuxLogits.fc.in_features, num_classes)
)
model.fc = nn.Sequential(
    nn.Dropout(p=0.3), nn.Linear(model.fc.in_features, num_classes)
)
# Fine-tune more layers
for param in model.parameters():
    param.requires_grad = True
state_dict = torch.load("inception_v3_best.pth")

# Update the keys to match the new model structure
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

# Load the updated state dictionary into the model
model.load_state_dict(new_state_dict)  # Move the entire model to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
transform = transforms.Compose(
    [
        transforms.Resize((299, 299)),  # Resize images to 299x299 for Inception v3
        transforms.RandomHorizontalFlip(),  # Random horizontal flip
        transforms.RandomRotation(10),  # Random rotation
        transforms.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2
        ),  # Random color jitter
        transforms.ToTensor(),  # Convert image to tensor
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),  # Normalize with dataset stats
    ]
)
train_dataset = datasets.ImageFolder(
    root="/home/shusrith/Downloads/aoml-hackathon-1/dataset/train", transform=transform
)


class_labels = train_dataset.classes

test_image_folder = "/home/shusrith/Downloads/aoml-hackathon-1/dataset/test"
test_image_paths = [
    os.path.join(test_image_folder, fname) for fname in os.listdir(test_image_folder)
]

predicted_class = []
file_paths = []
model.eval()  # Set the model to evaluation
# Perform inference on the test dataset
with torch.no_grad():
    for image_path in tqdm(test_image_paths):
        image = Image.open(image_path)
        image = (
            transform(image).unsqueeze(0).to(device)
        )  # Add batch dimension and move to GPU
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        predicted_class.append(class_labels[predicted.item()])
        file_paths.append(image_path)

# Create a DataFrame
df = pd.DataFrame({"Image_Name": file_paths, "Class": predicted_class})
df["Image_Name"] = df["Image_Name"].apply(lambda x: x.split("/")[-1])
df.to_csv("submission.csv", index=False)
