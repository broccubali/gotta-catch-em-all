import pandas as pd
from tqdm import tqdm
import torch
from torchvision import datasets, transforms
from PIL import Image
import resnet
import os

# Define the transformations (same as used during training)
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),  # Resize images to 224x224
        transforms.ToTensor(),  # Convert image to tensor
        transforms.Normalize(
            mean=[0.6020, 0.5866, 0.5546], std=[0.2477, 0.2404, 0.2478]
        ),  # Normalize with ImageNet stats
    ]
)

# Load the trained model
model = resnet.resnet50(num_classes=143).to("cuda")
model.load_state_dict(torch.load("resnet50.pth"))
model.eval()  # Set the model to evaluation mode

# Load the class labels
train_dataset = datasets.ImageFolder(
    root="/home/shusrith/Downloads/aoml-hackathon-1/dataset/train", transform=transform
)
class_labels = train_dataset.classes
print("Class labels:", class_labels)

# Load test images
test_image_folder = "/home/shusrith/Downloads/aoml-hackathon-1/dataset/test"
test_image_paths = [
    os.path.join(test_image_folder, fname)
    for fname in os.listdir(test_image_folder)
    if fname.endswith((".png", ".jpg", ".jpeg"))
]

predicted_class = []
file_paths = []

# Perform inference on the test dataset
with torch.no_grad():
    for image_path in tqdm(test_image_paths):
        image = Image.open(image_path)
        image = (
            transform(image).unsqueeze(0).to("cuda")
        )  # Add batch dimension and move to GPU
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        predicted_class.append(class_labels[predicted.item()])
        file_paths.append(image_path)

# Create a DataFrame
df = pd.DataFrame({"Image_name": file_paths, "Class": predicted_class})

# Save to CSV
df.to_csv("submission.csv", index=False)
