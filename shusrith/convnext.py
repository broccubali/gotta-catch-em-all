from tqdm import tqdm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
import torch.optim as optim
import pandas as pd
import os
from PIL import Image
import timm

# Load the pre-trained ConvNeXt Large model
model = timm.create_model("convnext_base", pretrained=True, num_classes=143)

# Move the entire model to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define the transformations (with additional data augmentation)
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),  # Resize images to 224x224
        transforms.RandomHorizontalFlip(),  # Random horizontal flip
        transforms.RandomRotation(10),  # Random rotation
        transforms.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2
        ),  # Random color jitter
        transforms.ToTensor(),  # Convert image to tensor
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),  # Normalize with ImageNet stats
    ]
)

# Define the loss function and optimizer
num_epochs = 50
criterion = nn.CrossEntropyLoss()  # For classification
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.reduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)

train_dataset = datasets.ImageFolder(
    root="/kaggle/input/aoml-hackathon-1/dataset/train", transform=transform
)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)

val_dataset = datasets.ImageFolder(
    root="/kaggle/input/aoml-hackathon-1/dataset/validation", transform=transform
)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True, num_workers=4)

device = "cuda"
for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    running_loss = 0.0

    for inputs, labels in tqdm(train_loader):
        inputs, labels = inputs.to("cuda"), labels.to(
            "cuda"
        )  # Move data to GPU if available

        optimizer.zero_grad()  # Zero the parameter gradients

        outputs = model(inputs)  # Forward pass
        loss = criterion(outputs, labels)  # Compute loss
        loss.backward()  # Backward pass
        optimizer.step()  # Update weights

        running_loss += loss.item() * inputs.size(0)

    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_loss /= len(val_loader.dataset)
    val_accuracy = 100 * correct / total

    # Print validation accuracy
    print(f"Validation Accuracy: {val_accuracy}%")

    epoch_loss = running_loss / len(train_loader.dataset)
    print(
        f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Validation Loss: {val_loss:.4f}"
    )

    prev_lr = optimizer.param_groups[0]["lr"]
    scheduler.step(val_loss)
    current_lr = optimizer.param_groups[0]["lr"]
    if current_lr < prev_lr:
        print(f"Learning rate reduced to {current_lr}")


print("Finished Training")

# Save the model's state dictionary
torch.save(model.state_dict(), "convnext_large.pth")

class_labels = train_dataset.classes

test_image_folder = "/kaggle/input/aoml-hackathon-1/dataset/test"
test_image_paths = [
    os.path.join(test_image_folder, fname) for fname in os.listdir(test_image_folder)
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
df = pd.DataFrame({"Image_Name": file_paths, "Class": predicted_class})
df["Image_Name"] = df["Image_Name"].apply(lambda x: x.split("/")[-1])
df.to_csv("submission.csv", index=False)
