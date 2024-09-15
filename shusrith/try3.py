import pandas as pd
import os
from PIL import Image
from tqdm import tqdm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import resnet

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
            mean=[0.6020, 0.5866, 0.5546], std=[0.2477, 0.2404, 0.2478]
        ),  # Normalize with ImageNet stats
    ]
)

# Load the datasets
train_dataset = datasets.ImageFolder(
    root="/home/shusrith/Downloads/aoml-hackathon-1/dataset/train", transform=transform
)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)

val_dataset = datasets.ImageFolder(
    root="/home/shusrith/Downloads/aoml-hackathon-1/dataset/validation",
    transform=transform,
)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True, num_workers=4)

# Define the model
model = resnet.resnet50(num_classes=143).to("cuda")

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, "min", patience=3, factor=0.5
)

# Training loop with early stopping
device = "cuda"
num_epochs = 50
best_val_loss = float("inf")
patience = 5
early_stop_counter = 0

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

    # Adjust learning rate based on validation loss
    scheduler.step(val_loss)

    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "resnet50_best.pth")
        early_stop_counter = 0
    else:
        early_stop_counter += 1
        if early_stop_counter >= patience:
            print("Early stopping triggered")
            break

print("Finished Training")

# Load the best model
model.load_state_dict(torch.load("resnet50_best.pth"))

class_labels = train_dataset.classes

test_image_folder = "/home/shusrith/Downloads/aoml-hackathon-1/dataset/test"
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

# Save to CSV
df.to_csv("submission.csv", index=False)
