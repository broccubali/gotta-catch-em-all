from tqdm import tqdm
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
import torch.optim as optim
import pandas as pd
import os
import torch.nn.functional as F
from PIL import Image


class CNN(nn.Module):
    def __init__(self, num_classes=143):
        super(CNN, self).__init__()
        # Convolutional Layer 1
        self.conv1 = nn.Conv2d(
            3, 32, kernel_size=3, padding=1
        )  # Change input channels to 3
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Convolutional Layer 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

        # Convolutional Layer 3
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        # Calculate the size of the flattened feature map
        self.flattened_size = 64 * (299 // 2 // 2 // 2) * (299 // 2 // 2 // 2)

        # Fully Connected Layer 1
        self.fc1 = nn.Linear(self.flattened_size, 64)

        # Output Layer
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # Conv Layer 1 + ReLU + Max Pool
        x = self.pool(F.relu(self.conv2(x)))  # Conv Layer 2 + ReLU + Max Pool
        x = self.pool(F.relu(self.conv3(x)))  # Conv Layer 3 + ReLU + Max Pool
        x = x.view(-1, self.flattened_size)  # Flatten the output
        x = F.relu(self.fc1(x))  # Fully Connected Layer 1 + ReLU
        x = self.fc2(x)  # Output Layer
        return F.log_softmax(x, dim=1)


# Instantiate the model
model = CNN(num_classes=143)
# Load the updated state dictionary into the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define the transformations (with additional data augmentation)
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

# Define the loss function and optimizer
num_epochs = 20
criterion = nn.CrossEntropyLoss()  # For classification
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

train_dataset = datasets.ImageFolder(
    root="/home/shusrith/Downloads/aoml-hackathon-1/dataset/train", transform=transform
)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)

val_dataset = datasets.ImageFolder(
    root="/home/shusrith/Downloads/aoml-hackathon-1/dataset/validation",
    transform=transform,
)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True, num_workers=4)

best_val_loss = float("inf")

for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    running_loss = 0.0

    for inputs, labels in tqdm(train_loader):
        inputs, labels = inputs.to(device), labels.to(
            device
        )  # Move data to GPU if available

        optimizer.zero_grad()  # Zero the parameter gradients

        outputs = model(inputs)  # Forward pass
        loss = criterion(outputs, labels)  # Compute loss for main output
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

    # scheduler.step(val_loss)
    current_lr = optimizer.param_groups[0]["lr"]
    print(f"Current learning rate: {current_lr}")

    # Checkpoint the best model
    if val_loss < best_val_loss and val_accuracy > 90:
        best_val_loss = val_loss
        torch.save(model.state_dict(), f"inception_v3_{best_val_loss:.4f}.pth")
        print("Saved best model")
    print("best val loss", best_val_loss)

print("Finished Training")

# Save the final model's state dictionary
torch.save(model.state_dict(), "inception_v3_final1.pth")

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
