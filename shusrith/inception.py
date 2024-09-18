from tqdm import tqdm
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
import torch.optim as optim
import pandas as pd
import os
from PIL import Image

# Load the pre-trained Inception v3 model
model = models.inception_v3(pretrained=True)

# Modify the final layer to match the number of classes in your dataset
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
optimizer = torch.optim.Adam(model.parameters(), lr=1e-6, weight_decay=1e-4)
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#     optimizer, mode="min", factor=0.1, patience=5, verbose=True
# )

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

        outputs, aux_outputs = model(inputs)  # Forward pass
        loss1 = criterion(outputs, labels)  # Compute loss for main output
        loss2 = criterion(aux_outputs, labels)  # Compute loss for auxiliary output
        loss = loss1 + 0.4 * loss2  # Combine losses
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
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        if val_accuracy > 95:
            torch.save(model.state_dict(), "inception_v3_best.pth")
            print("Saved best model")

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
