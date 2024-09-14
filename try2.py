import resnet
import pandas as pd
import os
from tqdm import tqdm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

labels = os.listdir("/home/shusrith/Downloads/aoml-hackathon-1/dataset/train/")
labels = {j: i for i, j in enumerate(labels)}
val_labels = os.listdir("/home/shusrith/Downloads/aoml-hackathon-1/dataset/validation/")
df = []
for i in labels:
    l = os.listdir(f"/home/shusrith/Downloads/aoml-hackathon-1/dataset/train/{i}")
    for j in l:
        df.append(
            [
                f"/home/shusrith/Downloads/aoml-hackathon-1/dataset/train/{i}/{j}",
                labels[i],
            ]
        )

df1 = []
for i in val_labels:
    l = os.listdir(f"/home/shusrith/Downloads/aoml-hackathon-1/dataset/validation/{i}")
    for j in l:
        df1.append(
            [
                f"/home/shusrith/Downloads/aoml-hackathon-1/dataset/validation/{i}/{j}",
                labels[i],
            ]
        )

df = pd.DataFrame(df)
df.drop(index=[1558, 2884, 8691, 8768], inplace=True)
df.reset_index(drop=True, inplace=True)
df1 = pd.DataFrame(df1)

model = resnet.resnet50(num_classes=143).to("cuda")


transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),  # Resize images to 224x224
        transforms.ToTensor(),  # Convert image to tensor
        transforms.Normalize(
            mean=[0.6020, 0.5866, 0.5546], std=[0.2477, 0.2404, 0.2478]
        ),  # Normalize with ImageNet stats
    ]
)

train_dataset = datasets.ImageFolder(
    root="/home/shusrith/Downloads/aoml-hackathon-1/dataset/train", transform=transform
)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 10
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

    epoch_loss = running_loss / len(train_loader.dataset)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")

print("Finished Training")
