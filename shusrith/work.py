import pandas as pd
import os
from PIL import Image
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms

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
print(df.loc[1558, 0])
df1 = pd.DataFrame(df1)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 9)  # 128 * 128 * 3 -> 120 * 120 * 6
        self.pool = nn.MaxPool2d(2, 2)  # 60 * 60 * 6
        self.conv2 = nn.Conv2d(6, 16, 5)  # 56 * 56 * 16 -> 28 * 28 * 16
        self.conv3 = nn.Conv2d(16, 32, 5)  # 24 * 24 * 32 -> 12 * 12 * 32
        self.dropout = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.2)
        self.fc1 = nn.Linear(32 * 12 * 12, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, 143)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.dropout(x)
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout2(x)
        x = self.pool(F.relu(self.conv3(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

resize_transform = transforms.Compose(
    [transforms.Resize((128, 128)), transforms.ToTensor()]
)

device = "cuda" if torch.cuda.is_available() else "cpu"
net.to(device)

batch_size = 8  

for epoch in range(2): 
    running_loss = 0.0
    net.train()  
    for i in range(0, len(df), batch_size):
        batch_df = df.iloc[i : i + batch_size]
        images = []
        labels = []
        for input_path, label in zip(batch_df[0], batch_df[1]):
            try:
                img = Image.open(input_path)
                img = resize_transform(img)
                if img.shape[0] == 3:
                    images.append(img)
                    labels.append(label)
            except Exception as e:
                print(f"Error processing {input_path}: {e}")

        if images:
            images = torch.stack(images).to(device)  # Stack images into a single tensor
            labels = torch.tensor(labels, dtype=torch.long).to(
                device
            )  # Convert labels to tensor and move to device

            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % (500 * batch_size) == 0:
                print(
                    f"[{epoch + 1}, {i + 1:5d}] loss: {running_loss / (500 * batch_size):.3f}"
                )
                running_loss = 0.0

    net.eval()  # Set the model to evaluation mode
    val_loss = 0.0
    with torch.no_grad():
        for i in range(0, len(df1), batch_size):
            batch_df1 = df1.iloc[i : i + batch_size]
            images = []
            labels = []
            for input_path, label in zip(batch_df1[0], batch_df1[1]):
                try:
                    img = Image.open(input_path)
                    img = resize_transform(img)
                    images.append(img)
                    labels.append(label)
                except Exception as e:
                    print(f"Error processing {input_path}: {e}")

            if images:
                images = torch.stack(images).to(
                    device
                )  # Stack images into a single tensor
                labels = torch.tensor(labels, dtype=torch.long).to(
                    device
                )  # Convert labels to tensor and move to device

                outputs = net(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                if i % (100 * batch_size) == 0:
                    print(
                        f"[{epoch + 1}, {i + 1:5d}] validation loss: {val_loss / (500 * batch_size):.3f}"
                    )
                    val_loss = 0.0

print("Finished Training")
