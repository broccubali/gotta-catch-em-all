import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Define the transformations (excluding normalization)
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),  # Resize images to a fixed size
        transforms.ToTensor(),  # Convert image to tensor
    ]
)

# Define the dataset and dataloader
dataset = datasets.ImageFolder(
    root="/home/shusrith/Downloads/aoml-hackathon-1/dataset/train", transform=transform
)
dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)


def calculate_mean_std(dataloader):
    mean = 0.0
    std = 0.0
    total_images = 0
    for images, _ in dataloader:
        batch_samples = images.size(0)  # batch size (number of images in the batch)
        images = images.view(
            batch_samples, images.size(1), -1
        )  # reshape to (batch_size, channels, height * width)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total_images += batch_samples

    mean /= total_images
    std /= total_images
    return mean, std


mean, std = calculate_mean_std(dataloader)
print(f"Mean: {mean}")
print(f"Std: {std}")
