from torchvision import models, transforms
from torch.utils.data import DataLoader
from src.dataset import CustomDataset

# Initialize DataLoaders
def initialize_dataloaders(datalist, image_size=128, batch_size=32, num_workers=4):

    # Define the transform for the training data
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),  # Resize
        transforms.RandomHorizontalFlip(),  # Random horizontal flip
        transforms.RandomVerticalFlip(),  # Random vertical flip
        transforms.RandomRotation(30),  # Random rotation by up to 30 degrees
        transforms.ToTensor(),  # Convert image to tensor
        # Optional: Add normalization here if needed, e.g., transforms.Normalize(mean, std)
    ])

    # Define the transform for the validation data
    valid_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),  # Resize
        transforms.ToTensor(),  # Convert image to tensor
        # Optional: Add normalization here if needed
    ])

    train_dataset = CustomDataset(datalist['validation'], transform=train_transform)
    val_dataset = CustomDataset(datalist['training'], transform=valid_transform)

    dataloaders = {
        "train": DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers),
        "val": DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    }
    return dataloaders