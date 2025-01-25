import torch
import torch.nn as nn
import torch.optim as optim
import os
import argparse
from src.dataloader import initialize_dataloaders
from src.dataset import load_dataset
from src.models.utils import initialize_model
from src.utils import plot_metrics
from src.metrics import class_wise_mmd, different_class_mmd

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths
datalist_pth = "data/synthetic_patches/datalist_train_valid.json"

# Hyperparameters
def parse_args():
    parser = argparse.ArgumentParser(description="Hyperparameters for training")
    
    # Hyperparameters
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size for training")
    parser.add_argument('--learning_rate', type=float, default=0.001, help="Learning rate")
    parser.add_argument('--num_epochs', type=int, default=20, help="Number of epochs for training")
    parser.add_argument('--num_workers', type=int, default=4, help="Number of workers for data loading")
    parser.add_argument('--image_size', type=int, default=256, help="Size of (resized) input images")
    parser.add_argument('--model_name', type=str, default="densenet", help="Name of the model (densenet or resnet supported)")
    parser.add_argument('--output_dir', type=str, default="results_ce", help="Directory to store the results")
    parser.add_argument('--datalist_pth', type=str, default="data/synthetic_patches/datalist_train_valid.json", help="Path to the datalist JSON file")
    
    return parser.parse_args()

args = parse_args()

# Training loop
def train_model(model, dataloaders, criterion, optimizer, num_epochs, output_dir="results"):
    best_acc = 0.0
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    train_cmmds, val_cmmds = [], []
    train_dcmmds, val_dcmmds = [], []

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        print("-" * 20)

        for phase in ["train", "val"]:
            model.train() if phase == "train" else model.eval()

            running_loss = 0.0
            running_corrects = 0
            running_cmmd = 0
            running_dcmmd = 0

            for inputs, _, labels, lut_flag in dataloaders[phase]:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs, features = model.forward_with_features(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()
    
                # filter features and labels in the two domains
                features_d1 = features[lut_flag == 1]
                features_d2 = features[lut_flag == 0]
                labels_d1 = labels[lut_flag == 1]
                labels_d2 = labels[lut_flag == 0]

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                running_cmmd += class_wise_mmd(features_d1.detach().numpy(), features_d2.detach().numpy(), labels_d1.detach().numpy(), labels_d2.detach().numpy())
                running_dcmmd += different_class_mmd(features_d1.detach().numpy(), features_d2.detach().numpy(), labels_d1.detach().numpy(), labels_d2.detach().numpy())

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            epoch_cmmd = running_cmmd / len(dataloaders[phase].dataset)
            epoch_dcmmd = running_dcmmd / len(dataloaders[phase].dataset)
    
            print(f"{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} CMMD: {epoch_cmmd:.4f} DCMMD: {epoch_dcmmd:.4f}")

            if phase == "train":
                train_losses.append(epoch_loss)
                train_accuracies.append(epoch_acc)
                train_cmmds.append(epoch_cmmd)
                train_dcmmds.append(epoch_dcmmd)
            else:
                val_losses.append(epoch_loss)
                val_accuracies.append(epoch_acc)
                val_cmmds.append(epoch_cmmd)
                val_dcmmds.append(epoch_dcmmd)

            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(model.state_dict(), os.path.join(output_dir, "best_model.pth"))
    
        torch.save(model.state_dict(), os.path.join(output_dir, "last_model.pth"))

    print(f"Best Validation Accuracy: {best_acc:.4f}")

    # Optionally, save the metrics to a file
    torch.save({
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accuracies': train_accuracies,
        'val_accuracies': val_accuracies,
        'train_cmmds': train_cmmds,
        'val_cmmds': val_cmmds,
        'train_dcmmds': train_dcmmds,
        'val_dcmmds': val_dcmmds
    }, os.path.join(output_dir, 'training_metrics.pth'))

    # Plot the training and validation metrics
    plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies, train_cmmds, val_cmmds, train_dcmmds, val_dcmmds, num_epochs, output_dir=output_dir)

# Main function
def main():
    # Load dataset and initialize dataloaders
    datalist = load_dataset(datalist_pth)
    dataloaders = initialize_dataloaders(datalist, image_size=args.image_size, batch_size=args.batch_size, num_workers=args.num_workers)
    num_classes = 3

    # Initialize model, loss function, and optimizer
    model = initialize_model(args.model_name.lowe(), num_classes, device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # Train the model
    train_model(model, dataloaders, criterion, optimizer, args.num_epochs, output_dir=args.output_dir)

if __name__ == "__main__":
    main()
