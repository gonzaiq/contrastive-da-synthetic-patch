import matplotlib.pyplot as plt
import os

# Function to plot the training and validation metrics, including cmmd and dcmmd
def plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies, train_cmmd, val_cmmd, train_dcmmd, val_dcmmd, num_epochs, output_dir="results"):
    epochs = range(1, num_epochs + 1)

    plt.figure(figsize=(12, 10))

    # Plot training and validation loss
    plt.subplot(2, 2, 1)
    plt.plot(epochs, train_losses, label='Training Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.xlabel('Epochs') 
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.grid()
    plt.legend()

    # Plot training and validation accuracy
    plt.subplot(2, 2, 2)
    plt.plot(epochs, train_accuracies, label='Training Accuracy')
    plt.plot(epochs, val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy Curve')
    plt.grid()
    plt.legend()

    # Plot training and validation cmmd
    plt.subplot(2, 2, 3)
    plt.plot(epochs, train_cmmd, label='Training CMMD')
    plt.plot(epochs, val_cmmd, label='Validation CMMD')
    plt.xlabel('Epochs')
    plt.ylabel('CMMD')
    plt.title('CMMD Curve')
    plt.grid()
    plt.legend()

    # Plot training and validation dcmmd
    plt.subplot(2, 2, 4)
    plt.plot(epochs, train_dcmmd, label='Training DCMMD')
    plt.plot(epochs, val_dcmmd, label='Validation DCMMD')
    plt.xlabel('Epochs')
    plt.ylabel('DCMMD')
    plt.title('DCMMD Curve')
    plt.grid()
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "train_curves.png"))
    plt.show()