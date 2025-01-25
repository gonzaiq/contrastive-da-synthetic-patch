from torchvision import models
import torch.nn as nn
from src.models.resnet import CustomResNet
from src.models.densenet import CustomDenseNet

# Initialize the ResNet model with a modified first convolution layer
def initialize_model(model_name, num_classes, device):
    if model_name == "resnet":
        model = CustomResNet(
            block=models.resnet.BasicBlock,
            num_classes=num_classes,
            in_kernel_size=3,
            in_channels=1
            )
    elif model_name == "densenet":
        
        model = CustomDenseNet(
            num_classes=num_classes,
            in_kernel_size=3,
            in_channels=1
            )

    model = model.to(device)

    return model