import torch
import torch.nn as nn
from torchvision import models
from tqdm import tqdm

class CustomResNet(nn.Module):
    def __init__(self, num_classes):
        super(CustomResNet, self).__init__()
        # Initialize a pre-trained ResNet-18
        self.resnet = models.resnet18(pretrained=True)

        # Freeze all layers
        for param in self.resnet.parameters():
            param.requires_grad = False
            
        # Replace the fully connected layer for custom classification
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x):
        return self.resnet(x)

def train_model(model, train_loader, criterion, optimizer, device, num_epochs=10):
    """
    Trains the model using the given DataLoader, criterion, and optimizer.

    Args:
        model: The neural network model to train.
        train_loader: DataLoader containing training data.
        criterion: Loss function.
        optimizer: Optimizer for updating model parameters.
        device: Device to run the model on (e.g., 'cuda' or 'cpu').
        num_epochs: Number of training epochs.
    """
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            images, labels = images.to(device), labels.to(device)

            # Zero the gradient buffers
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)

            # Compute loss
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

def evaluate_model(model, test_loader, device):
    """
    Evaluates the model using the test DataLoader.

    Args:
        model: The neural network model to evaluate.
        test_loader: DataLoader containing testing data.
        device: Device to run the model on (e.g., 'cuda' or 'cpu').
    """
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            # Forward pass
            outputs = model(images)
            # Get predictions
            _, predicted = torch.max(outputs, 1)
            # Update total and correct counts
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f"Test Accuracy: {100 * correct / total:.2f}%")
