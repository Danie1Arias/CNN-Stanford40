import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models

# Define the Googlenet-based model class
class GooglenetModel(nn.Module):
    def __init__(self, num_classes):
        super(GooglenetModel, self).__init__()
        self.googlenet = models.googlenet(pretrained=True)  # Load pretrained Googlenet model
        self.googlenet.fc = nn.Linear(self.googlenet.fc.in_features, num_classes)  # Modify the final layer for the number of classes

    def forward(self, x):
        return self.googlenet(x)


# Function to train the model
def train_model(model, train_loader, criterion, optimizer, device, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()  # Zero the gradients
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()  # Backward pass
            optimizer.step()  # Update weights

            running_loss += loss.item()

        # Print average loss per epoch
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}")


# Function to evaluate the model
def evaluate_model(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f"Model accuracy on the test set: {accuracy:.2f}%")
    return accuracy
