import argparse
import os
import sys
import torch
import torch.optim as optim
from torchvision import models, transforms
from torch import nn
from torch.utils.data import DataLoader

# Ensure the scripts directory is in the path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(os.path.join(parent_dir, 'scripts'))

# Import dataset and model
try:
    from ConditionDataset import ConditionDataset
    from ConditionClassifier import ConditionClassifier
except ImportError as e:
    print(f"Error importing modules: {e}. Ensure 'scripts/' contains the required files.")
    sys.exit(1)

# Argument parser
def parse_args():
    parser = argparse.ArgumentParser(description="Train a condition classifier.")
    parser.add_argument("--model", type=str, choices=["mobilenet", "custom"], default="mobilenet",
                        help="Choose model: 'mobilenet' or 'custom' (default: mobilenet)")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs (default: 10)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size (default: 32)")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate (default: 0.001)")
    return parser.parse_args()

# Training function
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=10):
    model.train()
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct, total = 0, 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass
            loss.backward()
            optimizer.step()

            # Metrics
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        scheduler.step()
        val_acc = evaluate_model(model, val_loader)
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {running_loss/len(train_loader):.4f} - Val Acc: {val_acc:.2f}%")

# Evaluation function
def evaluate_model(model, val_loader):
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return 100 * correct / total  # Return accuracy

if __name__ == "__main__":
    args = parse_args()

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load dataset using torch.utils.data.DataLoader
    train_dataset = ConditionDataset(original_path='../data/cityscapes/train', 
                                     augmented_path='../data/aug_cityscapes/train', transform=transform)
    val_dataset = ConditionDataset(original_path='../data/cityscapes/val', 
                                   augmented_path='../data/aug_cityscapes/val', transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Choose model
    if args.model == "mobilenet":
        print("Using MobileNetV3 as the model.")
        model = models.mobilenet_v3_small(pretrained=True)
        model.classifier[-1] = nn.Linear(1024, 3)  # Adjust output layer
    else:
        print("Using Custom Condition Classifier.")
        model = ConditionClassifier(num_classes=3)

    model = model.to(device)

    # Define loss, optimizer, scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # Train model
    train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=args.epochs)

    # Create models directory
    models_dir = os.path.join(parent_dir, 'models')
    if not os.path.exists(models_dir):
        os.mkdir(models_dir)

    # Save model to ../models/
    model_name = f"{args.model}_condition_classifier.pth"
    model_path = os.path.join('../models', model_name)
    torch.save(model.state_dict(), model_path)
    print(f"Model saved as {model_name}")
