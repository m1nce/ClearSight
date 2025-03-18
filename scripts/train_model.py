import argparse
import os
import sys
import csv
import json
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
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=10, log_path=None):
    model.train()
    
    results = []  # Store results for CSV/JSON output
    
    with open(log_path, 'w') as log_file:  # Open log file
        log_file.write("Epoch,Train Loss,Validation Accuracy\n")  # Header
        
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
            avg_loss = running_loss / len(train_loader)
            
            # Save to log file
            log_file.write(f"{epoch+1},{avg_loss:.4f},{val_acc:.2f}\n")
            
            # Save to results list for CSV/JSON
            results.append({"epoch": epoch+1, "train_loss": avg_loss, "val_acc": val_acc})
            
            print(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f} - Val Acc: {val_acc:.2f}%")
    
    return results 

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

    # Load dataset
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

    # Create logs directory
    os.makedirs('../logs', exist_ok=True)
    log_file_path = f"../logs/{args.model}_training_log.csv"

    # Train model and get results
    results = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=args.epochs, log_path=log_file_path)

    # Save model
    os.makedirs('../models', exist_ok=True)
    model_name = f"{args.model}_condition_classifier.pth"
    model_path = os.path.join('../models', model_name)
    torch.save(model.state_dict(), model_path)
    print(f"Model saved as {model_name}")

    # Save results as CSV
    csv_path = f"../logs/{args.model}_training_results.csv"
    with open(csv_path, 'w', newline='') as csvfile:
        fieldnames = ["epoch", "train_loss", "val_acc"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"Training results saved as {csv_path}")

    # Save results as JSON
    json_path = f"../logs/{args.model}_training_results.json"
    with open(json_path, 'w') as jsonfile:
        json.dump(results, jsonfile, indent=4)
    print(f"Training results saved as {json_path}")