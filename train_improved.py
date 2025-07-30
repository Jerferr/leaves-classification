import argparse
import torch
from torchvision import transforms
from datasets.leaf_dataset import get_dataloader
from tqdm import tqdm
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='data/Plant_leave_diseases_dataset_with_augmentation')
parser.add_argument('--val_dir', type=str, default='data/val', help='Validation dataset path')
parser.add_argument('--model', type=str, choices=['resnet_improved'], default='resnet_improved')
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
parser.add_argument('--patience', type=int, default=5, help='Early stopping patience')
args = parser.parse_args()

# Enhanced data preprocessing - training set
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.3),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Validation set preprocessing (no augmentation)
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_loader, classes = get_dataloader(args.data_dir, train_transform, args.batch_size, shuffle=True)
val_loader, _ = get_dataloader(args.val_dir, val_transform, args.batch_size, shuffle=False)
num_classes = len(classes)

# Load improved model
from models.resnet_improved import get_model
model = get_model(num_classes)

# Device selection
if torch.backends.mps.is_available():
    device = torch.device('mps')
elif torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
print(f"Using device: {device}")
model = model.to(device)

# Loss function and optimizer (with weight decay)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

# Learning rate scheduler
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

# Early stopping mechanism
best_val_loss = float('inf')
patience_counter = 0
best_model_state = None

print(f"Starting training for {args.epochs} epochs")
print(f"Training samples: {len(train_loader.dataset)}")
print(f"Validation samples: {len(val_loader.dataset)}")

for epoch in range(args.epochs):
    # Training phase
    model.train()
    train_loss = 0
    train_correct = 0
    train_total = 0
    
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
    for images, labels in progress_bar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()
        
        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100.*train_correct/train_total:.2f}%'
        })
    
    train_accuracy = 100. * train_correct / train_total
    avg_train_loss = train_loss / len(train_loader)
    
    # Validation phase
    model.eval()
    val_loss = 0
    val_correct = 0
    val_total = 0
    
    with torch.no_grad():
        progress_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]")
        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
            
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*val_correct/val_total:.2f}%'
            })
    
    val_accuracy = 100. * val_correct / val_total
    avg_val_loss = val_loss / len(val_loader)
    
    print(f"Epoch {epoch+1}/{args.epochs}:")
    print(f"  Train - Loss: {avg_train_loss:.4f}, Acc: {train_accuracy:.2f}%")
    print(f"  Val   - Loss: {avg_val_loss:.4f}, Acc: {val_accuracy:.2f}%")
    
    # Learning rate scheduling
    scheduler.step(avg_val_loss)
    
    # Early stopping check
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_counter = 0
        best_model_state = model.state_dict().copy()
        torch.save(model.state_dict(), f'{args.model}_best.pth')
        print(f"  → New best model saved!")
    else:
        patience_counter += 1
        print(f"  → Validation loss not improved ({patience_counter}/{args.patience})")
        
        if patience_counter >= args.patience:
            print(f"Early stopping triggered! Best validation loss: {best_val_loss:.4f}")
            break

print("Training completed!")
if best_model_state is not None:
    model.load_state_dict(best_model_state)
    torch.save(model.state_dict(), f'{args.model}_final.pth')
    print("Best model weights loaded and saved as final model") 