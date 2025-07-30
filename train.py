import argparse
import torch
from torchvision import transforms
from datasets.leaf_dataset import get_dataloader
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='data/Plant_leave_diseases_dataset_with_augmentation')
parser.add_argument('--model', type=str, choices=['vgg16', 'efficientnet', 'vit', 'convnext', 'swin', 'resnet'], default='vgg16')
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=32)
args = parser.parse_args()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_loader, classes = get_dataloader(args.data_dir, transform, args.batch_size, shuffle=True)
num_classes = len(classes)

if args.model == 'vgg16':
    from models.vgg16_transfer import get_model
elif args.model == 'efficientnet':
    from models.efficientnet import get_model
elif args.model == 'vit':
    from models.vit import get_model
elif args.model == 'convnext':
    from models.convnext import get_model
elif args.model == 'swin':
    from models.swin_transformer import get_model
elif args.model == 'resnet':
    from models.resnet import get_model

model = get_model(num_classes)

# Prioritize MPS (Apple Silicon), then CUDA, then CPU
if torch.backends.mps.is_available():
    device = torch.device('mps')
elif torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
print(f"Using device: {device}")
model = model.to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(args.epochs):
    model.train()
    total_loss = 0
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
    for images, labels in progress_bar:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
    print(f"Epoch {epoch+1}/{args.epochs}, Loss: {total_loss/len(train_loader):.4f}")

# Save model
torch.save(model.state_dict(), f'{args.model}_best.pth') 