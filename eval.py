import argparse
import torch
from torchvision import transforms
from datasets.leaf_dataset import get_dataloader
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import shutil

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, required=True, help='Validation/test dataset path')
parser.add_argument('--model', type=str, choices=['vgg16', 'efficientnet', 'vit', 'resnet', 'resnet_improved'], required=True)
parser.add_argument('--weights', type=str, required=True, help='Model weights file path')
parser.add_argument('--batch_size', type=int, default=32)
args = parser.parse_args()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_loader, classes = get_dataloader(args.data_dir, transform, args.batch_size, shuffle=False)
num_classes = len(classes)

if args.model == 'vgg16':
    from models.vgg16_transfer import get_model
elif args.model == 'efficientnet':
    from models.efficientnet import get_model
elif args.model == 'vit':
    from models.vit import get_model
elif args.model == 'resnet':
    from models.resnet import get_model
elif args.model == 'resnet_improved':
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

# Load weights
model.load_state_dict(torch.load(args.weights, map_location=device))
model.eval()

all_labels = []
all_preds = []
all_paths = []
error_samples = []

with torch.no_grad():
    for i, (images, labels) in enumerate(tqdm(val_loader, desc='Validating')):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(predicted.cpu().numpy())
        # Get filenames (ImageFolder has samples attribute by default)
        batch_start = i * args.batch_size
        batch_end = batch_start + images.size(0)
        if hasattr(val_loader.dataset, 'samples'):
            batch_paths = [val_loader.dataset.samples[idx][0] for idx in range(batch_start, batch_end)]
            all_paths.extend(batch_paths)
            for path, label, pred in zip(batch_paths, labels.cpu().numpy(), predicted.cpu().numpy()):
                if label != pred:
                    error_samples.append((path, label, pred))

# Confusion matrix
cm = confusion_matrix(all_labels, all_preds)
print('Confusion Matrix:')
print(cm)

# Visualize confusion matrix
plt.figure(figsize=(18, 16))
sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
plt.close()
print('Confusion matrix saved as confusion_matrix.png')

# Classification report (including F1 scores)
report = classification_report(all_labels, all_preds, target_names=classes)
print('Classification Report:')
print(report)

# Save error samples
with open('error_samples.txt', 'w') as f:
    for path, label, pred in error_samples:
        f.write(f'{path}\ttrue:{classes[label]}\tpred:{classes[pred]}\n')
print(f'Error samples count: {len(error_samples)}, saved to error_samples.txt')

# Save error images to folder for manual analysis
error_dir = 'error_samples'
os.makedirs(error_dir, exist_ok=True)
with open(os.path.join(error_dir, 'error_list.txt'), 'w') as f:
    for i, (path, label, pred) in enumerate(error_samples):
        fname = f'err_{i}_{os.path.basename(path)}'
        dst_path = os.path.join(error_dir, fname)
        shutil.copy2(path, dst_path)
        f.write(f'{fname}\ttrue:{classes[label]}\tpred:{classes[pred]}\n')
print(f'Error images saved to {error_dir}/ and error_list.txt generated') 