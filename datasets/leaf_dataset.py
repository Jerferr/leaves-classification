import torch
from torchvision import datasets

def get_dataloader(data_dir, transform, batch_size=32, shuffle=True):
    dataset = datasets.ImageFolder(data_dir, transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return loader, dataset.classes 