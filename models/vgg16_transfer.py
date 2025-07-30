import torch.nn as nn
from torchvision import models

def get_model(num_classes):
    model = models.vgg16(pretrained=True)
    model.classifier[6] = nn.Linear(4096, num_classes)
    for param in model.features.parameters():
        param.requires_grad = False
    return model 