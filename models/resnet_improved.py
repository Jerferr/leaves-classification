import torch.nn as nn
from torchvision import models

def get_model(num_classes, dropout_rate=0.5):
    """
    Improved ResNet50 with regularization techniques to reduce overfitting
    """
    model = models.resnet50(pretrained=True)
    
    # Freeze early layers to reduce overfitting
    for param in model.parameters():
        param.requires_grad = False
    
    # Only train the last few layers
    for param in model.layer4.parameters():
        param.requires_grad = True
    
    # Replace classifier head with dropout
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(dropout_rate),
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.Dropout(dropout_rate),
        nn.Linear(512, num_classes)
    )
    
    return model 