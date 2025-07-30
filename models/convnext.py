import timm

def get_model(num_classes):
    model = timm.create_model('convnext_base', pretrained=True, num_classes=num_classes)
    return model 