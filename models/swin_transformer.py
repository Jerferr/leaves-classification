import timm

def get_model(num_classes):
    model = timm.create_model('swin_base_patch4_window7_224', pretrained=True, num_classes=num_classes)
    return model 