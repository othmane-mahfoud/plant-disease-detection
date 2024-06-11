import torchvision

from torch import nn
from helpers.setup import set_seeds

# Create an EffNetB0 feature extractor
def create_effnetb0(device, out_features):
    # Get weights and model and send to target device
    weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
    model = torchvision.models.efficientnet_b0(weights=weights).to(device)

    # Freeze base layers
    for param in model.features.parameters():
        param.requires_grad = False

    # Seeds for reproducibility
    set_seeds(device=device)

    # Change classifier head
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(in_features=1280, out_features=out_features)
    ).to(device)

    print(f"EffNetB0 model created successfully.")
    
    return model

def create_vit(device, out_features):
    # Get weights and model and send to target device
    weights = torchvision.models.ViT_B_16_Weights.DEFAULT
    model = torchvision.models.vit_b_16(weights=weights).to(device)
    
    # Freeze base layers
    for parameter in model.parameters():
        parameter.requires_grad = False

    # Seeds for reproducibility
    set_seeds(device=device)

    model.heads = nn.Linear(
        in_features=768, 
        out_features=out_features
    ).to(device)
    
    print(f"ViT model created successfully.")

    return model