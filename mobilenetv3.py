import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_large

class MobileNetV3Encoder(nn.Module):
    def __init__(self, pretrained=True, return_skip=True):
        super().__init__()
        base_model = mobilenet_v3_large(pretrained=pretrained)
        features = list(base_model.features.children())

        # Select MobileNetV3 stages to use as downsample blocks
        self.stage1 = nn.Sequential(*features[3:4])   # Downsample 1 (e.g., 28x28)
        self.stage2 = nn.Sequential(*features[4:6])   # Downsample 2 (14x14)
        self.stage3 = nn.Sequential(*features[6:8])   # Downsample 3 (7x7)
        self.stage4 = nn.Sequential(*features[8:])    # Final bottleneck

        self.return_skip = return_skip

    def forward(self, x):
        skip_connections = []

        x = self.stage1(x)
        if self.return_skip:
            skip_connections.append(x)

        x = self.stage2(x)
        if self.return_skip:
            skip_connections.append(x)

        x = self.stage3(x)
        if self.return_skip:
            skip_connections.append(x)

        x = self.stage4(x)  # bottleneck

        return (x, skip_connections) if self.return_skip else x
