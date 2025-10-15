#7 the new fct 
import torch.nn as nn
import torchvision.models as models

class TickSpeciesSexModel(nn.Module):
    def __init__(self, num_species, num_sex, backbone_name='mobilenet_v2', pretrained=True):
        super().__init__()

        # 🔁 Choose the backbone
        if backbone_name == 'resnet18':
            self.backbone = models.resnet18(weights='DEFAULT' if pretrained else None)
        elif backbone_name == 'resnet50':
            self.backbone = models.resnet50(weights='DEFAULT' if pretrained else None)
        elif backbone_name == 'mobilenet_v2':
            self.backbone = models.mobilenet_v2(weights='DEFAULT' if pretrained else None).features
            self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
            num_features = 1280
        else:
            raise ValueError("Backbone not supported")

        if backbone_name.startswith("resnet"):
            # Remove the classification head for ResNet
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()

        # 🧠 Custom feature head (optional)
        self.feature_head = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # 🧩 Output heads
        self.fc_species = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_species)
        )

        self.fc_sex = nn.Sequential(
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_sex)
        )

    def forward(self, x):
        if hasattr(self, 'global_pool'):  # for MobileNet
            x = self.backbone(x)
            x = self.global_pool(x)
            x = x.view(x.size(0), -1)
        else:  # for ResNet
            x = self.backbone(x)

        features = self.feature_head(x)
        return self.fc_species(features), self.fc_sex(features)
