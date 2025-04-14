# /home/ponsankar/mde/model.py
import torch
import torch.nn as nn
import torchvision.models as models

class DepthModel(nn.Module):
    def __init__(self, task="depth_prediction"):
        super(DepthModel, self).__init__()
        self.task = task
        # Encoder: ResNet18
        self.encoder = models.resnet18(weights="ResNet18_Weights.IMAGENET1K_V1")
        self.encoder_channels = [64, 64, 128, 256, 512]  # Include layer1 for clarity

        # Modify input for depth completion (RGB + sparse depth)
        if task == "depth_completion":
            self.encoder.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Decoder
        self.decoder = nn.ModuleList([
            nn.ConvTranspose2d(self.encoder_channels[-1], 256, 4, stride=2, padding=1),  # 512 -> 256
            nn.ConvTranspose2d(256 + self.encoder_channels[-2], 128, 4, stride=2, padding=1),  # 256+256 -> 128
            nn.ConvTranspose2d(128 + self.encoder_channels[-3], 64, 4, stride=2, padding=1),  # 128+128 -> 64
            nn.ConvTranspose2d(64 + self.encoder_channels[-4], 32, 4, stride=2, padding=1),  # 64+64 -> 32
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),  # 32 -> 16
            nn.Conv2d(16, 1, 3, padding=1)  # 16 -> 1
        ])
        self.relu = nn.ReLU(inplace=True)

    def forward(self, image, sparse_depth=None):
        # Prepare input
        if self.task == "depth_completion" and sparse_depth is not None:
            x = torch.cat((image, sparse_depth), dim=1)  # (B, 4, H, W)
        else:
            x = image  # (B, 3, H, W)

        # Encoder
        features = []
        x = self.encoder.conv1(x)  # 352x1216 -> 176x608
        x = self.encoder.bn1(x)
        x = self.encoder.relu(x)
        x = self.encoder.maxpool(x)  # 176x608 -> 88x304
        features.append(x)  # 64 channels, 88x304
        x = self.encoder.layer1(x)  # 88x304
        features.append(x)  # 64 channels, 88x304
        x = self.encoder.layer2(x)  # 88x304 -> 44x152
        features.append(x)  # 128 channels, 44x152
        x = self.encoder.layer3(x)  # 44x152 -> 22x76
        features.append(x)  # 256 channels, 22x76
        x = self.encoder.layer4(x)  # 22x76 -> 11x38
        features.append(x)  # 512 channels, 11x38

        # Decoder
        x = self.decoder[0](x)  # 11x38 -> 22x76 (256 channels)
        x = self.relu(x)
        x = torch.cat((x, features[3]), dim=1)  # Concat 256 (decoder) + 256 (layer3), 22x76
        x = self.decoder[1](x)  # 22x76 -> 44x152 (128 channels)
        x = self.relu(x)
        x = torch.cat((x, features[2]), dim=1)  # Concat 128 (decoder) + 128 (layer2), 44x152
        x = self.decoder[2](x)  # 44x152 -> 88x304 (64 channels)
        x = self.relu(x)
        x = torch.cat((x, features[1]), dim=1)  # Concat 64 (decoder) + 64 (layer1), 88x304
        x = self.decoder[3](x)  # 88x304 -> 176x608 (32 channels)
        x = self.relu(x)
        x = self.decoder[4](x)  # 176x608 -> 352x1216 (16 channels)
        x = self.relu(x)
        x = self.decoder[5](x)  # 352x1216 -> 352x1216 (1 channel)

        # Ensure positive depth
        depth = torch.sigmoid(x) * 80.0  # Scale to max depth (e.g., 80m)
        return depth