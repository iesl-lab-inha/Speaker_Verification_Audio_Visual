import torch
import torch.nn as nn
import torchvision.models as models


class VisualLipResNet18Encoder(nn.Module):
    def __init__(self, output_dim=256, pretrained=False, grayscale=True):
        super().__init__()

        resnet = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT if pretrained else None
        )

        if grayscale:
            old_conv = resnet.conv1
            resnet.conv1 = nn.Conv2d(
                1,
                old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=False,
            )
            if pretrained:
                with torch.no_grad():
                    resnet.conv1.weight[:] = old_conv.weight.mean(dim=1, keepdim=True)

        feature_dim = resnet.fc.in_features
        resnet.fc = nn.Identity()

        self.backbone = resnet
        self.proj = nn.Linear(feature_dim, output_dim)

    def forward(self, video):
        """
        video: [B, T, C, H, W]
        returns: [B, T, D]
        """
        b, t, c, h, w = video.shape
        x = video.reshape(b * t, c, h, w)
        x = self.backbone(x)
        x = self.proj(x)
        x = x.view(b, t, -1)
        return x
