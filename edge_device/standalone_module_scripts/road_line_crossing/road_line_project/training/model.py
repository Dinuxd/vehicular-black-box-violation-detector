from __future__ import annotations

import argparse

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2

from config import INPUT_HEIGHT, INPUT_WIDTH, NUM_CLASSES


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = ConvBlock(in_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class MobileNetV2UNet(nn.Module):
    def __init__(self, num_classes: int = NUM_CLASSES, pretrained: bool = True) -> None:
        super().__init__()
        weights = None
        if pretrained:
            try:
                weights = MobileNet_V2_Weights.IMAGENET1K_V1
            except Exception:
                weights = None

        try:
            backbone = mobilenet_v2(weights=weights)
        except Exception as exc:
            print(f"WARNING: could not load pretrained MobileNetV2 weights ({exc}). Using random encoder weights.")
            backbone = mobilenet_v2(weights=None)

        features = backbone.features
        self.encoder0 = features[0]      # 32 channels, 1/2 resolution
        self.encoder1 = features[1]      # 16 channels, 1/2 resolution
        self.encoder2 = features[2:4]    # 24 channels, 1/4 resolution
        self.encoder3 = features[4:7]    # 32 channels, 1/8 resolution
        self.encoder4 = features[7:14]   # 96 channels, 1/16 resolution
        self.encoder5 = features[14:]    # 1280 channels, 1/32 resolution

        self.up4 = UpBlock(1280, 96, 256)
        self.up3 = UpBlock(256, 32, 128)
        self.up2 = UpBlock(128, 24, 64)
        self.up1 = UpBlock(64, 16, 48)
        self.final_up = ConvBlock(48, 32)
        self.segmentation_head = nn.Conv2d(32, num_classes, kernel_size=1)

    def encoder_parameters(self):
        for module in [self.encoder0, self.encoder1, self.encoder2, self.encoder3, self.encoder4, self.encoder5]:
            yield from module.parameters()

    def decoder_parameters(self):
        for module in [self.up4, self.up3, self.up2, self.up1, self.final_up, self.segmentation_head]:
            yield from module.parameters()

    def freeze_encoder(self) -> None:
        for parameter in self.encoder_parameters():
            parameter.requires_grad = False

    def unfreeze_encoder(self) -> None:
        for parameter in self.encoder_parameters():
            parameter.requires_grad = True

    def unfreeze_encoder_from(self, layer_index: int = 14) -> None:
        self.freeze_encoder()
        encoder_modules = [self.encoder0, self.encoder1, self.encoder2, self.encoder3, self.encoder4, self.encoder5]
        flat_modules = []
        for module in encoder_modules:
            if isinstance(module, nn.Sequential):
                flat_modules.extend(list(module.children()))
            else:
                flat_modules.append(module)
        for index, module in enumerate(flat_modules):
            if index >= layer_index:
                for parameter in module.parameters():
                    parameter.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[-2:]
        x0 = self.encoder0(x)
        x1 = self.encoder1(x0)
        x2 = self.encoder2(x1)
        x3 = self.encoder3(x2)
        x4 = self.encoder4(x3)
        x5 = self.encoder5(x4)

        x = self.up4(x5, x4)
        x = self.up3(x, x3)
        x = self.up2(x, x2)
        x = self.up1(x, x1)
        x = F.interpolate(x, size=input_size, mode="bilinear", align_corners=False)
        x = self.final_up(x)
        return self.segmentation_head(x)


def build_model(num_classes: int = NUM_CLASSES, pretrained: bool = True) -> MobileNetV2UNet:
    return MobileNetV2UNet(num_classes=num_classes, pretrained=pretrained)


def count_parameters(model: nn.Module) -> tuple[int, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return total, trainable


def main() -> int:
    parser = argparse.ArgumentParser(description="MobileNetV2-UNet forward-pass sanity check.")
    parser.add_argument("--no-pretrained", action="store_true")
    args = parser.parse_args()
    model = build_model(pretrained=not args.no_pretrained)
    model.eval()
    total, trainable = count_parameters(model)
    print(f"Parameters: total={total:,}, trainable={trainable:,}")
    x = torch.randn(1, 3, INPUT_HEIGHT, INPUT_WIDTH)
    with torch.no_grad():
        y = model(x)
    print(f"Input shape: {tuple(x.shape)}")
    print(f"Output shape: {tuple(y.shape)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
