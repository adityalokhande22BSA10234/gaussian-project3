"""Three CNN regression architectures for 4-parameter Gaussian beam fitting.

Every model outputs 5 values: a beam-presence logit (for the ambient
background/empty frames in the dataset) plus 4 sigmoid-bounded, normalized
regression targets [x, y, sigma_x, sigma_y]. They differ only in backbone
capacity/depth, so they're interchangeable via `build_model(name)`.

    tiny      - 3-block ConvNet, smallest and fastest, lowest accuracy ceiling.
    medium    - 4-block ConvNet (the default). Balanced accuracy/speed.
    resnet18  - torchvision ResNet-18, adapted for 1-channel input. Highest
                capacity, slowest to train.
"""
import torch
import torch.nn as nn


class _PresenceRegressionHead(nn.Module):
    """Shared output convention: logit for presence + sigmoid for 4 params."""

    @staticmethod
    def split(raw):
        presence_logit = raw[:, 0]
        params = torch.sigmoid(raw[:, 1:])
        return presence_logit, params


class TinyCNN(nn.Module):
    """Lightweight 3-block ConvNet. Cheapest to train, use for quick iteration."""

    def __init__(self, in_channels=1):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(4),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128), nn.ReLU(inplace=True), nn.Dropout(0.2),
            nn.Linear(128, 5),
        )

    def forward(self, x):
        raw = self.head(self.features(x))
        return _PresenceRegressionHead.split(raw)


class MediumCNN(nn.Module):
    """4-block custom ConvNet -- the default balanced choice."""

    def __init__(self, in_channels=1):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(4),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256), nn.ReLU(inplace=True), nn.Dropout(0.3),
            nn.Linear(256, 5),
        )

    def forward(self, x):
        raw = self.head(self.features(x))
        return _PresenceRegressionHead.split(raw)


class ResNet18Regressor(nn.Module):
    """torchvision ResNet-18 backbone, first conv swapped for 1-channel input.

    Trained from scratch (weights=None) since ImageNet weights don't transfer
    to single-channel synthetic beam images.
    """

    def __init__(self):
        super().__init__()
        import torchvision.models as models
        net = models.resnet18(weights=None)
        net.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        net.fc = nn.Linear(net.fc.in_features, 5)
        self.backbone = net

    def forward(self, x):
        raw = self.backbone(x)
        return _PresenceRegressionHead.split(raw)


MODEL_REGISTRY = {
    "tiny": TinyCNN,
    "medium": MediumCNN,
    "resnet18": ResNet18Regressor,
    "custom": MediumCNN,  # backward-compat alias for older checkpoints
}


def build_model(name="medium"):
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Choose from {sorted(set(MODEL_REGISTRY) - {'custom'})}")
    return MODEL_REGISTRY[name]()


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
