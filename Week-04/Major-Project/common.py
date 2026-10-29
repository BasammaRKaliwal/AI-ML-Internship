"""Model definitions, model settings and the image preprocessing shared by training, evaluation and the app."""
import torch
import torch.nn as nn
from torchvision import models, transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# The three configurations that are trained and compared.
CONFIGS = {
    "baseline": dict(title="Baseline CNN", arch="baseline", size=128, augment=False, epochs=15, lr=1e-3,
                     weight_decay=0.0,
                     description="3 convolution layers and 2 dense layers, no augmentation, no regularisation."),
    "improved": dict(title="Improved CNN", arch="improved", size=128, augment=True, epochs=25, lr=2e-3,
                     weight_decay=1e-4,
                     description="4 blocks of Conv + BatchNorm + ReLU + MaxPool, global average pooling, dropout 0.4, "
                                 "data augmentation (crop, flip, rotation, colour jitter) and weight decay."),
    "transfer": dict(title="Pretrained MobileNetV2", arch="transfer", size=224, augment=True, epochs=8, lr=1e-3,
                     weight_decay=1e-4,
                     description="MobileNetV2 pretrained on ImageNet: first only the new classifier is trained, then "
                                 "the last 4 blocks are fine-tuned. Same augmentation as the improved CNN."),
}


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def eval_transform(size):
    """The preprocessing used for validation, testing AND the app (so predictions match training)."""
    return transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor(),
                               transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])


def train_transform(size, augment):
    if not augment:
        return eval_transform(size)
    return transforms.Compose([
        transforms.RandomResizedCrop(size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class BaselineCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        def block(i, o):
            return [nn.Conv2d(i, o, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)]
        self.features = nn.Sequential(*block(3, 16), *block(16, 32), *block(32, 64))   # 128 -> 16 pixels
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(64 * 16 * 16, 64), nn.ReLU(), nn.Linear(64, num_classes))

    def forward(self, x):
        return self.classifier(self.features(x))


class ImprovedCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        def block(i, o):
            return [nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(), nn.MaxPool2d(2)]
        self.features = nn.Sequential(*block(3, 32), *block(32, 64), *block(64, 128), *block(128, 256))
        self.classifier = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.4), nn.Linear(256, num_classes))

    def forward(self, x):
        return self.classifier(self.features(x))


def build_model(arch, num_classes, pretrained=False):
    if arch == "baseline":
        return BaselineCNN(num_classes)
    if arch == "improved":
        return ImprovedCNN(num_classes)
    if arch == "transfer":
        weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v2(weights=weights)
        model.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(model.last_channel, num_classes))
        return model
    raise ValueError(f"unknown model type: {arch}")


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def load_checkpoint(path, device=None):
    """Load a saved model. Returns (model in eval mode, checkpoint dict with classes, size, ...)."""
    device = device or get_device()
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location="cpu")
    model = build_model(ckpt["arch"], len(ckpt["classes"]), pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval(), ckpt
