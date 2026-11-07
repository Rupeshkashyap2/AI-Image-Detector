from torchvision import transforms
from torchvision.transforms import InterpolationMode

from src.data.preprocessing import IMAGENET_MEAN, IMAGENET_STD, IMG_SIZE

FLIP_P = 0.5
ROTATION_DEGREES = 5          # small: larger angles create black corner wedges
CROP_SCALE = (0.85, 1.0)      # keep at least 85% of the image area
CROP_RATIO = (0.95, 1.05)     # images are square, so stay near square
BRIGHTNESS = 0.1
CONTRAST = 0.1


def get_train_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.RandomResizedCrop(
            IMG_SIZE, scale=CROP_SCALE, ratio=CROP_RATIO,
            interpolation=InterpolationMode.BILINEAR,
        ),
        transforms.RandomHorizontalFlip(p=FLIP_P),
        transforms.RandomRotation(ROTATION_DEGREES, interpolation=InterpolationMode.BILINEAR),
        transforms.ColorJitter(brightness=BRIGHTNESS, contrast=CONTRAST),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])