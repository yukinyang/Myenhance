import glob
import os
import torchvision.transforms as transforms

from torch.utils.data import Dataset
from PIL import Image
import cv2


def to_rgb(image):
    rgb_image = Image.new("RGB", image.size)
    rgb_image.paste(image)
    return rgb_image


class ImageDataset(Dataset):
    def __init__(self, root, transform_=None):
        self.files = sorted(glob.glob(root + "/*.*"))
        # print(len(self.files))
        self.transform_ = transforms.Compose(transform_)

    def __getitem__(self, index):
        # print(index)
        image_ = Image.open(self.files[index % len(self.files)])

        if image_.mode != "RGB":
            image_ = to_rgb(image_)

        item = self.transform_(image_)
        return {'img': item}

    def __len__(self):
        return len(self.files)