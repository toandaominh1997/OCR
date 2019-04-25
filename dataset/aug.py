import cv2
from PIL import Image
import numpy as np 
import random
from imgaug import augmenters as iaa
from torchvision import transforms
# Transform
def random_dilate(img):
    img = np.array(img)
    img = cv2.dilate(img, np.ones(shape=(random.randint(1,3), random.randint(1,3)), dtype=np.uint8))
    return Image.fromarray(img)

def random_erode(img):
    img = np.array(img)
    img = cv2.erode(img, np.ones(shape=(random.randint(1,3), random.randint(1,3)), dtype=np.uint8))
    return Image.fromarray(img)

class ResizeImage:
    def __init__(self, height):
        self.height = height 
    def __call__(self, img):
        img = np.array(img)
        h,w = img.shape[:2]

        new_w = int(self.height / h * w)
        img = cv2.resize(img, (new_w, self.height), interpolation=cv2.INTER_CUBIC)

        return Image.fromarray(img)

class ImgAugTransform:
    def __init__(self):
        self.aug = iaa.Sequential([
            iaa.Sometimes(0.5,
                          iaa.OneOf(
                              [
                                  iaa.GaussianBlur(sigma=(0, 3.0)),
                                  iaa.AverageBlur(k=(3, 11)),
                                  iaa.MedianBlur(k=(3, 11))
                              ])
                          ),

        ])

    def __call__(self, img):
        img = np.array(img)
        transformed_img =  self.aug.augment_image(img)

        return Image.fromarray(transformed_img)
def train_transforms(height):
    transform = transforms.Compose([

            transforms.RandomApply(
                [
                    random_dilate,
                ],
                p=0.15),

            transforms.RandomApply(
                [
                    random_erode,
                ],
                p=0.15),

            transforms.RandomApply(
                [
                    ImgAugTransform(),
                ],
                p=0.15),
                
            transforms.RandomApply(
                [
                    transforms.Pad(3, fill=255, padding_mode='constant'),
                ],
                p=0.15),

            transforms.RandomApply(
                [
                    transforms.Pad(3, fill=255, padding_mode='reflect'),
                ],
                p=0.15),

            transforms.RandomAffine(degrees=5, scale=(0.9, 1.1), shear=5, resample=Image.NEAREST, fillcolor=255),
            ResizeImage(height),
            transforms.ToTensor()
        ])
    return transform
def test_transforms(height):
    transform = transforms.Compose([
        ResizeImage(height),
        transforms.ToTensor() 
    ])
    return transform
def target_transforms():
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    return transform