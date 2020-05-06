import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from matplotlib.path import Path
from torchvision.transforms.functional import resize, to_pil_image
import cv2
from albumentations.pytorch import ToTensor
from albumentations import *
import random


class Znacky_set(Dataset):

    def __init__(self,
                shape,
                X,
                y=None,
                DIR=None,
                aug_pool=None):
        
        self.aug_pool = aug_pool
        self.X = X
        self.y = y
        self.DIR = DIR
        self.shape = shape

    def __getitem__(self, index):

        img_path = self.X[index]
        image = self.get_image(img_path)
        vertices = self.y[img_path]
        mask = self.create_mask(vertices, img_path)

        resize = [Resize(self.shape[1], self.shape[0], interpolation=1, p=1)]
        preprocessing = [Normalize(mean=(0.485, 0.456,0.406),
            std=(0.229, 0.224, 0.225), max_pixel_value=255.0, p=1),
            ToTensor()]

        if self.aug_pool:
            aug = [random.choice(self.aug_pool)]
            transforms = Compose(resize + aug + preprocessing)
            augmented = transforms(image=image, masks=mask)
            image = augmented['image']
            mask = augmented['masks']
        else:
            transforms = Compose(resize + preprocessing)
            augmented = transforms(image=image, masks=mask)
            image = augmented['image']
            mask = augmented['masks']

        return image, np.expand_dims(np.array(mask[0]),0), img_path

    def __len__(self):
        return len(self.X)

    def get_image(self, img_path):
        
        image = cv2.imread(self.DIR + img_path)
        image = np.asarray(cv2.cvtColor(image,
            cv2.COLOR_BGR2RGB)).astype('uint8')

        return image

    def create_mask(self, vertices, img_path):
        
        mask_whole = np.zeros((1, self.shape[0], self.shape[1]))

        for vertex in vertices:

            vertex = np.asarray(vertex)

            img = np.array(Image.open(self.DIR + img_path))

            path = Path(vertex)
            xmin, ymin, xmax, ymax = np.asarray(path.get_extents(),
                    dtype=int).ravel()

            x, y = np.mgrid[:img.shape[1], :img.shape[0]]

            points = np.vstack((x.ravel(), y.ravel())).T

            mask = path.contains_points(points)
            path_points = points[np.where(mask)]

            img_mask = mask.reshape(x.shape).T.astype('float32')
            img_mask = np.asarray(resize(to_pil_image(img_mask),
                self.shape))
            img_mask = np.expand_dims(img_mask, 0)

            mask_copy = img_mask.copy()
            mask_whole += mask_copy

        return mask_whole
