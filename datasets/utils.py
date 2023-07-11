# load img 
import cv2 
from torchvision.transforms import transforms
import albumentations as A

import numpy as np 
import _init_path
import configs.constants as C  # relative impor: cannot do import constants as C 

def load_img(img_path, crop_size, return_gray = False):
    if return_gray:
        img = cv2.imread(img_path, cv2.COLOR_BGR2GRAY)
    else:
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)

    if crop_size:
        img = cv2.resize(img, crop_size, interpolation = cv2.INTER_NEAREST)

    return img 


def load_label(label_path, crop_size):
    label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
    if crop_size:
        label = cv2.resize(label, crop_size, interpolation = cv2.INTER_NEAREST)

    return label

    

def heavy_transform(crop_size):
    return A.Compose([
        A.HorizontalFlip(),
        A.RandomScale(scale_limit = (0.2, 2.0), p = 0.5),
        A.RandomCrop(width = crop_size[0], height = crop_size[1]),

        ], 
        additional_targets={'label': 'image'}
        )


def basic_transform(np_image):

    np_image = np_image.astype(np.float32)[:, :, ::-1]
    np_image = np_image / 255.0
    np_image -= C.mean
    np_image /= C.std
    return np_image.transpose(2, 1, 0) # h, w, c --> c, w, h



def id_to_rgb(label, num_classes):
    rgb_label = np.zeros((label.shape[0], label.shape[1], 3))
    for i in range(num_classes):
        rgb_label[label == i] = C.index_to_color[i]
    rgb_label = rgb_label.astype('uint8')

    return rgb_label