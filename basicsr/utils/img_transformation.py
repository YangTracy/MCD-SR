import numpy as np
import cv2
import torch

import random
from scipy import ndimage
import scipy
import scipy.stats as ss
from scipy.interpolate import interp2d
from scipy.linalg import orth


def add_Gaussian_noise(img, noise_level):
    img += np.random.normal(0, noise_level, img.shape).astype(np.float32)
    img = np.clip(img, 0.0, 255.0)
    return img
