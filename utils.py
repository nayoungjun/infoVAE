import math

import numpy as np
import torch

def cycle(iterable):
    while True:
        for item in iterable:
            yield item


def tile_images(img, img_size=32, rows=4, cols=4, spacing=1):
    """
    Return the input and reconstructed images side by side as tiled images for visualization

    Returns
    -------
    np.ndarray, shape = [rows * (2 * input[1]+ spacing) - spacing, cols * (input[2] + spacing) - spacing, 1]
    """
    images = np.ones([3, rows * (img_size + spacing) - spacing, cols * (img_size + spacing)], dtype=np.float32)
    coords = [(i, j) for i in range(rows) for j in range(cols)]

    for (i, j), image in zip(coords, img):
        x = i * (img_size + spacing)
        y = j * (img_size + spacing)
        images[:, x: x+img_size, y:y+img_size] = image

    return images

