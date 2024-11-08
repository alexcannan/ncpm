import itertools
import random
import sys

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

from ncpm.draw.color import equicolors
from ncpm.draw.square import draw_square


def draw_square_grid(grid_size: int, x_nodes: int, y_nodes: int, smoothing_dim: int, *args, **kwargs):
    tile_width = 300
    tile_height = 300
    im = Image.new("RGB", (grid_size * tile_width, grid_size * tile_height), "black")
    for i, j in tqdm(itertools.product(range(grid_size), range(grid_size)), desc="drawing tiles", total=grid_size**2):
        im.paste(draw_square(x_nodes, y_nodes, width=tile_width, height=tile_height, *args, **kwargs),
                 (i * tile_width, j * tile_height))

    grid_width = 6000
    grid_height = 6000
    im = im.resize((grid_width, grid_height), resample=Image.Resampling.BICUBIC)

    img_array = np.array(im.convert("L"))
    img_binary = np.where(img_array == 0, 1, 0).astype(np.uint8)

    num_labels, labels = cv2.connectedComponents(img_binary)

    colors = np.array(equicolors(num_labels, **kwargs), dtype=np.uint8)
    np.random.shuffle(colors)
    colors = np.insert(colors, 0, [0, 0, 0], axis=0)

    colored_img = colors[labels]  # Direct mapping from labels to colors via advanced indexing

    # antialiasing
    kernel = np.ones((smoothing_dim,smoothing_dim),np.float32)/(smoothing_dim**2)
    smoothed_colored_img = cv2.filter2D(colored_img,-1,kernel)

    return Image.fromarray(smoothed_colored_img)
