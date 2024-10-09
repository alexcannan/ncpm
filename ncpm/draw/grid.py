import itertools
import random
import sys

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

from ncpm.draw.color import equicolors
from ncpm.draw.square import draw_square


def draw_square_grid(grid_size: int, x_nodes: int, y_nodes: int, *args, **kwargs):
    tile_width = 300
    tile_height = 300
    im = Image.new("RGB", (grid_size * tile_width, grid_size * tile_height), "black")
    for i, j in tqdm(itertools.product(range(grid_size), range(grid_size)), desc="drawing tiles", total=grid_size**2):
        im.paste(draw_square(x_nodes, y_nodes, width=tile_width, height=tile_height, *args, **kwargs), (i * tile_width, j * tile_height))
    grid_width = 6000
    grid_height = 6000
    im = im.resize((grid_width, grid_height), resample=Image.Resampling.BICUBIC)

    img_array = np.array(im.convert("L"))  # Convert to grayscale for connected components
    img_binary = np.where(img_array == 0, 1, 0).astype(np.uint8)  # Invert: 0 for background, 1 for foreground
    num_labels, labels = cv2.connectedComponents(img_binary)

    print(f"{num_labels=}", file=sys.stderr)
    colors = equicolors(num_labels, **kwargs)
    random.shuffle(colors)
    label_colors = {label: color for label, color in zip(range(1, num_labels), colors)}

    colored_img = np.zeros((*img_array.shape, 3), dtype=np.uint8)  # Prepare an RGB array
    for label in range(1, num_labels):
        colored_img[labels == label] = label_colors[label]  # Assign color to the label region

    im_colored = Image.fromarray(colored_img)

    return im_colored
