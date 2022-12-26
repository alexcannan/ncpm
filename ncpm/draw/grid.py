import itertools
import math

import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm

from ncpm.draw.color import color_generator
from ncpm.draw.square import draw_square


def draw_square_grid(grid_size: int, x_nodes: int, y_nodes: int, color_type: str="prism", *args, **kwargs):
    tile_width = 300
    tile_height = 300
    im = Image.new("RGB", (grid_size * tile_width, grid_size * tile_height), "black")
    for i, j in tqdm(itertools.product(range(grid_size), range(grid_size)), desc="drawing tiles", total=grid_size**2):
        im.paste(draw_square(x_nodes, y_nodes, width=tile_width, height=tile_height, *args, **kwargs), (i * tile_width, j * tile_height))
    grid_width = 6000
    grid_height = 6000
    im = im.resize((grid_width, grid_height), resample=Image.Resampling.BICUBIC)
    if color_type:
        xbuf = im.width / grid_size / (x_nodes+1) / 2
        ybuf = im.height / grid_size / (y_nodes+1) / 2
        horiz_color_points = itertools.product(np.linspace(xbuf, im.width-xbuf, (x_nodes+1) * grid_size, dtype=int), np.linspace(0, im.height, grid_size+1, dtype=int))
        vert_color_points = itertools.product(np.linspace(0, im.width, grid_size+1, dtype=int), np.linspace(ybuf, im.height-ybuf, (y_nodes+1) * grid_size, dtype=int))
        color_points = set(horiz_color_points) | set(vert_color_points)
        color_points = sorted(color_points, key=lambda x: math.sqrt(x[0]**2 + x[1]**2))
        color_iter = color_generator(color_type)
        for x, y in tqdm(color_points, desc="filling colors"):
            x = min(x, im.width - 1)
            y = min(y, im.height - 1)
            if im.getpixel((x, y)) == (0, 0, 0):
                ImageDraw.floodfill(im, (x, y), next(color_iter))
        # find any remaining black areas and fill them in
        for x, y in tqdm(itertools.product(range(im.width), range(im.height)), desc="filling remaining colors", total=im.width*im.height):
            if im.getpixel((x, y)) == (0, 0, 0):
                ImageDraw.floodfill(im, (x, y), next(color_iter))
    return im
