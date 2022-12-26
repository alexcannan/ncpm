import warnings

import numpy as np
from PIL import ImageDraw

from ncpm import square_pair_type, square_relative_coordinates, node_distance
from ncpm.draw import line_width


def draw_formulaic_curve(draw: ImageDraw, x_nodes: int, y_nodes: int, x_node: int, y_node: int):
    """ for each node pair, draw a certain curve depending on the case (same edge, adjacent edge, opposite edge) """
    pair_type = square_pair_type(x_node, y_node, x_nodes, y_nodes)
    width = draw._image.width
    height = draw._image.height
    lr = line_width // 2
    r1 = square_relative_coordinates(x_node, x_nodes, y_nodes)
    r2 = square_relative_coordinates(y_node, x_nodes, y_nodes)
    x1 = r1[0] * width; y1 = r1[1] * height
    x2 = r2[0] * width; y2 = r2[1] * height
    if pair_type == "same":
        # draw a semicircle
        cx = (x1 + x2) / 2; cy = (y1 + y2) / 2
        r = max(abs(x1-cx), abs(y1-cy))
        draw.ellipse((cx-r-lr, cy-r-lr, cx+r+lr, cy+r+lr), outline="white", width=line_width)
    elif pair_type == "adjacent":
        # draw a quarter circle
        # discover what corner the nodes are adjacent to
        min_x = min(x1, x2)
        min_y = min(y1, y2)
        max_x = max(x1, x2)
        max_y = max(y1, y2)
        if np.isclose(min_x, 0) and np.isclose(min_y, 0):
            cx = min_x; cy = min_y
        elif np.isclose(min_x, 0) and np.isclose(max_y, height):
            cx = min_x; cy = max_y
        elif np.isclose(max_x, width) and np.isclose(min_y, 0):
            cx = max_x; cy = min_y
        elif np.isclose(max_x, width) and np.isclose(max_y, height):
            cx = max_x; cy = max_y
        else:
            warnings.warn(f"could not determine corner for adjacent nodes {x_node} and {y_node} ({x_nodes} {y_nodes}). drawing line.")
            draw.line((x1, y1, x2, y2), fill="white", width=line_width)
            return
        x_spacing = max(abs(x1-cx), abs(x2-cx))
        y_spacing = max(abs(y1-cy), abs(y2-cy))
        draw.ellipse((cx-x_spacing-lr, cy-y_spacing-lr, cx+x_spacing+lr, cy+y_spacing+lr), outline="white", width=line_width)
    elif pair_type == "opposite":
        draw.line((x1, y1, x2, y2), fill="white", width=line_width)
    return
