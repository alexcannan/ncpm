"""
given nodes on the edges of a square, generates a non-crossing perfect matching and draws with bezier curves
"""

import itertools
import random

from PIL import Image, ImageDraw


def generate_matching(n_nodes: int) -> list[tuple[int]]:
    """ given a number of nodes, generates a non-crossing perfect matching """
    assert n_nodes % 2 == 0, "number of nodes must be even"
    nodes = list(range(n_nodes))
    matches = []
    selected_nodes = set()
    random.shuffle(nodes)
    matches = [(nodes[i], nodes[i + 1]) for i in range(0, n_nodes, 2)]
    return matches


def draw_square(x_nodes: int=3, y_nodes: int=3):
    width = 1000
    height = 1000
    im = Image.new("RGB", (width, height), "black")
    draw = ImageDraw.Draw(im)
    n_nodes = 2 * x_nodes + 2 * y_nodes
    # draw nodes on edges
    x_spacing = width / (x_nodes + 1)
    for i in range(x_nodes):
        x = (i + 1) * x_spacing
        draw.ellipse((x - 5, -5, x + 5, 5), fill="white")
        draw.ellipse((x - 5, height - 5, x + 5, height + 5), fill="white")
    y_spacing = height / (y_nodes + 1)
    for i in range(y_nodes):
        y = (i + 1) * y_spacing
        draw.ellipse((-5, y - 5, 5, y + 5), fill="white")
        draw.ellipse((width - 5, y - 5, width + 5, y + 5), fill="white")
    matches = generate_matching(n_nodes)
    for match in matches:
        edge_iterator = [(x_spacing, 0)] * x_nodes + [(x_spacing, y_spacing)] + [(0, y_spacing)] * (y_nodes-1) + [(-x_spacing, y_spacing)] + [(-x_spacing, 0)] * (x_nodes-1) + [(-x_spacing, -y_spacing)] + [(0, -y_spacing)] * (y_nodes-1)
        edge_coordinates = []
        prev_coord = (0, 0)
        for coord in edge_iterator:
            edge_coordinates.append((prev_coord[0] + coord[0], prev_coord[1] + coord[1]))
            prev_coord = edge_coordinates[-1]
        print('edge_coords', edge_coordinates)
        p0 = edge_coordinates[match[0]]
        p1 = edge_coordinates[match[1]]
        print(p0, p1)
        draw.line((p0[0], p0[1], p1[0], p1[1]), fill="white", width=4)
    im.show()


if __name__ == "__main__":
    draw_square(x_nodes=5, y_nodes=5)