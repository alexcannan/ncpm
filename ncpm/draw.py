"""
given nodes on the edges of a square, generates a non-crossing perfect matching and draws with bezier curves
"""

import itertools
import math
import random

import matplotlib.cm as cm
import numpy as np
from PIL import Image, ImageDraw, ImageColor
from tqdm import tqdm


def generate_matching(n_nodes: int) -> list[tuple[int]]:
    """ given a number of nodes, generates a non-crossing perfect matching """
    assert n_nodes % 2 == 0, "number of nodes must be even"
    nodes = list(range(n_nodes))
    def distance_between_nodes(node1: int, node2: int) -> int:
        return abs(node1 - node2) % n_nodes
    matches = []
    selected_nodes = set()
    while (remaining_nodes := list(set(nodes) - selected_nodes)):
        if len(remaining_nodes) == 2:
            # XXX: this is a hack, sometimes this gets stuck on last 2 but unsure why
            matches.append(tuple(remaining_nodes))
            break
        pair = sorted(random.sample(remaining_nodes, 2))
        # if any of the nodes in between are already selected, try again
        if any(node in selected_nodes for node in range(*pair)):
            continue
        # if the number of unselected nodes on either side of the pair, before any other selected node, is odd, try again
        left_iter = (pair[0] - 1) % n_nodes
        left_count = 0
        while nodes[left_iter] not in selected_nodes and nodes[left_iter] != pair[1]:
            left_count += 1
            left_iter = (left_iter - 1) % n_nodes
        right_iter = (pair[1] + 1) % n_nodes
        right_count = 0
        while nodes[right_iter] not in selected_nodes and nodes[right_iter] != pair[0]:
            right_count += 1
            right_iter = (right_iter + 1) % n_nodes
        if left_count % 2 == 1 or right_count % 2 == 1:
            continue
        # if the distance between the indices is odd, that means an even number between them, add to matches
        dist = distance_between_nodes(*pair)
        if dist == 1:
            # if the distance is 1, try and choose another a certain % of the time
            if random.random() < 0.9:
                continue
        if dist % 2 == 1:
            print(f"adding {pair} with distance {dist}")
            matches.append(pair)
            selected_nodes.update(pair)
        else:
            continue
    return matches


def pascal_row(n, memo={}):
    # https://stackoverflow.com/questions/246525/how-can-i-draw-a-bezier-curve-using-pythons-pil
    # This returns the nth row of Pascal's Triangle
    if n in memo:
        return memo[n]
    result = [1]
    x, numerator = 1, n
    for denominator in range(1, n//2+1):
        # print(numerator,denominator,x)
        x *= numerator
        x /= denominator
        result.append(x)
        numerator -= 1
    if n&1 == 0:
        # n is even
        result.extend(reversed(result[:-1]))
    else:
        result.extend(reversed(result))
    memo[n] = result
    return result


def make_bezier(xys):
    # https://stackoverflow.com/questions/246525/how-can-i-draw-a-bezier-curve-using-pythons-pil
    # xys should be a sequence of 2-tuples (Bezier control points)
    n = len(xys)
    combinations = pascal_row(n-1)
    def bezier(ts):
        # This uses the generalized formula for bezier curves
        # http://en.wikipedia.org/wiki/B%C3%A9zier_curve#Generalization
        result = []
        for t in ts:
            tpowers = (t**i for i in range(n))
            upowers = reversed([(1-t)**i for i in range(n)])
            coefs = [c*a*b for c, a, b in zip(combinations, tpowers, upowers)]
            result.append(
                tuple(sum([coef*p for coef, p in zip(coefs, ps)]) for ps in zip(*xys)))
        return result
    return bezier


def draw_square(x_nodes: int=3, y_nodes: int=3, draw_points: bool=False, samples: int=100):
    width = 1000
    height = 1000
    max_distance = math.sqrt(width ** 2 + height ** 2)
    im = Image.new("RGB", (width, height), "black")
    draw = ImageDraw.Draw(im)
    n_nodes = 2 * x_nodes + 2 * y_nodes
    x_spacing = width / (x_nodes + 1)
    y_spacing = height / (y_nodes + 1)
    ts = [t/samples for t in range(samples+1)]
    if draw_points:
        # draw nodes on edges
        for i in range(x_nodes):
            x = (i + 1) * x_spacing
            draw.ellipse((x - 5, -5, x + 5, 5), fill="white")
            draw.ellipse((x - 5, height - 5, x + 5, height + 5), fill="white")
        for i in range(y_nodes):
            y = (i + 1) * y_spacing
            draw.ellipse((-5, y - 5, 5, y + 5), fill="white")
            draw.ellipse((width - 5, y - 5, width + 5, y + 5), fill="white")
    # get coordinates of nodes on edges
    edge_iterator = [(x_spacing, 0)] * x_nodes + [(x_spacing, y_spacing)] + [(0, y_spacing)] * (y_nodes-1) + [(-x_spacing, y_spacing)] + [(-x_spacing, 0)] * (x_nodes-1) + [(-x_spacing, -y_spacing)] + [(0, -y_spacing)] * (y_nodes-1)
    control_iterator = [(x_spacing, y_spacing)] + [(x_spacing, 0)] * (x_nodes-1) + [(0,0)] + [(0, y_spacing)] * (y_nodes-1) + [(0,0)] + [(-x_spacing, 0)] * (x_nodes-1) + [(0,0)] + [(0, -y_spacing)] * (y_nodes-1)
    edge_coordinates = []
    prev_coord = (0, 0)
    for coord in edge_iterator:
        edge_coordinates.append((prev_coord[0] + coord[0], prev_coord[1] + coord[1]))
        prev_coord = edge_coordinates[-1]
    control_coordinates = []
    prev_coord = (0, 0)
    for coord in control_iterator:
        control_coordinates.append((prev_coord[0] + coord[0], prev_coord[1] + coord[1]))
        prev_coord = control_coordinates[-1]
    # generate matchings and draw
    matches = generate_matching(n_nodes)
    for match in matches:
        p0 = edge_coordinates[match[0]]
        p1 = edge_coordinates[match[1]]
        c0 = control_coordinates[match[0]]
        c1 = control_coordinates[match[1]]
        # adjust control points towards center the greater the distance between nodes are
        match_distance = math.sqrt((p0[0] - p1[0])**2 + (p0[1] - p1[1])**2)
        control_bias = -0.3
        control_weight = 0.5
        control_adjustment = (match_distance / max_distance + control_bias) * control_weight
        c0 = (c0[0] + (width / 2 - c0[0]) * control_adjustment, c0[1] + (height / 2 - c0[1]) * control_adjustment)
        c1 = (c1[0] + (width / 2 - c1[0]) * control_adjustment, c1[1] + (height / 2 - c1[1]) * control_adjustment)
        bezier = make_bezier([p0, c0, c1, p1])
        points = bezier(ts)
        for i in range(len(points) - 1):
            draw.line((points[i], points[i+1]), fill="white", width=2)
    return im


def color_generator(type: str="discrete"):
    if type == "discrete":
        for color in itertools.cycle(["red", "green", "blue", "yellow", "purple", "orange", "pink", "cyan", "magenta"]):
            yield ImageColor.getrgb(color)
    elif type == "grayscale":
        for i in itertools.cycle(itertools.chain(range(1, 256), range(255, 0, -1))):
            yield (i, i, i)
    elif type == "rainbow":
        for i in itertools.cycle(range(256)):
            yield ImageColor.getrgb(f"hsl({i}, 100%, 50%)")
    elif type == "rainbow2":
        for i in itertools.cycle(range(256)):
            yield ImageColor.getrgb(f"hsl({i}, 100%, 50%)")
            yield ImageColor.getrgb(f"hsl({i+128}, 100%, 50%)")
    elif type == "rainbow3":
        for i in itertools.cycle(range(256)):
            yield ImageColor.getrgb(f"hsl({i}, 100%, 50%)")
            yield ImageColor.getrgb(f"hsl({i+85}, 100%, 50%)")
            yield ImageColor.getrgb(f"hsl({i+171}, 100%, 50%)")
    elif type == "prism":
        for i in itertools.cycle(itertools.chain(np.linspace(0, 1, 100), np.linspace(1, 0, 100))):
            c = tuple([int(255*x) for x in cm.prism(i)])
            yield c

def draw_square_grid(grid_size: int, x_nodes: int, y_nodes: int, *args, **kwargs):
    tile_width = 1000
    tile_height = 1000
    im = Image.new("RGB", (grid_size * tile_width, grid_size * tile_height), "black")
    for i in range(grid_size):
        for j in range(grid_size):
            im.paste(draw_square(x_nodes, y_nodes, *args, **kwargs), (i * 1000, j * 1000))
    grid_width = 6000
    grid_height = 6000
    im = im.resize((grid_width, grid_height), resample=Image.Resampling.BICUBIC)
    horiz_color_points = itertools.product(np.linspace(0, im.width, (x_nodes+1) * grid_size, dtype=int), np.linspace(0, im.height, grid_size+1, dtype=int))
    vert_color_points = itertools.product(np.linspace(0, im.width, grid_size+1, dtype=int), np.linspace(0, im.height, (y_nodes+1) * grid_size, dtype=int))
    color_points = set(horiz_color_points) | set(vert_color_points)
    color_points = sorted(color_points, key=lambda x: math.sqrt(x[0]**2 + x[1]**2))
    color_iter = color_generator("prism")
    for x, y in tqdm(color_points, desc="filling colors"):
        x = min(x, im.width - 1)
        y = min(y, im.height - 1)
        if im.getpixel((x, y)) == (0, 0, 0):
            ImageDraw.floodfill(im, (x, y), next(color_iter))
    return im


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("x_nodes", type=int)
    parser.add_argument("y_nodes", type=int)
    parser.add_argument("--points", action="store_true", help="draw points on edges of tiles")
    parser.add_argument("--samples", type=int, default=100, help="number of samples per curve")
    parser.add_argument("--grid", action="store_true", help="create a grid of squares")
    parser.add_argument("--grid-size", type=int, default=5, help="size of grid")
    args = parser.parse_args()
    if args.grid:
        draw_square_grid(args.grid_size, args.x_nodes, args.y_nodes, draw_points=args.points, samples=args.samples).show()
    else:
        draw_square(args.x_nodes, args.y_nodes, draw_points=args.points, samples=args.samples).show()