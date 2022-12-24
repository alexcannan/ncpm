"""
given nodes on the edges of a square, generates a non-crossing perfect matching and draws with bezier curves
"""

import random

from PIL import Image, ImageDraw


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
    ts = [t/100.0 for t in range(101)]
    for match in matches:
        p0 = edge_coordinates[match[0]]
        p1 = edge_coordinates[match[1]]
        c0 = control_coordinates[match[0]]
        c1 = control_coordinates[match[1]]
        bezier = make_bezier([p0, c0, c1, p1])
        points = bezier(ts)
        for i in range(len(points) - 1):
            draw.line((points[i], points[i+1]), fill="white", width=2)
    im.show()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("x_nodes", type=int)
    parser.add_argument("y_nodes", type=int)
    args = parser.parse_args()
    draw_square(args.x_nodes, args.y_nodes)