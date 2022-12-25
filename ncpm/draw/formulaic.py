from PIL import ImageDraw

from ncpm.draw.bezier import draw_bezier_curve
from ncpm import square_pair_type


def draw_formulaic_curve(draw: ImageDraw, x_nodes: int, y_nodes: int, x_node: int, y_node: int):
    """ for each node pair, draw a certain curve depending on the case (same edge, adjacent edge, opposite edge) """
    pair_type = square_pair_type(x_node, y_node, x_nodes, y_nodes)
    width = draw._image.width
    height = draw._image.height
    if pair_type == "same":
        # draw a semicircle
        x1 = (x_node + 1) * width / (x_nodes + 1)
        y1 = (y_node + 1) * height / (y_nodes + 1)
        x2 = (x_node + 2) * width / (x_nodes + 1)
        y2 = (y_node + 2) * height / (y_nodes + 1)
        draw.arc((x1, y1, x2, y2), 0, 180, fill="white", width=2)
    elif pair_type == "adjacent":
        # draw a quarter circle
        x1 = (x_node + 1) * width / (x_nodes + 1)
        y1 = (y_node + 1) * height / (y_nodes + 1)
        x2 = (x_node + 2) * width / (x_nodes + 1)
        y2 = (y_node + 2) * height / (y_nodes + 1)
        draw.arc((x1, y1, x2, y2), 0, 90, fill="white", width=2)
    elif pair_type == "opposite":
        # draw a bezier curve
        x_spacing = width / (x_nodes + 1)
        y_spacing = height / (y_nodes + 1)
        x1 = (x_node + 1) * x_spacing
        y1 = (y_node + 1) * y_spacing
        x2 = (x_node + 2) * x_spacing
        y2 = (y_node + 2) * y_spacing
        xys = [(x1, y1), (x1, y2), (x2, y2), (x2, y1)]
        draw_bezier_curve(draw, xys, samples=100)
    return
