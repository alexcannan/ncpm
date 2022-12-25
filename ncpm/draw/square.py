from PIL import Image, ImageDraw

from ncpm import generate_matching
from ncpm.draw import line_width
from ncpm.draw.bezier import draw_bezier_curve
from ncpm.draw.formulaic import draw_formulaic_curve


def draw_square(x_nodes: int=3, y_nodes: int=3, draw_points: bool=False, samples: int=100, curve_type: str="bezier"):
    width = 1000
    height = 1000
    im = Image.new("RGB", (width, height), "black")
    draw = ImageDraw.Draw(im)
    n_nodes = 2 * x_nodes + 2 * y_nodes
    x_spacing = width / (x_nodes + 1)
    y_spacing = height / (y_nodes + 1)
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
        if curve_type == "bezier":
            c0 = control_coordinates[match[0]]
            c1 = control_coordinates[match[1]]
            draw_bezier_curve(draw, [p0, c0, c1, p1], samples)
        elif curve_type == "bezier_centered":
            c0 = control_coordinates[match[0]]
            c1 = control_coordinates[match[1]]
            draw_bezier_curve(draw, [p0, c0, (im.width / 2, im.height / 2), c1, p1], samples)
        elif curve_type == "line":
            draw.line((p0, p1), fill="white", width=line_width)
        elif curve_type == "formulaic":
            draw_formulaic_curve(draw, x_nodes, y_nodes, match[0], match[1])
    return im
