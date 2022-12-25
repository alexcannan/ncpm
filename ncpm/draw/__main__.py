import argparse

from ncpm.draw.square import draw_square
from ncpm.draw.grid import draw_square_grid


parser = argparse.ArgumentParser()
parser.add_argument("x_nodes", type=int)
parser.add_argument("y_nodes", type=int)
parser.add_argument("--points", action="store_true", help="draw points on edges of tiles")
parser.add_argument("--samples", type=int, default=100, help="number of samples per curve")
parser.add_argument("--color", type=str, help="color type (discrete, grayscale, rainbow, rainbow2, rainbow3, prism)")
parser.add_argument("--curve", type=str, default="bezier", help="curve type (bezier, bezier_centered, line)")
parser.add_argument("--grid", action="store_true", help="create a grid of squares")
parser.add_argument("--grid-size", type=int, default=5, help="size of grid")
args = parser.parse_args()
if args.grid:
    draw_square_grid(args.grid_size, args.x_nodes, args.y_nodes, draw_points=args.points, samples=args.samples, color_type=args.color, curve_type=args.curve).show()
else:
    draw_square(args.x_nodes, args.y_nodes, draw_points=args.points, samples=args.samples, curve_type=args.curve).show()