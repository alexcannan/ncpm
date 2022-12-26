import itertools

import matplotlib.cm as cm
import numpy as np
from PIL import ImageColor


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
        for i in itertools.cycle(range(0, 256, 3)):
            yield ImageColor.getrgb(f"hsl({i}, 50%, 50%)")
            yield ImageColor.getrgb(f"hsl({i+85}, 50%, 50%)")
            yield ImageColor.getrgb(f"hsl({i+171}, 50%, 50%)")
    elif type == "prism":
        for i in itertools.cycle(itertools.chain(np.linspace(0, 1, 100), np.linspace(1, 0, 100))):
            c = tuple([int(128*x) for x in cm.prism(i)])
            yield c
    elif type == "random":
        for i in itertools.cycle(range(256)):
            yield tuple(np.random.randint(0, 256, 3))
