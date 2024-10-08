import itertools
import sys

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
    elif type == "rainbow_distant":
        # randomly select hue value. chance to skip a hue value if it's too close to a previous one.
        # normal distribution with std dev of 64/len(hues) makes closer colors very unlikely at the beginning, becoming more lenient over time
        hues: list[float] = []
        while True:
            hue = np.random.rand() * 256
            probabilities = [np.exp(-((hue - h) ** 2) / (2 * (64 / len(hues)) ** 2)) for h in hues]
            # print(hue, hues, probabilities, file=sys.stderr)
            if not all(np.random.rand() > prob for prob in probabilities):
                # print(f"hue {hue} failed vibe check", file=sys.stderr)
                continue
            else:
                hues.append(hue)
                yield ImageColor.getrgb(f"hsl({hue}, 60%, 30%)")
    elif type == "prism":
        for i in itertools.cycle(itertools.chain(np.linspace(0, 1, 100), np.linspace(1, 0, 100))):
            c = tuple([int(128*x) for x in cm.prism(i)])
            yield c
    elif type == "random":
        for i in itertools.cycle(range(256)):
            yield tuple(np.random.randint(0, 256, 3))
