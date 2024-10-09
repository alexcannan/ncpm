from PIL import ImageColor


def equicolors(n: int, width: int=256, saturation: float=100, luminence: float=50, **kwargs) -> list[ImageColor]:
    return [
        ImageColor.getrgb(f"hsl({i}, {saturation}%, {luminence}%)")
        for i in range(0, width, width // n)
    ]
