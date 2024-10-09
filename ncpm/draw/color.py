from PIL import ImageColor


def equicolors(n: int, width: int=256, saturation: float=100, luminence: float=50, **kwargs) -> list[ImageColor]:
    huestep = width / n
    colors = [
        ImageColor.getrgb(f"hsl({i*huestep}, {saturation}%, {luminence}%)")
        for i in range(n)
    ]
    return colors
