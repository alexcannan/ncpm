from PIL import ImageDraw

from ncpm.draw import line_width


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


def draw_bezier_curve(draw: ImageDraw, xys, samples=100):
    # XXX: these curves seem to have whitespace inside the lines, it's breaking connected components
    ts = [t/samples for t in range(samples+1)]
    bezier = make_bezier(xys)
    points = bezier(ts)
    for i in range(len(points) - 1):
        draw.line((points[i], points[i+1]), fill="white", width=line_width)
