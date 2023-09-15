from numba import jit, njit
import numba
import numpy as np


@jit(nopython=True)
def check_is_inside(x, y, poly):
    n = len(poly)
    inside = False
    xints = 0.0
    p1x, p1y = poly[0]
    for i in numba.prange(n + 1):
        p2x, p2y = poly[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xints = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xints:
                        inside = not inside
        p1x, p1y = p2x, p2y

    return inside


@njit(parallel=True)
def point_in_polygon(points, polygon):
    is_inside = np.empty(len(points), dtype=numba.boolean)
    for i in numba.prange(0, len(is_inside)):
        is_inside[i] = check_is_inside(points[i, 0], points[i, 1], polygon)
    return is_inside
