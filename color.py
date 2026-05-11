import matplotlib.colors as mcolors

base_colors = tuple(
    mcolors.to_rgb(tuple(c/255 for c in rgb))
    for rgb in [
        (86, 91, 96),
        (154, 215, 246),
        (250, 192, 154),
        (192, 222, 186),
        (247, 190, 209),
        (255, 243, 109)
    ]
)

def darken_color(color, amount=0.6):
    c = mcolors.to_rgb(color)
    return tuple(amount * x for x in c)

import matplotlib.colors as mcolors
import colorsys

def saturate_color(color, amount=0.6):
    r, g, b = mcolors.to_rgb(color)
    h, l, s = colorsys.rgb_to_hls(r, g, b)

    # scale saturation (clamped to [0, 1])
    s = max(0.0, min(1.0, s * amount))

    r, g, b = colorsys.hls_to_rgb(h, l, s)
    return (r, g, b)

import matplotlib.colors as mcolors
import colorsys

def vivid_color(color, amount=1.3):
    r, g, b = mcolors.to_rgb(color)
    h, l, s = colorsys.rgb_to_hls(r, g, b)

    # boost saturation to reduce "pale" look
    s = max(0.0, min(1.0, s * amount))

    r, g, b = colorsys.hls_to_rgb(h, l, s)
    return (r, g, b)