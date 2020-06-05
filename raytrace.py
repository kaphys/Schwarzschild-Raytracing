import numpy as np
import math as m
from PIL import Image
import time

start = time.time()

# Schwarzschild radius
M = 0.5
ORIGIN = [0, 0, 0]
BG = 5
# Iterations
STEPS = 100
SIZE = 0.1

# Window dimensions
W = 300
H = 300
# Camera settings
S_RATIO = (4*M)/W
Z_START = 3
OBSERVER = [0, 0, 4]



# Blackhole color
BLACK = [0, 0, 0]

data = np.zeros((H, W, 3), dtype=np.uint8)
ray = np.zeros((H, W, 3))


def colorangles(ray):
    """
    angle at origin;
    polar angle of ray: phi
    azimuthal angle of ray: theta

    gives back color for background:
    depending on angle; theta and phi
    only after norm exceeds certain value.
    0 =< theta < 2pi
    0 =< phi =< pi
    """
    GREY = ([150, 150, 150])
    WHITE = ([255, 255, 255])

    norm = np.linalg.norm(ray)
    xn = ray[0]/norm
    yn = ray[1]/norm
    zn = ray[2]/norm
    phi = np.arccos(yn)
    theta = m.atan2(zn, xn)

    switch = (int(20*theta/np.pi) + int(20*phi/np.pi))% 2 == 0
    if switch == 0:
        bg_color = GREY
    else:
        bg_color = WHITE

    return bg_color


for i in range(H):
    for j in range(W):
        ray[i, j, :] = np.array([(i - 1 / 2 * H) * S_RATIO, (j - 1 / 2 * W) * S_RATIO, Z_START])
        p = ray[i, j, :] - OBSERVER
        direction = p/np.linalg.norm(p)
        for a in range(STEPS):
            if np.linalg.norm((ray[i, j, :] - ORIGIN)) <= 2*M:
                data[i, j, :] = BLACK
                break
            elif np.linalg.norm((ray[i, j, :] - ORIGIN)) >= BG:
                data[i, j, :] = colorangles(ray[i, j, :] - ORIGIN)
                break
            else:
                ray[i, j, :] += SIZE * direction

print('Runtime = ', time.time()-start, 'seconds.')

im = Image.fromarray(data)
im.show()
im = im.save("linear_render2.png")