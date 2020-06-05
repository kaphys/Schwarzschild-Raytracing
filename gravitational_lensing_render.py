import numpy as np
import math as m
from PIL import Image
from functions import *
import matplotlib.pyplot as plt
import time

start_time = time.time()

GREY = ([150, 150, 150])
WHITE = ([255, 255, 255])


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
    norm = np.linalg.norm(ray)
    xn = ray[0]/norm
    yn = ray[1]/norm
    zn = ray[2]/norm
    phi = np.arccos(yn)
    theta = m.atan2(zn, xn)

    switch = (int(20*theta/np.pi) + int(20*phi/np.pi)) % 2 == 0
    if switch == 0:
        bg_color = GREY
    else:
        bg_color = WHITE

    return bg_color


# Window dimensions
W = 300
H = 300

# amount of rays
N_r = W*H

# Camera settings
S_RATIO = 1 / 50
Z_SCREEN = 3
OBSERVER = np.array([0, 0, 4])

# Schwarzschild radius
M = 0.5
ORIGIN = [0, 0, 0]
BG = 5
ScreenWidth = 4*M
ScreenHeigth = ScreenWidth

# Iterations
STEPS = 2000
SIZE = 0.01

# Blackhole color
BLACK = [0, 0, 0]


X, Y = np.meshgrid(np.arange(-W/2, W/2), np.arange(-H/2, H/2))
X = (X+1/2)/W*ScreenWidth
Y = (Y+1/2)/H*ScreenHeigth

R0 = np.zeros((N_r, 3))
R0[:, 0] = X.ravel()
R0[:, 1] = Y.ravel()
R0[:, 2] = Z_SCREEN

p = R0 - np.array(OBSERVER)
k0 = p/np.linalg.norm(p, axis=1, keepdims=True)


S = np.zeros((N_r, 6))


S[:, 0:3] = R0
S[:, 3:6] = k0


# use Runge-Kutta to trace the ray
for n in range(0, STEPS-1):

    Phi = Phi_g(M, S[:, 0:3])
    S = RK4F(ray_equation, S, SIZE, Phi)


norm_dist = np.linalg.norm(S[:, 0:3], axis=1)

# check collision event horizon
ray_index = np.where(np.abs(norm_dist) >= 2*M + 0.1,
                     True, False)

pixel_mask = np.reshape(ray_index, (-1, W))

b = np.zeros((W, H, 3), dtype=np.uint8)
color = np.array([255, 255, 255])


# Make True pixels white
b[pixel_mask] = [255, 255, 255]
# Make False pixels BLACK
b[~pixel_mask] = [0, 0, 0]

index = 0
for i in range(W):
    for j in range(H):
        if pixel_mask[i, j] == True:
            b[i, j, :] = colorangles(S[index, 0:3])
        index += 1


im = Image.fromarray(b)
im.show()
im.save("graviational_lensing_render.png")
elapsed_time = time.time() - start_time

print(f"time taken for simulation :{elapsed_time}")


plt.show()
