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

    switch = (int(20*theta/np.pi) + int(20*phi/np.pi))% 2 == 0
    if switch == 0:
        bg_color = GREY
    else:
        bg_color = WHITE

    return bg_color

# Window dimensions
W = 50
H = 50

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
STEPS = 100
SIZE = 0.1

# Blackhole color
BLACK = [0, 0, 0]
N_r = W*H


X, Y = np.meshgrid(np.arange(-W/2, W/2), np.arange(-H/2, H/2))
X = (X+1/2)/W*ScreenWidth
Y = (Y+1/2)/H*ScreenHeigth

R0 = np.zeros((N_r, 3))
R0[:, 0] = X.ravel()
R0[:, 1] = Y.ravel()
R0[:, 2] = Z_SCREEN

p = R0 - np.array(OBSERVER)
k0 = p/np.linalg.norm(p, axis=1, keepdims=True)


N = 2000
h = 0.01
r = np.zeros((N_r, 3))
k = np.zeros((N_r, 3))

r[:, :] = R0
k[:, :] = k0

norm_dist = np.linalg.norm(r[:, :], axis=1)
ray_index = np.zeros(N_r)

Tr = np.ones(N_r, dtype=bool)
Fa = np.zeros(N_r, dtype=bool)
booal = norm_dist - 2*M > 0.1

ray_index = np.where(booal, Tr, Fa)


light_source = np.array([0, 0, -5])

# use Runge-Kutta to trace the ray
for n in range(0, N-1):

    save_r = r[:, :]
    save_k = k[:, :]
    Phi = Phi_g(M, r[:, :])
    r[:, :] = rk_4step_r(gr, h, save_r, save_k, Phi)
    k[:, :] = rk_4step_k(gk, h, save_r, save_k, Phi)


norm_dist = np.linalg.norm(r[:, :], axis=1)

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
            b[i, j, :] = colorangles(r[index, :])
        index += 1

im = Image.fromarray(b)
im.show()
im.save("graviational_lensing_render.png")
elapsed_time = time.time() - start_time

print(f"time taken for simulation :{elapsed_time}")


plt.show()
