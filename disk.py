import numpy as np
import math as m
from PIL import Image
from functions import *
import matplotlib.pyplot as plt
import time
import scipy.misc as spm

start_time = time.time()

# Pixels
W = 500
H = 500

# amount of rays
N_r = W*H

# Schwarzschild radius
M = 0.5

# Camera settings
S_RATIO = 1 / 50
Z_SCREEN = np.array([0, 10, 8])
OBSERVER = np.array([0, 10, 10])
ScreenWidth = 15*M
ScreenHeigth = ScreenWidth

# Iterations
STEPS = 100
SIZE = 0.1

ones = np.ones((N_r))
# initialize rays
S = np.zeros((N_r, 6))

X, Y = np.meshgrid(np.arange(-W/2, W/2), np.arange(-H/2, H/2))
X = (X+1/2)/W*ScreenWidth + Z_SCREEN[0]
Y = (Y+1/2)/H*ScreenHeigth + Z_SCREEN[1]

R0 = np.zeros((N_r, 3))
R0[:, 0] = X.ravel()
R0[:, 1] = Y.ravel()
R0[:, 2] = Z_SCREEN[2]

p = R0 - np.array(OBSERVER)
k0 = p/np.linalg.norm(p, axis=1, keepdims=True)

S[:, 0:3] = R0
S[:, 3:6] = k0


# initializing the colour buffer
object_colour = np.zeros((N_r, 3))
object_alpha = np.zeros(N_r)

# use Runge-Kutta to trace the ray
for n in range(0, STEPS-1):
    oldpoint = S[:, 0:3]

    Phi = Phi_g(M, S[:, 0:3])
    S = RK4F(ray_equation, S, SIZE, Phi)

    point = S[:, 0:3]
    pointsqr = sqrnorm(point)

    # whether it just crossed the horizontal plane
    mask_crossing = np.logical_xor(oldpoint[:, 1] > 0., point[:, 1] > 0.)
    mask_distance = np.logical_and((pointsqr < 16), (pointsqr > 4))  # whether it's close enough
    diskmask = np.logical_and(mask_crossing, mask_distance)

    if (diskmask.any()):

        diskcolor = np.array([1., 1., .98])
        diskalpha = diskmask
        object_colour = blendcolors(diskcolor, diskalpha, object_colour, object_alpha)
        object_alpha = blendalpha(diskalpha, object_alpha)


def saveToImg(arr, fname):

    # copy
    imgout = np.array(arr)
    # clip
    imgout = np.clip(imgout, 0.0, 1.0)
    # rgb->srgb

    # unflattening
    imgout = imgout.reshape((W, H, 3))
    plt.imsave(fname, imgout)


saveToImg(object_colour, 'pics/disk_fromlargeangle.png')

elapsed_time = time.time() - start_time

print(f"time taken for simulation :{elapsed_time}")
