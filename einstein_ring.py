import numpy as np
import math as m
from PIL import Image
from functions import *
import matplotlib.pyplot as plt
import time


def blendcolors(cb, balpha, ca, aalpha):
    # * np.outer(aalpha, np.array([1.,1.,1.])) + \
    # return  ca + cb * np.outer(balpha*(1.-aalpha),np.array([1.,1.,1.]))
    return ca + cb * (balpha*(1.-aalpha))[:, np.newaxis]


# this is for the final alpha channel after blending
def blendalpha(balpha, aalpha):
    return aalpha + balpha*(1.-aalpha)


start_time = time.time()


GREY = ([150, 150, 150])
WHITE = ([255, 255, 255])

# Window dimensions
W = 500
H = 500

# amount of rays
N_r = W*H

# Schwarzschild radius
M = 0.5

# Camera settings
S_RATIO = 1 / 50
Z_SCREEN = 6
OBSERVER = np.array([0, 0, 8])
ScreenWidth = 5*M
ScreenHeigth = ScreenWidth


# Iterations
STEPS = 600
SIZE = 0.03

# initialize rays
S = np.zeros((N_r, 6))

X, Y = np.meshgrid(np.arange(-W/2, W/2), np.arange(-H/2, H/2))
X = (X+1/2)/W*ScreenWidth
Y = (Y+1/2)/H*ScreenHeigth

R0 = np.zeros((N_r, 3))
R0[:, 0] = X.ravel()
R0[:, 1] = Y.ravel()
R0[:, 2] = Z_SCREEN

p = R0 - np.array(OBSERVER)
k0 = p/np.linalg.norm(p, axis=1, keepdims=True)

S[:, 0:3] = R0
S[:, 3:6] = k0


light_source = np.array([-0.5, 0, -5])


# initializing the colour buffer
object_colour = np.zeros((N_r, 3))
object_alpha = np.zeros(N_r)

# use Runge-Kutta to trace the ray
for n in range(0, STEPS-1):

    Phi = Phi_g(M, S[:, 0:3])
    S = RK4F(ray_equation, S, SIZE, Phi)

    point = S[:, 0:3]
    # check collision star
    norm_dist_star = sqrnorm(point - light_source)

    star_mask = np.where(norm_dist_star <= M + 0.05, True, False)

    if star_mask.any():
        star_colour = np.array([1, 1, 0.1])
        star_alpha = star_mask
        object_colour = blendcolors(star_colour, star_alpha, object_colour, object_alpha)
        object_alpha = blendalpha(star_alpha, object_alpha)


def saveToImg(arr, fname):

    # copy
    imgout = np.array(arr)
    # clip
    imgout = np.clip(imgout, 0.0, 1.0)
    # rgb->srgb

    # unflattening
    imgout = imgout.reshape((W, H, 3))
    plt.imsave(fname, imgout)


saveToImg(object_colour, 'pics/einsteinring_littleOff_axisbig1.png')


elapsed_time = time.time() - start_time

print(f"time taken for simulation :{elapsed_time}")
