import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


def plot_rays_with_circle(r, M):

    fig, ax = plt.subplots()
    ax.add_patch(plt.Circle((0, 0), 2*M, color='#000000', alpha=0.5))
    plt.plot(r[:, :, 0], r[:, :, 1], "r")
    ax.plot()

    plt.xlabel("X - coordinate in units of Schwardzchild radius")
    plt.ylabel("Y - coordinate in units of Schwardzchild radius")
    ax.set_aspect('equal', adjustable='datalim')
    plt.show()


def plot_rays_observer_star(r, M, observer, star):
    "observer and star should be coordinates"

    fig, ax = plt.subplots()
    ax.add_patch(plt.Circle(np.array([0, 0]), 2*M, color='#000000', alpha=0.5))

    # add observer and star(light source) point
    ax.add_patch(plt.Circle(observer, 0.1, color='#FF0000', alpha=0.5))
    plt.annotate("Observer", observer[0:2])
    ax.add_patch(plt.Circle(star, 0.1, color='#FFFF00', alpha=0.5))
    plt.annotate("Star", star[0:2])

    plt.plot(r[:, :, 0], r[:, :, 1])
    ax.plot()

    plt.xlabel("X - coordinate in units of Schwardzchild radius")
    plt.ylabel("Y - coordinate in units of Schwardzchild radius")
    ax.set_aspect('equal', adjustable='datalim')
    plt.savefig("photon_orbit.png")
    plt.show()


def plot_3d(r, z):
    "observer and star should be coordinates"

    fig = plt.figure()
    ax = plt.axes(projection='3d')

    ax.plot3D(r[:, :, 0], r[:, :, 1], z, 'gray')
    plt.show()
