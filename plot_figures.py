import matplotlib.pyplot as plt


def plot_rays_with_circle(r, M):

    fig, ax = plt.subplots()
    ax.add_patch(plt.Circle((0, 0), 2*M, color='#000000', alpha=0.5))
    plt.plot(r[0, :, :], r[1, :, :])
    ax.plot()

    plt.xlabel("X - coordinate in units of Schwardzchild radius")
    plt.ylabel("Y - coordinate in units of Schwardzchild radius")
    ax.set_aspect('equal', adjustable='datalim')
    plt.show()
