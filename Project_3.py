import numpy as np
import matplotlib.pyplot as plt


def r_s(M):
    "schwardzchild radius"
    return 2*M


def Phi_g(M, r):
    rs = r_s(M)
    phi = -rs/r
    return phi


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def normalize(vector):
    return np.linalg.norm(vector)


def matrix_elements(r, k, Phi):
    psi = angle_between(r, k)

    rnorm = normalize(r)
    knorm = normalize(k)
    print(rnorm)

    y11 = 2*knorm*Phi/rnorm*np.cos(psi)
    y12 = 2*(1 + Phi)
    y21 = knorm**2 * Phi**2 / rnorm * (1 + 3*np.cos(psi)**2)
    y22 = -2*knorm*Phi/rnorm * np.cos(psi)
    print(y11)
    return (y11, y12, y21, y22)


def g1(r, k, Phi):
    (y11, y12, y21, y22) = matrix_elements(r, k, Phi)

    g1 = y11*r + y12*k

    return g1


def g2(r, k, Phi):
    (y11, y12, y21, y22) = matrix_elements(r, k, Phi)

    g2 = y21*r + y22*k

    return g2


M = 1
N = 15
h = 0.01

r_4RK = np.zeros((3, N))  # Runge-Kutta
r_4RK[:, 0] = np.array([-50, -1, 0])
k_4RK = np.zeros((3, N))  # Runge-Kutta
k_4RK[:, 0] = np.array([5, 0, 0])

# implementation of runge kutta doesnt work

for n in range(0, N-1):

    Phi = Phi_g(M, r_4RK)

    k1r = h*g1(r_4RK[:, n], k_4RK[:, n], Phi)
    k2r = h*g1(r_4RK[:, n] + k1r/2, k_4RK[:, n] + (h/2), Phi)
    k3r = h*g1(r_4RK[:, n] + k2r/2, k_4RK[:, n] + (h/2), Phi)
    k4r = h*g1(r_4RK[:, n] + k3r, k_4RK[:, n] + h, Phi)

    k1k = h*g2(r_4RK[:, n], k_4RK[:, n], Phi)
    k2k = h*g2(r_4RK[:, n] + k1k/2, k_4RK[:, n] + (h/2), Phi)
    k3k = h*g2(r_4RK[:, n] + k2k/2, k_4RK[:, n] + (h/2), Phi)
    k4k = h*g2(r_4RK[:, n] + k3k, k_4RK[:, n] + h, Phi)

    r_4RK[:, n+1] = r_4RK[:, n] + k1r/6 + k2r/3 + k3r/3 + k4r/6
    k_4RK[:, n+1] = k_4RK[:, n] + k1k/6 + k2k/3 + k3k/3 + k4k/6


plt.plot(r_4RK[:, 0], r_4RK[:, 1])
plt.show()
