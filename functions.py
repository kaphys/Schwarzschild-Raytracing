import numpy as np


def r_s(M):
    "schwardzchild radius"
    # Units of c^2/G
    return 2*M


def Phi_g(M, r):
    "Gravitational potential"
    rs = r_s(M)
    phi = -rs/np.linalg.norm(r)
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


def matrix_elements(r, k, Phi):
    "Matrix elements from the paper"

    psi = angle_between(r, k)

    rnorm = np.linalg.norm(r)
    knorm = np.linalg.norm(k)

    y11 = 2*knorm*Phi/rnorm*(1 + Phi)*np.cos(psi)
    y12 = 2*(1 + Phi)
    y21 = knorm**2 * Phi / rnorm**2 * (1 + (3 + 4*Phi)*np.cos(psi)**2)
    y22 = -y11  # 2*knorm*Phi/rnorm * np.cos(psi)
    return (y11, y12, y21, y22)


def gr(r, k, Phi):
    "Right hand side of the diffential eqaution for vector r "
    (y11, y12, _, _) = matrix_elements(r, k, Phi)
    g1 = y11*r + y12*k

    return g1


def gk(r, k, Phi):
    "Right hand side of the diffential eqaution for vector k "

    (_, _, y21, y22) = matrix_elements(r, k, Phi)
    g2 = y21*r + y22*k

    return g2


def rk_4step_r(func, h, r, k, Phi):
    "Uses runge kutta method to calculate next step for r "

    s1 = h*func(r, k, Phi)
    s2 = h*func(r + s1/2 + h/2, k + s1/2, Phi)
    s3 = h*func(r + s2/2 + h/2, k + s2/2, Phi)
    s4 = h*func(r + s3 + h, k + s3, Phi)

    return r + s1/6 + s2/3 + s3/3 + s4/6


def rk_4step_k(func, h, r, k, Phi):
    "Uses runge kutta method to calculate next step for k "

    s1 = h*func(r, k, Phi)
    s2 = h*func(r + s1/2 + h/2, k + s1/2, Phi)
    s3 = h*func(r + s2/2 + h/2, k + s2/2, Phi)
    s4 = h*func(r + s3 + h, k + s3, Phi)

    return k + s1/6 + s2/3 + s3/3 + s4/6
