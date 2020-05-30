import numpy as np


def r_s(M):
    "schwardzchild radius"
    # Units of c^2/G
    return 2*M


def Phi_g(M, r):
    "Gravitational potential"
    rs = r_s(M)
    phi = -rs/np.linalg.norm(r, axis=1)  # , keepdims=True)
    phi[phi < -1] = -1
    return phi


def angle_rowwise(A, B):
    p1 = np.einsum('ij,ij->i', A, B)
    p2 = np.linalg.norm(A, axis=1)
    p3 = np.linalg.norm(B, axis=1)
    p4 = p1 / (p2*p3)
    return np.arccos(np.clip(p4, -1.0, 1.0))


def matrix_elements(r, k, Phi):
    "Matrix elements from the paper"
    psi = angle_rowwise(r, k)

    rnorm = np.linalg.norm(r, axis=1)
    knorm = np.linalg.norm(k, axis=1)

    y11 = 2*knorm*Phi/rnorm*(1 + Phi)*np.cos(psi)
    y12 = 2*(1 + Phi)
    y21 = knorm**2 * Phi / rnorm**2 * (1 + (3 + 4*Phi)*np.cos(psi)**2)
    y22 = -y11
    return (y11, y12, y21, y22)


def gr(r, k, Phi):
    "Right hand side of the diffential eqaution for vector r "
    (y11, y12, _, _) = matrix_elements(r, k, Phi)

    g1 = y11*r.T + y12*k.T
    g1 = g1.T
    return g1


def gk(r, k, Phi):
    "Right hand side of the diffential eqaution for vector k "

    (_, _, y21, y22) = matrix_elements(r, k, Phi)

    g2 = y21*r.T + y22*k.T
    g2 = g2.T
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
