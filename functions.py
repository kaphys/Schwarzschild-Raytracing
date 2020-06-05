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


def ray_equation(S, Phi):
    r = S[:, 0:3]
    k = S[:, 3:6]
    dxdy = np.zeros(S.shape)

    psi = angle_rowwise(r, k)

    rnorm = np.linalg.norm(r, axis=1)
    knorm = np.linalg.norm(k, axis=1)

    y11 = 2*knorm*Phi/rnorm*(1 + Phi)*np.cos(psi)
    y12 = 2*(1 + Phi)
    y21 = knorm**2 * Phi / rnorm**2 * (1 + (3 + 4*Phi)*np.cos(psi)**2)
    y22 = -y11

    g1 = y11*r.T + y12*k.T
    g2 = y21*r.T + y22*k.T
    dxdy[:, 0:3] = g1.T
    dxdy[:, 3:6] = g2.T

    return dxdy


def RK4F(func, S, h, Phi):

    s1 = h*func(S, Phi)
    s2 = h*func(S + s1/2.0 + h/2, Phi)
    s3 = h*func(S + s2/2.0 + h/2, Phi)
    s4 = h*func(S + s3 + h, Phi)

    return S + s1/6 + s2/3 + s3/3 + s4/6
