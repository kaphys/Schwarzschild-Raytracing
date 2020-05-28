from functions import *
from plot_figures import *

M = 1
N = 10000
N_r = 10
h = 0.002

r_4RK = np.zeros((2, N, N_r))
k_4RK = np.zeros((2, N, N_r))

# initial conditions for r and k
for i in range(N_r):
    r_4RK[:, 0, i] = np.array([-10, -10+1*i])
    k_4RK[:, 0, i] = np.array([1, 0])

# use runge kutta to trace the ray
for a in range(N_r):
    for n in range(0, N-1):

        Phi = Phi_g(M, r_4RK[:, n, a])

        r_4RK[:, n+1, a] = rk_4step_r(gr, h, r_4RK[:, n, a], k_4RK[:, n, a], Phi)
        k_4RK[:, n+1, a] = rk_4step_k(gk, h, r_4RK[:, n, a], k_4RK[:, n, a], Phi)


plot_rays_with_circle(r_4RK, M)
