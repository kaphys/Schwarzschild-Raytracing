from functions import *
from plot_figures import *
import time


def initialize_ray_inf(S):
    # initial conditions for r and k, with the rays coming from inf

    N_r = S.shape[1]
    for i in range(N_r):
        S[0, i, 0:3] = np.array([-50, -11 + 1*i, 0])
        S[0, i, 3:6] = np.array([1, 0, 0])

    return S


def main(M, N, N_r, h, plots):
    start_time = time.time()

    S = np.zeros((N, N_r, 6))
    S = initialize_ray_inf(S)

    # use runge kutta to trace the ray
    for n in range(0, N-1):

        Phi = Phi_g(M, S[n, :, 0:3])
        S[n+1, :, :] = RK4F(ray_equation, S[n, :, :], h, Phi)

    elapsed_time = time.time() - start_time
    print(f"time taken for simulation :{elapsed_time}")

    #plt.plot(np.arange(N), )
    if plots == True:
        plot_rays_with_circle(S[:, :, 0:3], M)


M = 1
N = 500
N_r = 40000
h = 0.01
plots = False


main(M, N, N_r, h, plots)
