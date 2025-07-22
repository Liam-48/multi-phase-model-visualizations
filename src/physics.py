import numpy as np
from numba import njit

@njit
def laplacian_2D(u, dx=1.0):
    """
    Compute the discrete 2D Laplacian of a scalar field u using
    a standard 5-point finite difference stencil with periodic boundaries.

    Parameters:

    u : 2D ndarray
        Scalar field.
    dx : float, optional
        Grid spacing. Default is 1.0.
    """
    N = u.shape
    lap = np.empty_like(u)

    for i in range(N):
        for j in range(N):
            val = (
                u[(i+1)%N, j] + u[(i-1)%N, j] +
                u[i, (j+1)%N] + u[i, (j-1)%N] -
                4.0 * u[i, j]
            )
            lap[i, j] = val / dx**2

    return lap

def laplacian(u):
    return (
        np.roll(u, 1, axis=0) + np.roll(u, -1, axis=0) +
        np.roll(u, 1, axis=1) + np.roll(u, -1, axis=1) -
        4 * u
    )

@njit
def laplacian_3D(u, dx=1.0):
    """
    Compute the discrete 3D Laplacian of a scalar field u using
    a standard 7-point finite difference stencil with periodic boundaries.

    Parameters:

    u : 3D ndarray
        Scalar field.
    dx : float, optional
        Grid spacing. Default is 1.0.
    """

    N = u.shape[0]
    lap = np.empty_like(u)

    for i in range(N):
        for j in range(N):
            for k in range(N):
                lap[i,j,k] = (
                    u[(i+1)%N, j, k] + u[(i - 1)%N, j, k] +
                    u[i, (j+1)%N, k] + u[i, (j-1)%N, k] +
                    u[i, j, (k+1)%N] + u[i, j, (k-1)%N] -
                    6 * u[i, j, k] ) / dx**2
                
    return lap

def chem_potential(c, gamma, lap):
    """
    Chemical potential for the concentration field c.

    Parameters:

    c : ndarray
        Scalar field.
    gamma : float
        Gradient energy coefficient.
    lap : function
        Laplacian used (2D or 3D).
    """
    return 2.0 * c * (1.0 - c) * (1.0 - 2.0 * c) - gamma * lap(c)    

def source_term(t, tf, k):
    return 1/(1+np.exp(-k*(t-tf)))

def CH_step(c1, c2, gamma, lap, dt, source_term):
    """
    Performs a time step of a ternary Cahn-Hilliard model with 
    a source term that transfers material between phases, using finite 
    differences and periodic boundary conditions.

    Parameters:

    c1, c2 : ndarray
        Scalar field.
    gamma : float
        Gradient energy coefficient.
    lap : function
        Laplacian used (2D or 3D).
    dt : float
        Time step.
    steps : int
        Number of iterations.
    A : function
        Function that computes the conversion term A(t, tf, c1, c2, c3).
    """
    c3 = 1 - c1 - c2

    mu1 = chem_potential(c1, gamma, lap)
    mu2 = chem_potential(c2, gamma, lap)
    mu3 = chem_potential(c3, gamma, lap)

    c1_new = c1 + dt * (2 * lap(mu1) - lap(mu2) - lap(mu3))
    c2_new = c2 + dt * (2 * lap(mu2) - lap(mu1) - lap(mu3) + source_term)

    c3_new = 1 - c1_new - c2_new

    return c1_new, c2_new, c3_new