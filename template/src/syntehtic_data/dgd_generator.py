"""
Algorithms for simulating matrices that exhibit the same characteristics as Norwegian cervical cancer screening data.
"""

import numpy as np

def _dgd_distribution(x, theta, dom):
    return np.exp(-theta*(x - dom)**2)


def simulate_float_from_named_basis(
    basis_name,
    N=38000,
    T=321,
    K=5,
    domain=[1, 4],
    random_state=42
):
    """Examples of continuous real-valued profiles.

    Parameters
    ----------
    basis_name : str
        Nickname of basis to be used.

    N : int, default=38000
        n_samples to generate.

    T : int, default=321
        n_timesteps.

    K : int, default=5
        number of basis profiles to use. Corresponds to the approximate rank
        of D

    domain : list of float, default=[1, 4]
        Upper and lower bound on the resulting real-valued profiles.

    random_state : int, default=None
        For reproducibility.

    Returns
    -------
    M : array of shape (n_samples, n_timesteps)
        The resulting real-valued matrix.
    """

    if not(random_state is None):
        np.random.seed(random_state)

    if basis_name == 'simple_peaks':
        V = np.empty(shape=(T, K))

        centers = np.linspace(70, 170, K)

        x = np.linspace(0, T, T)
        for i_k in range(K):
            V[:, i_k] = 1 + 3.0 * np.exp(-5e-4*(x - centers[i_k])**2)

        shape=1.0
        scale=1.0

    elif basis_name == 'hard_peaks':
        V = np.empty(shape=(T, K))

        centers = np.linspace(70, 170, K)

        x = np.linspace(0, T, T)
        for i_k in range(K):
            V[i_k] = 1 + 3.0 * np.exp(-5e-4*(x - centers[i_k]**2))

        shape=1.0
        scale=10.0

    U = np.random.gamma(shape, scale, size=(N, K))

    M_unscaled = U@V.T

    M = domain[0] + (M_unscaled - np.min(M_unscaled))/(np.max(M_unscaled) - np.min(M_unscaled))*(domain[1] - domain[0])

    return M


def simulate_dgd(
    M,
    domain_z,
    theta,
    random_state=None
):
    """Simulation of a missing data mask.

    Parameters
    ----------
    M : array of shape (n_samples, n_timesteps)
        The real-valued matrix.

    domain_z : array of shape (n_class_z, )
        Allowed integers in output.

    random_state : int, default=None
        For reproducibility.

    Returns
    -------
    D : array of shape (n_samples, n_timesteps)
        The resulting integer-valued matrix.
    """
    if not(random_state is None):
        np.random.seed(random_state)

    domain_max = np.max(domain_z)
    domain_min = np.min(domain_z)

    N = M.shape[0]
    T = M.shape[1]
    Z = domain_z.shape[0]

    X_float_scaled = domain_min + (domain_max - domain_min)*(M -
                                                             np.min(M))/(np.max(M) - np.min(M))

    domain_repeated = np.repeat(domain_z, N).reshape((N, Z), order='F')

    D = np.empty_like(X_float_scaled)

    # Initialization
    column_repeated = np.repeat(
        X_float_scaled[:, 0], 4).reshape((N, 4), order='C')
    pdf = _dgd_distribution(column_repeated, theta, domain_repeated)
    cdf = np.cumsum(pdf / np.reshape(np.sum(pdf, axis=1), (N, 1)), axis=1)

    u = np.random.uniform(size=(N, 1))
    indices = np.argmax(u <= cdf, axis=1)
    D[:, 0] = domain_z[indices]

    # Timestepping
    for j in range(1, T):
        column_repeated = np.repeat(
            X_float_scaled[:, j], 4).reshape((N, 4), order='C')
        pdf = _dgd_distribution(column_repeated, theta, domain_repeated)
        cdf = np.cumsum(pdf / np.reshape(np.sum(pdf, axis=1), (N, 1)), axis=1)

        u = np.random.uniform(size=(N, 1))
        indices = np.argmax(u <= cdf, axis=1)
        D[:, j] = domain_z[indices]

    return D