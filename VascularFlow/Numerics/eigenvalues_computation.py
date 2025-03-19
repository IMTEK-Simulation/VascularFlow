import numpy as np


def compute_lambda(A_old, Q_old, A0, density, channel_elasticity):
    """
    compute the eigenvalues of H matrix.
    ∂U/∂t + H ∂U/∂x + B = 0

    Parameters:
    - A_old: np.Array, previous time step values of A(x,t) (cross-sectional area)
    - Q_old: np.Array, previous time step values of Q(x,t) (flow rate)
    - A0: float, channel initial cross-sectional area
    - density: float, fluid density
    - channel_elasticity: float, channel elasticity (in the second p-A model)

    Returns:
    - lambda_1: np.Array, previous time step values of wave speed to the right hand side of the computational domain
    - lambda_2: np.Array, previous time step values of wave speed to the left hand side of the computational domain
    """
    sqrt_term = np.sqrt(channel_elasticity/(2 * density * A0)) * (A_old ** 0.25)
    lambda_1 = Q_old / A_old + sqrt_term
    lambda_2 = Q_old / A_old - sqrt_term
    return lambda_1, lambda_2
