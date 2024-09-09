import numpy as np


def compute_W(A_old, Q_old, A0, density, channel_elasticity):
    """
    compute the global characteristic variables W(U)=[W1(U), W2(U)]T.

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
    sqrt_term = 4 * np.sqrt(channel_elasticity/(2 * density * A0)) * (A_old ** 0.25)
    W_1 = Q_old / A_old + sqrt_term
    W_2 = Q_old / A_old - sqrt_term
    return W_1, W_2
