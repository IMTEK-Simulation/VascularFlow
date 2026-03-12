import numpy as np
from scipy.sparse.linalg import spsolve

def newton(fun, x0, jac, tol=1e-6, maxiter=200, callback=None):
    """
    Newton solver expects vector-valued function fun and matrix-valued jac
    and initial vector x0

    Parameters
    ----------
    fun : callable(x)
        Function value of f at position x
    x0 : np.array
        Initial vector-valued value
    jac : callable(f, x)
        Jacobian of f at position x
    tol : float, optional
        Absolute tolerance for convergence
    maxiter : int, optional
        maximum number of Newton iterations
    callback : callable(x, **kwargs), optional
        callback function to handle logging and convergence
        stats

    Returns
    -------
    x : float
        approximate local root
    """
    x = x0
    if callback is not None:
        callback(0, x, None, None)
    for i in range(maxiter):
        f = fun(x)
        j = jac(x)
        x -= spsolve(j, f)
        if callback is not None:
            callback(i+1, x, f, j)
        if np.linalg.norm(f) < tol:
            return x
    raise RuntimeError('Newton solver did not converge.')