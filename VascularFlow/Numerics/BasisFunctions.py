import numpy as np


class BasisFunction:
    """Basis function on the unit interval [0, 1]"""

    _nb_nodes = None

    @property
    def nb_nodes(self):
        return self._nb_nodes

    def eval(self, x: np.ndarray) -> np.ndarray:
        """Return the Basis functions evaluated at x"""
        raise NotImplementedError

    def first_derivative(self, x: np.array) -> np.ndarray:
        """Return the first derivative of the Basis functions evaluated at x"""
        raise NotImplementedError

    def second_derivative(self, x: np.array) -> np.ndarray:
        """Return the second derivative of the Basis functions evaluated at x"""
        raise NotImplementedError


class LinearBasis(BasisFunction):
    _nb_nodes = 2

    def eval(self, x: np.array) -> np.ndarray:
        return np.array([1 - x, x])

    def first_derivative(self, x: np.array) -> np.ndarray:
        return np.array([-np.ones_like(x), np.ones_like(x)])


class QuadraticBasis(BasisFunction):
    _nb_nodes = 3

    def eval(self, x: np.array) -> np.ndarray:
        return np.array([1 - 3 * x + 2 * x**2, 4 * x - 4 * x**2, -x + 2 * x**2])

    def first_derivative(self, x: np.array) -> np.ndarray:
        return np.array([-3 + 4 * x, 4 - 8 * x, -1 + 4 * x])

    def second_derivative(self, x: np.array) -> np.ndarray:
        return np.array(
            [4 * np.ones_like(x), -8 * np.ones_like(x), 4 * np.ones_like(x)]
        )


class HermiteBasis(BasisFunction):
    """Third-degree polynomial basis function with two degrees of freedom at each node of the element"""

    _nb_nodes = 4

    def eval(self, x: np.array) -> np.ndarray:
        return np.array(
            [
                2 * x**3 - 3 * x**2 + 1,
                x**3 - 2 * x**2 + x,
                -2 * x**3 + 3 * x**2,
                x**3 - x**2,
            ]
        )

    def first_derivative(self, x: np.array) -> np.ndarray:
        return np.array(
            [
                6 * x**2 - 6 * x,
                3 * x**2 - 4 * x + 1,
                -6 * x**2 + 6 * x,
                3 * x**2 - 2 * x,
            ]
        )

    def second_derivative(self, x: np.array) -> np.ndarray:
        return np.array([12 * x - 6, 6 * x - 4, -12 * x + 6, 6 * x - 2])
