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

    def interpolate(self, y_n: np.array, w_g: np.array, y_k: np.array) -> np.array:
        """Interpolate the function at x_k"""
        nb_elements = len(y_n) - 1
        element_k = np.searchsorted(y_n, y_k) - 1
        element_k[element_k < 0] = 0
        element_k[element_k >= nb_elements] = nb_elements - 1
        assert len(element_k) == len(y_k)
        n = self.nb_nodes - 1
        w_nk = []
        for i in range(self.nb_nodes):
            w_nk += [w_g[n * element_k + i]]
        w_nk = np.array(w_nk)
        x_k = (y_k - y_n[element_k]) / (y_n[element_k + 1] - y_n[element_k])
        return np.sum(w_nk*self.eval(x_k), axis=0)


class LinearBasis(BasisFunction):
    _nb_nodes = 2

    def eval(self, x: np.array) -> np.ndarray:
        return np.array([1 - x, x])

    def first_derivative(self, x: np.array) -> np.ndarray:
        return np.array([-np.ones_like(x), np.ones_like(x)])


class QuadraticBasis(BasisFunction):
    _nb_nodes = 3

    def eval(self, x: np.array) -> np.ndarray:
        return np.array([1 - 3 * x + 2 * x ** 2, 4 * x - 4 * x ** 2, -x + 2 * x ** 2])

    def first_derivative(self, x: np.array) -> np.ndarray:
        return np.array([-3 + 4 * x, 4 - 8 * x, -1 + 4 * x])

    def second_derivative(self, x: np.array) -> np.ndarray:
        return np.array(
            [4 * np.ones_like(x), -8 * np.ones_like(x), 4 * np.ones_like(x)]
        )
