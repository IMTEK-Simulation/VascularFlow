"""
Definition of basis functions and their derivatives used in 1D finite element methods.

Intervals:
- Unit interval [0, 1]
- Interpolation in an arbitrary interval

Basis function types:
- LinearBasis
- QuadraticBasis
- HermiteBasis
"""

import numpy as np


class BasisFunction:

    _nb_nodes = None

    @property
    def nb_nodes(self):
        """
        Return the number of nodes in which the basis functions are being calculated (the degrees of freedom of each element).
        """
        return self._nb_nodes

    def eval(self, x: np.array):
        """Return the basis functions in unit interval."""
        raise NotImplementedError

    def first_derivative(self, x: np.array):
        """Return the first derivative of the Basis functions in unit interval"""
        raise NotImplementedError

    def second_derivative(self, x: np.array):
        """Return the second derivative of the Basis functions in unit interval"""
        raise NotImplementedError

    def interpolate(self, y_n: np.array, w_g: np.array, y_k: np.array) -> np.array:
        """
        Interpolate the basis functions in an arbitrary interval.

        Parameters
        ----------
        y_n : np.ndarray
            Nodal positions at each end of the interval.
        w_g : np.ndarray
            The basis function values at all degrees of freedom.
        y_k : np.ndarray
            The arbitrary interval.

        Returns
        -------
        np.ndarray
            The interpolated the basis function values at y_k.
        """

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
        return np.sum(w_nk * self.eval(x_k), axis=0)

    def interpolate_first_derivative(
        self, y_n: np.ndarray, w_g: np.ndarray, y_k: np.ndarray
    ) -> np.ndarray:
        """
        Interpolate the first derivative of a basis function in an arbitrary interval.

        Parameters
        ----------
        y_n : np.ndarray
            Nodal positions at each end of the interval.
        w_g : np.ndarray
            The basis function values at all degrees of freedom.
        y_k : np.ndarray
            The arbitrary interval.

        Returns
        -------
        np.ndarray
            The first derivative of the interpolated basis function.
        """
        nb_elements = len(y_n) - 1
        element_k = np.searchsorted(y_n, y_k) - 1
        element_k[element_k < 0] = 0
        element_k[element_k >= nb_elements] = nb_elements - 1
        assert len(element_k) == len(y_k)
        n = self.nb_nodes - 1
        w_nk = np.array([w_g[n * element_k + i] for i in range(self.nb_nodes)])
        x_k = (y_k - y_n[element_k]) / (y_n[element_k + 1] - y_n[element_k])
        h_k = y_n[element_k + 1] - y_n[element_k]  # element width
        return np.sum(w_nk * self.first_derivative(x_k) / h_k, axis=0)

    def interpolate_second_derivative(
        self, y_n: np.ndarray, w_g: np.ndarray, y_k: np.ndarray
    ) -> np.ndarray:
        """
        Interpolate the second derivative of a basis function in an arbitrary interval.

        Parameters
        ----------
        y_n : np.ndarray
            Nodal positions at each end of the interval.
        w_g : np.ndarray
            The basis function values at all degrees of freedom.
        y_k : np.ndarray
            The arbitrary interval.

        Returns
        -------
        np.ndarray
            The second derivative of the interpolated basis function.
        """
        nb_elements = len(y_n) - 1
        element_k = np.searchsorted(y_n, y_k) - 1
        element_k[element_k < 0] = 0
        element_k[element_k >= nb_elements] = nb_elements - 1
        assert len(element_k) == len(y_k)
        n = self.nb_nodes - 1
        w_nk = np.array([w_g[n * element_k + i] for i in range(self.nb_nodes)])
        x_k = (y_k - y_n[element_k]) / (y_n[element_k + 1] - y_n[element_k])
        h_k = y_n[element_k + 1] - y_n[element_k]
        return np.sum(w_nk * self.second_derivative(x_k) / (h_k**2), axis=0)


class LinearBasis(BasisFunction):
    """
    Linear basis functions for 1D finite element methods.

    This basis provides two shape functions per element:
    - Function value at node 0
    - Function value at node 1
    """

    _nb_nodes = 2

    def eval(self, x: np.array):
        """
        Evaluate the Linear basis functions at a given point x.

        Parameters
        ----------
        x: np.array
            The point at which to evaluate the basis functions in [0, 1].

        Returns
        ----------
        eval: np.ndarray
            A 1D array of shape (2,) containing the values of the basis functions at x.
        """
        return np.array([1 - x, x])

    def first_derivative(self, x: np.array):
        """
        Evaluate the first derivative of the Linear basis functions at a given point x.

        Parameters
        ----------
        x: np.array
            The point at which to evaluate the first derivatives of basis functions in [0, 1].

        Returns
        ----------
        first_derivative: np.ndarray
            A 1D array of shape (2,) containing the values of the first derivative of basis functions at x.
        """
        return np.array([-np.ones_like(x), np.ones_like(x)])


class QuadraticBasis(BasisFunction):
    """
    Quadrature basis functions for 1D finite element methods.

    This basis provides three shape functions per element:
    - Function value at node 0
    - Function value at node 1/2
    - Function value at node 1
    """

    _nb_nodes = 3

    def eval(self, x: np.array):
        """
        Evaluate the Quadrature basis functions at a given point x.

        Parameters
        ----------
        x: np.array
            The point at which to evaluate the basis functions in [0, 1].

        Returns
        ----------
        eval: np.ndarray
            A 1D array of shape (3,) containing the values of the basis functions at x.
        """
        return np.array([1 - 3 * x + 2 * x**2, 4 * x - 4 * x**2, -x + 2 * x**2])

    def first_derivative(self, x: np.array):
        """
        Evaluate the first derivative of the Quadrature basis functions at a given point x.

        Parameters
        ----------
        x: np.array
            The point at which to evaluate the first derivatives of basis functions in [0, 1].

        Returns
        ----------
        first_derivative: np.ndarray
            A 1D array of shape (3,) containing the values of the first derivative of basis functions at x.
        """
        return np.array([-3 + 4 * x, 4 - 8 * x, -1 + 4 * x])

    def second_derivative(self, x: np.array):
        """
        Evaluate the second derivative of the Quadrature basis functions at a given point x.

        Parameters
        ----------
        x: np.array
            The point at which to evaluate the first derivatives of basis functions in [0, 1].

        Returns
        ----------
        first_derivative: np.ndarray
            A 1D array of shape (3,) containing the values of the first derivative of basis functions at x.
        """

        return np.array(
            [4 * np.ones_like(x), -8 * np.ones_like(x), 4 * np.ones_like(x)]
        )


class HermiteBasis(BasisFunction):
    """
    Third-degree Hermite polynomial basis functions for 1D finite element methods.

    This basis provides four shape functions per element:
    - Function value at node 0
    - Derivative at node 0
    - Function value at node 1
    - Derivative at node 1
    """

    _nb_nodes = 4

    def eval(self, x: np.array):
        """
        Evaluate the cubic Hermite basis functions at a given point x.

        Parameters
        ----------
        x: np.array
            The point at which to evaluate the basis functions in [0, 1].

        Returns
        ----------
        eval: np.ndarray
            A 1D array of shape (4,) containing the values of the basis functions at x.
        """
        return np.array(
            [
                2 * x**3 - 3 * x**2 + 1,
                x**3 - 2 * x**2 + x,
                -2 * x**3 + 3 * x**2,
                x**3 - x**2,
            ]
        )

    def first_derivative(self, x: np.array):
        """
        Evaluate the first derivative of cubic Hermite basis functions at a given point x.

        Parameters
        ----------
        x: np.array
            The point at which to evaluate the first derivatives of basis functions in [0, 1].

        Returns
        ----------
        first_derivative: np.ndarray
            A 1D array of shape (4,) containing the values of the first derivative of basis functions at x.
        """
        return np.array(
            [
                6 * x**2 - 6 * x,
                3 * x**2 - 4 * x + 1,
                -6 * x**2 + 6 * x,
                3 * x**2 - 2 * x,
            ]
        )

    def second_derivative(self, x: np.array):
        """
        Evaluate the second derivative of cubic Hermite basis functions at a given point x.

        Parameters
        ----------
        x: np.array
            The point at which to evaluate the second derivatives of basis functions in [0, 1].

        Returns
        ----------
        second_derivative: np.ndarray
            A 1D array of shape (4,) containing the values of the second derivative of basis functions at x.
        """
        return np.array([12 * x - 6, 6 * x - 4, -12 * x + 6, 6 * x - 2])
