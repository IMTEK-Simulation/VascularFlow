"""
1)
    Definition of basis functions and their derivatives used in 1D finite element methods.

    Intervals:
        - Unit interval [0, 1]
        - Interpolation in an arbitrary interval

    Basis function types:
        - LinearBasis
        - QuadraticBasis
        - HermiteBasis

2)
    Definition of shape functions and their derivatives used in 2D finite element methods.

    Element type:
        -Quadrilateral elements

    Intervals:
        -Square reference element Ê = [−1, 1]×[−1, 1] centered at the origin of the Cartesian (ξ, η) coordinate system

    Shape functions types:
        - Bi-linear shape functions
        - Adini-Clough-Melosh (ACM)
            * The shape functions are defined by polynomials of degree three
            * The ACM element has 12 degrees of freedom
            * The ACM element is widely used in the analysis of plates
"""

import numpy as np


class BasisFunction:

    _nb_nodes = None
    _dof_per_node = 1

    @property
    def nb_nodes(self):
        """
        Return the number of nodes in which the basis functions are being calculated (the degrees of freedom of each element).
        """
        return self._nb_nodes

    @property
    def dof_per_node(self):
        """
        Return the number of degrees of freedom of each element (used in the assembly matrix).
        """
        return self._dof_per_node

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


class ShapeFunction:
    """
    Abstract base class for shape functions on a reference element.
    Subclasses must implement methods for evaluating the shape function
    and its derivatives.
    """

    _nb_nodes = None
    _dof_per_node = None

    @property
    def nb_nodes(self):
        """
        Returns:
            int: Number of nodes in the reference element.
        """
        return self._nb_nodes

    @property
    def dof_per_node(self):
        """
        Returns:
            int: Number of degrees of freedom per node.
        """
        return self._dof_per_node

    def eval(self, s: np.array, n: np.array):
        """
        Evaluate the shape function values on the reference element.

        Args:
            s (np.ndarray): Local horizontal (ξ) coordinates.
            n (np.ndarray): Local vertical (η) coordinates.

        Returns:
            np.ndarray: Shape function values at (s, n).
        """

        raise NotImplementedError

    def first_derivative(self, s: np.array, n: np.array):
        """
        Evaluate the first derivatives of the shape functions.

        Args:
            s (np.ndarray): Local horizontal (ξ) coordinates.
            n (np.ndarray): Local vertical (η) coordinates.

        Returns:
            np.ndarray: First derivatives ∇φ_k with shape (4, 2), where
                        column 0 is ∂φ/∂s and column 1 is ∂φ/∂n.
        """

        raise NotImplementedError

    def second_derivative(self, s: np.array, n: np.array):
        """
        Evaluate the second derivatives of the shape functions.

        Args:
            s (np.ndarray): Local horizontal (ξ) coordinates.
            n (np.ndarray): Local vertical (η) coordinates.

        Returns:
            np.ndarray: Second derivatives (not implemented here).
        """
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
        return np.sum(w_nk * self.eval(x_k), axis=0)


class LinearBasis(BasisFunction):
    """
    Linear basis functions for 1D finite element methods.

    This basis provides two shape functions per element:
    - Function value at node 0
    - Function value at node 1
    """

    _nb_nodes = 2
    _dof_per_node = 1

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
    _dof_per_node = 1

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
    _dof_per_node = 2

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


class BilinearShapeFunctions(ShapeFunction):
    """
    Bilinear shape functions for 4-node quadrilateral (Q1) elements.
    Defined on the reference square [-1, 1] x [-1, 1].

    Node order:
        - Node 1: (-1, -1)
        - Node 2: (+1, -1)
        - Node 3: (+1, +1)
        - Node 4: (-1, +1)
    """

    _nb_nodes = 4
    _dof_per_node = 1

    def eval(self, s: np.array, n: np.array):
        """
        Evaluate the bilinear shape functions φ₁ to φ₄.

        Args:
            s (np.ndarray): Local horizontal (ξ) coordinates.
            n (np.ndarray): Local vertical (η) coordinates.

        Returns:
            np.ndarray: Array of shape function values [φ₁, φ₂, φ₃, φ₄].
        """

        return np.array(
            [
                0.25 * (1 - s) * (1 - n),  # φ1
                0.25 * (1 + s) * (1 - n),  # φ2
                0.25 * (1 + s) * (1 + n),  # φ3
                0.25 * (1 - s) * (1 + n),  # φ4
            ]
        )

    def first_derivative(self, s: np.array, n: np.array):
        """
        Evaluate the first derivatives ∇φ_k of the bilinear shape functions.

        Args:
            s (np.ndarray): Local horizontal (ξ) coordinates.
            n (np.ndarray): Local vertical (η) coordinates.

        Returns:
            np.ndarray: Array of shape (4, 2) where:
                        - Row k contains the gradient of φ_k
                        - Column 0 = ∂φ/∂ξ (s), Column 1 = ∂φ/∂η (n)
        """

        return (
            np.array(
                [
                    [-(1 - n), -(1 - s)],  # ∇φ1
                    [(1 - n), -(1 + s)],  # ∇φ2
                    [(1 + n), (1 + s)],  # ∇φ3
                    [-(1 + n), (1 - s)],  # ∇φ4
                ]
            )
            * 0.25
        )


class ACMShapeFunctions(ShapeFunction):
    """
    Adini-Clough-Melosh (ACM) shape functions for 12-node quadrilateral elements.
    Defined on the reference square [-1, 1] x [-1, 1].

    Node order:
        - Node 1: (-1, -1)
        - Node 2: (+1, -1)
        - Node 3: (+1, +1)
        - Node 4: (-1, +1)
    """

    _nb_nodes = 12
    _dof_per_node = 1

    def eval(self, s: np.array, n: np.array):
        """
        Evaluate the ACM shape functions φ₁ to φ₁₂.

        Args:
            s (np.ndarray): Local horizontal (ξ) coordinates.
            n (np.ndarray): Local vertical (η) coordinates.

        Returns:
            np.ndarray: Array of shape function values [φ₁, φ₂, ..., φ₁₂].
        """
        return np.array(
            [
                # N1 =
                -0.125 * (-1 + n) * (-1 + s) * (-2 + n + n**2 + s + s**2),
                # N2 =
                -0.0625 * (-1 + n) ** 2 * (1 + n) * (-1 + s),
                # N3 =
                -0.0625 * (-1 + n) * (-1 + s) ** 2 * (1 + s),
                # N4 =
                0.125 * (-1 + n) * (1 + s) * (-2 + n + n**2 - 1 * s + s**2),
                # N5 =
                -(n + 1) * (1.0 * n - 0.25 * (n + 1) ** 2) * (s + 1) / 4,
                # N6 =
                (s + 1) ** 2 * (0.125 * n + 0.125 * s - 0.0625 * (n + 1) * (s + 1)),
                # N7 =
                (n + 1)
                * (s + 1)
                * (1.5 * n + 1.5 * s - 0.5 * (n + 1) ** 2 - 0.5 * (s + 1) ** 2 + 2.0)
                / 4,
                # N8 =
                0.0625 * (n - 1) * (n + 1) ** 2 * (s + 1),
                # N9 =
                0.0625 * (n + 1) * (s - 1) * (s + 1) ** 2,
                # N10 =
                0.125 * (1 + n) * (-1 + s) * (-2 - 1 * n + n**2 + s + s**2),
                # N11 =
                (n + 1) ** 2 * (0.125 * n + 0.125 * s - 0.0625 * (n + 1) * (s + 1)),
                # N12 =
                -(n + 1) * (s + 1) * (1.0 * s - 0.25 * (s + 1) ** 2) / 4,
            ]
        )

    def first_derivative(self, s: np.array, n: np.array):
        """
        Evaluate the first derivatives ∇φ_k of the ACM shape functions.

        Args:
            s (np.ndarray): Local horizontal (ξ) coordinates.
            n (np.ndarray): Local vertical (η) coordinates.

        Returns:
            np.ndarray: Array of shape (12, 2) where:
                        - Row k contains the gradient of φ_k
                        - Column 0 = ∂φ/∂ξ (s), Column 1 = ∂φ/∂η (n)
        """

        return np.array(
            [
                # ∇N1 =
                [
                    -(n - 1) * (n**2 + n + 3 * s**2 - 3) / 8,
                    -(s - 1) * (3 * n**2 + s**2 + s - 3) / 8,
                ],
                # ∇N2 =
                [
                    -((n - 1) ** 2) * (n + 1) / 16,
                    -(n - 1) * (3 * n + 1) * (s - 1) / 16,
                ],
                # ∇N3 =
                [
                    -0.1875 * (-1 + n) * (-1 + s) * (1 / 3 + s),
                    -0.0625 * (-1 + s) ** 2 * (1 + s),
                ],
                # ∇N4 =
                [
                    0.125 * (n**3 - 3 * (-1 + s**2) + n * (-4 + 3 * s**2)),
                    0.125 * (-3 - 4 * s + s**3 + 3 * n**2 * (1 + s)),
                ],
                # ∇N5 =
                [
                    0.0625 * (1 - n) ** 2 * (1 + n),
                    0.0625 * (-1 + n) * (1 + 3 * n) * (1 + s),
                ],
                # ∇N6 =
                [
                    -0.0625
                    + 0.0625 * n
                    + s * (0.125 - 0.125 * n + (0.1875 - 0.1875 * n) * s),
                    (0.0625 - 0.0625 * s) * (1 + s) ** 2,
                ],
                # ∇N7 =
                [
                    -(n + 1) * (n**2 - n + 3 * s**2 - 3) / 8,
                    -(s + 1) * (3 * n**2 + s**2 - s - 3) / 8,
                ],
                # ∇N8 =
                [
                    0.0625 * (-1 + n) * (1 + n) ** 2,
                    0.1875 * (-1 / 3 + n) * (1 + n) * (1 + s),
                ],
                # ∇N9 =
                [
                    0.1875 * (1 + n) * (-1 / 3 + s) * (1 + s),
                    0.0625 * (-1 + s) * (1 + s) ** 2,
                ],
                # ∇N10 =
                [
                    (n + 1) * (n**2 - n + 3 * s**2 - 3) / 8,
                    (s - 1) * (3 * n**2 + s**2 + s - 3) / 8,
                ],
                # ∇N11 =
                [
                    -(n - 1) * (n + 1) ** 2 / 16,
                    -(n + 1) * (3 * n - 1) * (s - 1) / 16,
                ],
                # ∇N12 =
                [
                    (n + 1) * (s - 1) * (3 * s + 1) / 16,
                    (s - 1) ** 2 * (s + 1) / 16,
                ],
            ]
        )

    def second_derivative(self, s: np.array, n: np.array):
        """
        Evaluate the second derivatives of the ACM shape functions.

        Args:
            s (np.ndarray): Local horizontal (ξ) coordinates.
            n (np.ndarray): Local vertical (η) coordinates.

        Returns:
            np.ndarray: Array of shape (12, 2, 2) where:
                        - Row k contains the second derivative of φ_k
                        - Column 0 = ∂²φ/∂ξ², Column 1, 2 = ∂²φ/∂ξ∂η, Column 3 = ∂²φ/∂η²
        """
        return np.array(
            [
                # ∇²N1 =
                [
                    [-3 * s * (n - 1) / 4, -(3 * n**2 + 3 * s**2 - 4) / 8],
                    [-(3 * n**2 + 3 * s**2 - 4) / 8, -3 * n * (s - 1) / 4],
                ],
                # ∇²N2 =
                [
                    [0, -(n - 1) * (3 * n + 1) / 16],
                    [-(n - 1) * (3 * n + 1) / 16, -(3 * n - 1) * (s - 1) / 8],
                ],
                # ∇²N3 =
                [
                    [
                        -0.125 * (-1 + n) * (-1 + 3 * s),
                        -0.1875 * (-1 + s) * (1 / 3 + s),
                    ],
                    [-0.1875 * (-1 + s) * (1 / 3 + s), 0],
                ],
                # ∇²N4 =
                [
                    [(-0.75 + 0.75 * n) * s, 0.125 * (-4 + 3 * n**2 + 3 * s**2)],
                    [0.125 * (-4 + 3 * n**2 + 3 * s**2), 0.75 * n * (1 + s)],
                ],
                # ∇²N5 =
                [
                    [0, 0.1875 * (-1 + n) * (1 / 3 + n)],
                    [0.1875 * (-1 + n) * (1 / 3 + n), 0.125 * (-1 + 3 * n) * (1 + s)],
                ],
                # ∇²N6 =
                [
                    [-0.125 * (-1 + n) * (1 + 3 * s), -0.1875 * (-1 / 3 + s) * (1 + s)],
                    [-0.1875 * (-1 / 3 + s) * (1 + s), 0],
                ],
                # ∇²N7 =
                [
                    [-0.75 * s * (1 + n), 1 / 8 * (4 - 3 * n**2 - 3 * s**2)],
                    [1 / 8 * (4 - 3 * n**2 - 3 * s**2), -0.75 * n * (1 + s)],
                ],
                # ∇²N8 =
                [
                    [0, 0.1875 * (n - 1 / 3) * (1 + n)],
                    [0.1875 * (n - 1 / 3) * (1 + n), 0.125 * (1 + 3 * n) * (1 + s)],
                ],
                # ∇²N9 =
                [
                    [0.125 * (1 + n) * (1 + 3 * s), 0.1875 * (s - 1 / 3) * (1 + s)],
                    [0.1875 * (s - 1 / 3) * (1 + s), 0],
                ],
                # ∇²N10 =
                [
                    [3 * s * (n + 1) / 4, (3 * n**2 + 3 * s**2 - 4) / 8],
                    [(3 * n**2 + 3 * s**2 - 4) / 8, 3 * n * (s - 1) / 4],
                ],
                # ∇²N11 =
                [
                    [0, -(n + 1) * (3 * n - 1) / 16],
                    [-(n + 1) * (3 * n - 1) / 16, -(3 * n + 1) * (s - 1) / 8],
                ],
                # ∇²N12 =
                [
                    [(n + 1) * (3 * s - 1) / 8, (s - 1) * (3 * s + 1) / 16],
                    [(s - 1) * (3 * s + 1) / 16, 0],
                ],
            ]
        )
