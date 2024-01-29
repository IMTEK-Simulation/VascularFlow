import numpy as np

def test_homogeneous_tube():
    nb_grid_pts = 10
    radius = 1.0 * np.ones(nb_grid_pts)

    # compute pressure here
    # check with
    # np.testing.assert_allclose(p_numerical, p_analytic)