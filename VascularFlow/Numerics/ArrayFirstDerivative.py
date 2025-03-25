import numpy as np

def array_first_derivative(array: np.ndarray, dx_e: float) -> np.ndarray:
    derivative = np.zeros_like(array)

    # Central difference for interior points
    derivative[1:-1] = (array[2:] - array[:-2]) / (2 * dx_e)

    # Forward difference for the first point
    derivative[0] = (-3 * array[0] + 4 * array[1] - array[2]) / (2 * dx_e)

    # Backward difference for the last point
    derivative[-1] = (3 * array[-1] - 4 * array[-2] + array[-3]) / (2 * dx_e)

    return derivative

