import pytest
import numpy as np
from mpi4py import MPI
import dolfinx



from VascularFlow.FEniCSx.FluidFlow.Pressure2DVelocityInlet import pressure_2d_velocity_inlet
from VascularFlow.FEniCSx.PostProcessing import visualize_mixed

@pytest.mark.parametrize(
    "fluid_domain, reynolds_number",
    [
        (
            dolfinx.mesh.create_rectangle(
                MPI.COMM_WORLD,
                np.array([[0, 0], [50, 1]]),
                [500, 10],
                cell_type=dolfinx.mesh.CellType.triangle,
            ),
            8,
        ),
    ],
)
def test_pressure_2d_velocity_inlet(fluid_domain, reynolds_number, plot: bool = True):
    mixed_function = pressure_2d_velocity_inlet(fluid_domain, reynolds_number)

    if plot:
        visualize_mixed(mixed_function, fluid_domain)


