import numpy as np
from scipy.stats import qmc

d = 5
n_samples = 10

x_channel_right_min,   x_channel_right_max   = 0.0  , 1.0
x_channel_left_min,    x_channel_left_max    = 2.0  , 3.0
dp_min,                dp_max                = 0.0  , 100.0
channel_height_min,    channel_height_max    = 1.0  , 5.0
p_external_min,        p_external_max        = 0.0  , 10

lower_bounds = np.array([
    x_channel_right_min,
    x_channel_left_min,
    dp_min,
    channel_height_min,
    p_external_min,
])

upper_bounds = np.array([
    x_channel_right_max,
    x_channel_left_max,
    dp_max,
    channel_height_max,
    p_external_max,
])

# LHS sampler generation
sampler = qmc.LatinHypercube(d=d)

sample_unit = sampler.random(n=n_samples)

X_design = qmc.scale(sample_unit, lower_bounds, upper_bounds)

param_names = ["idx", "x_right", "x_left", "dp", "channel_height", "p_external"]
widths = [4, 12, 12, 14, 17, 14]

# Header
header = "".join(f"{name:<{w}}" for name, w in zip(param_names, widths))
print(header)

# Rows
for i, row in enumerate(X_design):
    print(f"{i:<{widths[0]}}"
          f"{row[0]:<{widths[1]}.6f}"
          f"{row[1]:<{widths[2]}.6f}"
          f"{row[2]:<{widths[3]}.6f}"
          f"{row[3]:<{widths[4]}.6f}"
          f"{row[4]:<{widths[5]}.6f}")


n_samples = X_design.shape[0]
Q_values = np.array([0.01265, 0.944, 2.0998, 3.3346, 10.8706])

for i in range(n_samples):
    x_right, x_left, dp, channel_height, p_external = X_design[i,:]

########################################################################################################################





