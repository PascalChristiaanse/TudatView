import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk

orbit1 = inputs[0]
orbit2 = inputs[1]

# Extract positions
p1 = np.array(
    [orbit1.GetPoint(i) for i in range(orbit1.GetNumberOfPoints())],
    dtype=float
)

p2 = np.array(
    [orbit2.GetPoint(i) for i in range(orbit2.GetNumberOfPoints())],
    dtype=float
)

# Extract time
t1 = vtk_to_numpy(
    orbit1.GetPointData().GetArray("Time")
).astype(float)

t2 = vtk_to_numpy(
    orbit2.GetPointData().GetArray("Time")
).astype(float)

# Sort
idx1 = np.argsort(t1)
idx2 = np.argsort(t2)

t1 = t1[idx1]
p1 = p1[idx1]

t2 = t2[idx2]
p2 = p2[idx2]

# Common time range
t_start = np.maximum(np.min(t1), np.min(t2))
t_end   = np.minimum(np.max(t1), np.max(t2))

if t_start >= t_end:
    raise RuntimeError("The two orbits have no overlapping time range")

# Orbit 1 samples in common range
mask = (t1 >= t_start) & (t1 <= t_end)

times = t1[mask]
pos1 = p1[mask]

# Interpolate orbit 2
x2 = np.interp(times, t2, p2[:, 0])
y2 = np.interp(times, t2, p2[:, 1])
z2 = np.interp(times, t2, p2[:, 2])

# Distance
distance = np.sqrt(
    (pos1[:, 0] - x2)**2 +
    (pos1[:, 1] - y2)**2 +
    (pos1[:, 2] - z2)**2
)

# ------------------------------------------------------------
# Construct output PolyData
# ------------------------------------------------------------
output.Initialize()

points = vtk.vtkPoints()
points.SetNumberOfPoints(len(times))

for i in range(len(times)):
    points.SetPoint(
        i,
        pos1[i, 0],
        pos1[i, 1],
        pos1[i, 2]
    )

output.SetPoints(points)

# Time array
time_vtk = numpy_to_vtk(times, deep=True)
time_vtk.SetName("Time")
output.GetPointData().AddArray(time_vtk)

# Distance array
distance_vtk = numpy_to_vtk(distance, deep=True)
distance_vtk.SetName("OrbitDistance")
output.GetPointData().AddArray(distance_vtk)