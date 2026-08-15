"""ParaView reader plugin for Tudat simulation results (.tudat files).

This module provides a ParaView Python plugin that reads binary Tudat
simulation result files (from :meth:`save_to_binary` / :meth:`save_binary`)
and converts trajectory data with dependent variables into VTK PolyData
suitable for visualization.

The output is a polyline (temporal trajectory) with point-data arrays:
  - Time (epoch)
  - Cartesian positions (x, y, z) — the mesh points themselves
  - Velocities (vx, vy, vz)
  - All dependent variables stored in the simulation result

Usage in ParaView
-----------------
  1. Place this file (or the whole package) somewhere ParaView can find it.
  2. *Tools* → *Manage Plugins* → *Load New* → select this file.
  3. *File* → *Open* → pick a ``.tudat`` file.
  4. The reader creates a trajectory polyline; apply the *Tube* filter or
     *Glyph* filter for better visibility.

Alternatively, use as a standalone converter::

    from TudatParaview.single_trajectory_reader import (
        load_tudat_results,
        simulation_results_to_polydata,
    )
    polydata = simulation_results_to_polydata("my_simulation.tudat")
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from tudatpy.dynamics.propagation import SingleArcSimulationResults

# ---------------------------------------------------------------------------
# VTK / ParaView imports (only available inside ParaView or with vtk installed)
# ---------------------------------------------------------------------------
try:
    from vtkmodules.util.vtkAlgorithm import VTKPythonAlgorithmBase
    from vtkmodules.vtkCommonCore import vtkDoubleArray, vtkPoints
    from vtkmodules.vtkCommonDataModel import vtkPolyData, vtkCellArray

    _VTK_AVAILABLE = True
except ImportError:
    _VTK_AVAILABLE = False

# ParaView plugin decorators (only available inside ParaView)
try:
    from paraview.util.vtkAlgorithm import smproxy, smproperty, smdomain, smhint

    _PARAVIEW_AVAILABLE = True
except ImportError:
    _PARAVIEW_AVAILABLE = False


# ===================================================================
#  Standalone helpers  (no ParaView dependency)
# ===================================================================


def load_tudat_results(filepath: str | Path):
    """Load a Tudat simulation result from a binary file.

    Parameters
    ----------
    filepath : str | Path
        Path to the ``.tudat`` file.

    Returns
    -------
    SingleArcSimulationResults
        The loaded simulation result.

    Raises
    ------
    RuntimeError
        If the file cannot be loaded.
    """
    path = str(filepath)
    # load_from_binary auto-appends .tudat, but ParaView already passes the
    # path with .tudat, so strip it to avoid a double extension.
    if path.endswith(".tudat"):
        path = path[:-6]
    return SingleArcSimulationResults.load_from_binary(path)


def simulation_results_to_polydata(results, output=None) -> "vtkPolyData":
    """Convert a Tudat ``SingleArcSimulationResults`` to ``vtkPolyData``.

    The output contains:

    * **Points** — Cartesian position (x, y, z) at each epoch.
    * **Lines** — A single polyline connecting points in temporal order.
    * **Point-data arrays:**
        - ``Time`` — simulation epoch at each point.
        - ``Velocity`` — (vx, vy, vz) 3-component array.
        - One array per dependent variable (named by the variable ID string).

    Parameters
    ----------
    results : SingleArcSimulationResults
        A loaded Tudat simulation result.
    output : vtkPolyData, optional
        If provided, populate this existing polydata instead of creating a new
        one.  Use this inside ParaView's ``RequestData`` to build directly on
        the pipeline output, avoiding any copy.

    Returns
    -------
    vtkPolyData
        The populated polydata (same as *output* if one was given).
    """
    if not _VTK_AVAILABLE:
        raise ImportError(
            "VTK is not available. Run this inside ParaView or "
            "install vtk: pip install vtk"
        )

    state_history = results.state_history
    dep_var_history = results.dependent_variable_history

    # ---- Build sorted time grid -------------------------------------------
    # Both histories share the same epochs — use state epochs as the master key.
    times = np.array(sorted(state_history.keys()))
    n_points = len(times)

    if n_points == 0:
        raise ValueError("Simulation result contains no states.")

    # ---- Extract positions & velocities ------------------------------------
    coords = np.empty((n_points, 3))
    vels = np.empty((n_points, 3))

    for i, t in enumerate(times):
        s = state_history[float(t)]
        coords[i] = [float(s[0]), float(s[1]), float(s[2])]
        vels[i] = [float(s[3]), float(s[4]), float(s[5])]

    # ---- Build vtkPoints ---------------------------------------------------
    pts = vtkPoints()
    pts.SetNumberOfPoints(n_points)
    for i in range(n_points):
        pts.SetPoint(i, *coords[i])

    # ---- Build polyline connectivity ---------------------------------------
    lines = vtkCellArray()
    lines.InsertNextCell(n_points)
    for i in range(n_points):
        lines.InsertCellPoint(i)

    # ---- Build or reuse polydata -------------------------------------------
    if output is None:
        polydata = vtkPolyData()
    else:
        polydata = output

    polydata.SetPoints(pts)
    polydata.SetLines(lines)

    # ---- Add time array ----------------------------------------------------
    time_arr = vtkDoubleArray()
    time_arr.SetName("Time")
    time_arr.SetNumberOfValues(n_points)
    for i in range(n_points):
        time_arr.SetValue(i, float(times[i]))
    polydata.GetPointData().AddArray(time_arr)

    # ---- Add velocity array ------------------------------------------------
    vel_arr = vtkDoubleArray()
    vel_arr.SetName("Velocity")
    vel_arr.SetNumberOfComponents(3)
    vel_arr.SetNumberOfTuples(n_points)
    for i in range(n_points):
        vel_arr.SetTuple3(i, float(vels[i, 0]), float(vels[i, 1]), float(vels[i, 2]))
    polydata.GetPointData().AddArray(vel_arr)

    # ---- Add dependent variable arrays -------------------------------------
    _add_dependent_variable_arrays(polydata, results, times)

    # ---- Set active scalars (default to the first dep-var, or Time) --------
    if results.dependent_variable_ids:
        first_name = next(iter(results.dependent_variable_ids.values()))
        polydata.GetPointData().SetActiveScalars(first_name)
    else:
        polydata.GetPointData().SetActiveScalars("Time")

    return polydata


def _add_dependent_variable_arrays(
    polydata: "vtkPolyData",
    results,
    times: np.ndarray,
) -> None:
    """Add dependent variables from *results* as point-data arrays on
    *polydata*, indexed by *times*."""
    dep_var_history = results.dependent_variable_history
    dep_var_ids = results.dependent_variable_ids
    n_points = len(times)

    # dep_var_ids maps { (start_idx, size) : name_string }
    # We need the ordered list of id entries.
    ordered_ids: list[tuple[tuple[int, int], str]] = sorted(
        dep_var_ids.items(), key=lambda kv: kv[0][0]
    )

    for (start_idx, size), name in ordered_ids:
        arr = vtkDoubleArray()
        arr.SetName(name)
        if size > 1:
            arr.SetNumberOfComponents(size)
            arr.SetNumberOfTuples(n_points)
        else:
            arr.SetNumberOfValues(n_points)

        for i, t in enumerate(times):
            dv = dep_var_history[float(t)]
            segment = dv[start_idx : start_idx + size]
            if size > 1:
                arr.SetTuple(i, [float(segment[j]) for j in range(size)])
            else:
                arr.SetValue(i, float(segment))

        polydata.GetPointData().AddArray(arr)


# ===================================================================
#  ParaView Plugin  (decorator-based reader)
# ===================================================================

if _PARAVIEW_AVAILABLE and _VTK_AVAILABLE:

    @smproxy.reader(
        name="TudatSingleArcReader",
        label="Tudat Single Arc Reader",
        extensions="tudat",
        file_description="Tudat Simulation Results",
    )
    class TudatSingleArcReader(VTKPythonAlgorithmBase):
        """ParaView reader for ``.tudat`` binary simulation-result files.

        Produces a ``vtkPolyData`` polyline with point-data arrays for time,
        velocity, and all dependent variables.
        """

        def __init__(self):
            VTKPythonAlgorithmBase.__init__(
                self, nInputPorts=0, nOutputPorts=1, outputType="vtkPolyData"
            )
            self._filename: Optional[str] = None

        # ------------------------------------------------------------------
        # File name property
        # ------------------------------------------------------------------
        @smproperty.stringvector(name="FileName", number_of_elements=1, panel_visibility="never")
        @smdomain.filelist()
        @smhint.filechooser(extensions="tudat", file_description="Tudat Simulation Results (.tudat)")
        def SetFileName(self, name: str) -> None:
            """Specify the ``.tudat`` file to read."""
            if self._filename != name:
                self._filename = name
                self.Modified()

        # ------------------------------------------------------------------
        # Output information (pipeline bounds)
        # ------------------------------------------------------------------
        def RequestInformation(self, request, inInfoVec, outInfoVec) -> int:
            """Provide meta-data about the output to the ParaView pipeline."""
            return 1

        # ------------------------------------------------------------------
        # Data production
        # ------------------------------------------------------------------
        def RequestData(self, request, inInfoVec, outInfoVec) -> int:
            """Produce the output polydata from the current file."""
            if self._filename is None:
                raise RuntimeError("No filename specified")

            results = load_tudat_results(self._filename)
            output = vtkPolyData.GetData(outInfoVec, 0)
            simulation_results_to_polydata(results, output=output)
            return 1

else:
    # Provide a minimal stub so the module can be imported without ParaView
    class TudatSingleArcReader:  # type: ignore[no-redef]
        """Stub available when ParaView is not present.

        Use the free functions :func:`load_tudat_results` and
        :func:`simulation_results_to_polydata` directly instead.
        """
        def __init__(self):
            raise RuntimeError(
                "TudatSingleArcReader requires ParaView. "
                "Use load_tudat_results() and simulation_results_to_polydata() directly."
            )


