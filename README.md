# ClimateModel

[Model still in construction.]

Author: Christoph Heim

Simple global climate model, hydrostatic and on a regular lat-lon grid.
Implemented in Python. Performance is obtained by using just-in-time
compiled code with the Python package numba either on CPU or on GPU
(for GPU Cuda is required).

Implementation of dynamical core according to:
Jacobson 2005
Fundamentals of Atmospheric Modeling, Second Edition Chapter 7

History:
~20181001       Start of development
20190531        First good version of dynamical core

Usage:
- Simulation settings can be done in namelist.py
- To launch simulation run 'python solver.py' which is the main entrance
  to the simulation.

Model description:
Vertical direction is represented in sigma pressure coordinates.
Horizontal directions are represented on a latitude-longitude grid.
