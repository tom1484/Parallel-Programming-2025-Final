"""
Geometry generation package for DSMC solver.

This package provides tools for generating geometry files (.dat) that define
solid objects in the simulation domain.

Modules:
    base: Core data structures and utilities
    circle: Circle geometry generator

Usage:
    # As a library
    from geometry.base import GridConfig, GeometryData
    from geometry.circle import CircleGenerator

    grid = GridConfig(nx=20, ny=20, lx=0.3, ly=0.3)
    gen = CircleGenerator(grid, center_x=0.15, center_y=0.15, radius=0.05)
    gen.generate()
    gen.save("circle.dat")

    # As a command-line tool
    python -m geometry.circle -o circle.dat --nx 20 --ny 20 --lx 0.3 --ly 0.3 \
                              --cx 0.15 --cy 0.15 --radius 0.05
"""

from .base import (
    GridConfig,
    Segment,
    InsideCell,
    GeometryData,
    GeometryGenerator,
    create_base_parser,
    grid_from_args,
)

from .circle import CircleGenerator

__all__ = [
    'GridConfig',
    'Segment',
    'InsideCell',
    'GeometryData',
    'GeometryGenerator',
    'CircleGenerator',
    'create_base_parser',
    'grid_from_args',
]
