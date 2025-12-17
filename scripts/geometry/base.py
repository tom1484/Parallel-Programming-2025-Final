"""
Base module for geometry generation.

This module provides core data structures and utilities for generating
geometry files (.dat) for the DSMC solver.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Tuple
import argparse


@dataclass
class GridConfig:
    """Grid configuration matching the simulation setup."""
    nx: int          # Number of cells in X direction
    ny: int          # Number of cells in Y direction
    lx: float        # Domain width (meters)
    ly: float        # Domain height (meters)

    @property
    def dx(self) -> float:
        """Cell width."""
        return self.lx / self.nx

    @property
    def dy(self) -> float:
        """Cell height."""
        return self.ly / self.ny

    def cell_id(self, cx: int, cy: int) -> int:
        """Convert cell coordinates to cell ID."""
        return cy * self.nx + cx

    def cell_coords(self, cell_id: int) -> Tuple[int, int]:
        """Convert cell ID to cell coordinates (cx, cy)."""
        cy = cell_id // self.nx
        cx = cell_id % self.nx
        return cx, cy

    def cell_center(self, cx: int, cy: int) -> Tuple[float, float]:
        """Get the center position of a cell."""
        x = (cx + 0.5) * self.dx
        y = (cy + 0.5) * self.dy
        return x, y

    def cell_bounds(self, cx: int, cy: int) -> Tuple[float, float, float, float]:
        """Get the bounds of a cell (x_min, y_min, x_max, y_max)."""
        x_min = cx * self.dx
        y_min = cy * self.dy
        x_max = (cx + 1) * self.dx
        y_max = (cy + 1) * self.dy
        return x_min, y_min, x_max, y_max


@dataclass
class Segment:
    """A line segment representing part of a solid boundary."""
    cell_id: int
    start_x: float
    start_y: float
    end_x: float
    end_y: float
    normal_x: float
    normal_y: float

    def to_record(self) -> str:
        """Convert to geometry file record format (type 0)."""
        return (f"{self.cell_id} 0 "
                f"{self.start_x:.6f} {self.start_y:.6f} "
                f"{self.end_x:.6f} {self.end_y:.6f} "
                f"{self.normal_x:.6f} {self.normal_y:.6f}")


@dataclass
class InsideCell:
    """A cell that is entirely inside a solid object."""
    cell_id: int

    def to_record(self) -> str:
        """Convert to geometry file record format (type 1)."""
        return f"{self.cell_id} 1"


@dataclass
class GeometryData:
    """Container for all geometry data."""
    grid: GridConfig
    segments: List[Segment] = field(default_factory=list)
    inside_cells: List[InsideCell] = field(default_factory=list)

    def add_segment(self, segment: Segment) -> None:
        """Add a segment to the geometry."""
        self.segments.append(segment)

    def add_inside_cell(self, cell_id: int) -> None:
        """Mark a cell as being inside a solid object."""
        self.inside_cells.append(InsideCell(cell_id))

    def save(self, path: str, description: str = "") -> None:
        """
        Save geometry data to a .dat file.

        Args:
            path: Output file path
            description: Optional description comment
        """
        with open(path, 'w') as f:
            # Write header comments
            if description:
                f.write(f"# {description}\n")
            f.write("# Header: nx ny lx ly\n")
            f.write(f"{self.grid.nx} {self.grid.ny} {self.grid.lx} {self.grid.ly}\n")
            f.write("# Format: cell_id type [start_x start_y end_x end_y normal_x normal_y]\n")
            f.write("# type 0 = segment, type 1 = inside solid\n")

            # Write segments
            for segment in self.segments:
                f.write(segment.to_record() + "\n")

            # Write inside cells
            for inside in self.inside_cells:
                f.write(inside.to_record() + "\n")

        print(f"Saved geometry to {path}: {len(self.segments)} segments, "
              f"{len(self.inside_cells)} inside cells")


class GeometryGenerator(ABC):
    """Abstract base class for geometry generators."""

    def __init__(self, grid: GridConfig):
        """
        Initialize the generator with a grid configuration.

        Args:
            grid: Grid configuration matching the simulation
        """
        self.grid = grid
        self.data = GeometryData(grid)

    @abstractmethod
    def generate(self) -> GeometryData:
        """
        Generate the geometry data.

        Returns:
            GeometryData containing segments and inside cells
        """
        pass

    def save(self, path: str, description: str = "") -> None:
        """Save the generated geometry to a file."""
        self.data.save(path, description)


def create_base_parser(description: str) -> argparse.ArgumentParser:
    """
    Create a base argument parser with common grid options.

    Args:
        description: Description for the argument parser

    Returns:
        ArgumentParser with common options
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-o', '--output', type=str, required=True,
                        help='Output geometry file path (.dat)')
    parser.add_argument('--nx', type=int, required=True,
                        help='Number of cells in X direction')
    parser.add_argument('--ny', type=int, required=True,
                        help='Number of cells in Y direction')
    parser.add_argument('--lx', type=float, required=True,
                        help='Domain width (meters)')
    parser.add_argument('--ly', type=float, required=True,
                        help='Domain height (meters)')
    return parser


def grid_from_args(args: argparse.Namespace) -> GridConfig:
    """
    Create a GridConfig from parsed arguments.

    Args:
        args: Parsed argument namespace

    Returns:
        GridConfig instance
    """
    return GridConfig(
        nx=args.nx,
        ny=args.ny,
        lx=args.lx,
        ly=args.ly
    )
