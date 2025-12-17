"""
Circle geometry generator for DSMC solver.

Generates geometry data for a solid circle, including boundary segments
and cells that are entirely inside the circle.

Usage:
    python scripts/geometry/circle.py \
            -o output.dat \
            --nx 20 --ny 20 \
            --lx 0.3 --ly 0.3 \
            --cx 0.15 --cy 0.15 --radius 0.05
"""

import math
from typing import Tuple, List

from base import (
    GridConfig,
    GeometryData,
    GeometryGenerator,
    Segment,
    create_base_parser,
    grid_from_args,
)


class CircleGenerator(GeometryGenerator):
    """Generator for solid circle geometry."""

    def __init__(
        self,
        grid: GridConfig,
        center_x: float,
        center_y: float,
        radius: float,
        num_segments: int = 32
    ):
        """
        Initialize the circle generator.

        Args:
            grid: Grid configuration
            center_x: X coordinate of circle center
            center_y: Y coordinate of circle center
            radius: Circle radius
            num_segments: Number of segments to approximate the circle
        """
        super().__init__(grid)
        self.center_x = center_x
        self.center_y = center_y
        self.radius = radius
        self.num_segments = num_segments

    def _point_in_circle(self, x: float, y: float) -> bool:
        """Check if a point is inside the circle."""
        dx = x - self.center_x
        dy = y - self.center_y
        return dx * dx + dy * dy < self.radius * self.radius

    def _cell_fully_inside(self, cx: int, cy: int) -> bool:
        """Check if a cell is entirely inside the circle."""
        x_min, y_min, x_max, y_max = self.grid.cell_bounds(cx, cy)
        # Check all four corners
        corners = [
            (x_min, y_min),
            (x_max, y_min),
            (x_min, y_max),
            (x_max, y_max),
        ]
        return all(self._point_in_circle(x, y) for x, y in corners)

    def _cell_intersects_circle(self, cx: int, cy: int) -> bool:
        """Check if a cell intersects the circle boundary."""
        x_min, y_min, x_max, y_max = self.grid.cell_bounds(cx, cy)
        
        # Find closest point on cell to circle center
        closest_x = max(x_min, min(self.center_x, x_max))
        closest_y = max(y_min, min(self.center_y, y_max))
        
        # Check if closest point is inside circle
        dx = closest_x - self.center_x
        dy = closest_y - self.center_y
        closest_inside = dx * dx + dy * dy < self.radius * self.radius
        
        # Check if cell center is inside
        cell_cx, cell_cy = self.grid.cell_center(cx, cy)
        center_inside = self._point_in_circle(cell_cx, cell_cy)
        
        # Check corners
        corners = [
            (x_min, y_min),
            (x_max, y_min),
            (x_min, y_max),
            (x_max, y_max),
        ]
        corners_inside = [self._point_in_circle(x, y) for x, y in corners]
        
        # Cell intersects if some corners are inside and some are outside
        # or if the closest point is inside but not all corners are inside
        if any(corners_inside) and not all(corners_inside):
            return True
        if closest_inside and not all(corners_inside):
            return True
        return False

    def _circle_segment_in_cell(
        self, cx: int, cy: int
    ) -> Tuple[float, float, float, float, float, float] | None:
        """
        Calculate the circle segment within a cell.

        Returns:
            Tuple of (start_x, start_y, end_x, end_y, normal_x, normal_y)
            or None if no segment exists in this cell.
        """
        x_min, y_min, x_max, y_max = self.grid.cell_bounds(cx, cy)
        
        # Find intersection points of circle with cell edges
        intersections: List[Tuple[float, float]] = []
        
        # Check each edge of the cell
        edges = [
            ((x_min, y_min), (x_max, y_min)),  # Bottom
            ((x_max, y_min), (x_max, y_max)),  # Right
            ((x_max, y_max), (x_min, y_max)),  # Top
            ((x_min, y_max), (x_min, y_min)),  # Left
        ]
        
        for (ex1, ey1), (ex2, ey2) in edges:
            pts = self._line_circle_intersections(ex1, ey1, ex2, ey2)
            for px, py in pts:
                # Check if point is on the edge segment
                if (min(ex1, ex2) - 1e-9 <= px <= max(ex1, ex2) + 1e-9 and
                    min(ey1, ey2) - 1e-9 <= py <= max(ey1, ey2) + 1e-9):
                    intersections.append((px, py))
        
        if len(intersections) < 2:
            return None
        
        # Remove duplicates (intersection at corners)
        unique_intersections = []
        for p in intersections:
            is_dup = False
            for q in unique_intersections:
                if abs(p[0] - q[0]) < 1e-9 and abs(p[1] - q[1]) < 1e-9:
                    is_dup = True
                    break
            if not is_dup:
                unique_intersections.append(p)
        
        if len(unique_intersections) < 2:
            return None
        
        # Take the first two intersection points
        p1, p2 = unique_intersections[0], unique_intersections[1]
        
        # Order points counter-clockwise around the circle
        angle1 = math.atan2(p1[1] - self.center_y, p1[0] - self.center_x)
        angle2 = math.atan2(p2[1] - self.center_y, p2[0] - self.center_x)
        
        if angle1 > angle2:
            p1, p2 = p2, p1
        
        # Calculate outward normal (pointing away from center)
        mid_x = (p1[0] + p2[0]) / 2
        mid_y = (p1[1] + p2[1]) / 2
        normal_x = mid_x - self.center_x
        normal_y = mid_y - self.center_y
        norm_len = math.sqrt(normal_x ** 2 + normal_y ** 2)
        if norm_len > 1e-9:
            normal_x /= norm_len
            normal_y /= norm_len
        
        return (p1[0], p1[1], p2[0], p2[1], normal_x, normal_y)

    def _line_circle_intersections(
        self, x1: float, y1: float, x2: float, y2: float
    ) -> List[Tuple[float, float]]:
        """
        Find intersection points of a line segment with the circle.

        Args:
            x1, y1: Start of line segment
            x2, y2: End of line segment

        Returns:
            List of intersection points
        """
        # Translate to circle center
        dx = x2 - x1
        dy = y2 - y1
        fx = x1 - self.center_x
        fy = y1 - self.center_y
        
        a = dx * dx + dy * dy
        b = 2 * (fx * dx + fy * dy)
        c = fx * fx + fy * fy - self.radius * self.radius
        
        discriminant = b * b - 4 * a * c
        
        if discriminant < 0 or a < 1e-12:
            return []
        
        intersections = []
        sqrt_disc = math.sqrt(discriminant)
        
        t1 = (-b - sqrt_disc) / (2 * a)
        t2 = (-b + sqrt_disc) / (2 * a)
        
        for t in [t1, t2]:
            if -1e-9 <= t <= 1 + 1e-9:
                px = x1 + t * dx
                py = y1 + t * dy
                intersections.append((px, py))
        
        return intersections

    def generate(self) -> GeometryData:
        """Generate circle geometry data."""
        # Iterate over all cells
        for cy in range(self.grid.ny):
            for cx in range(self.grid.nx):
                cell_id = self.grid.cell_id(cx, cy)
                
                # Check if cell is fully inside
                if self._cell_fully_inside(cx, cy):
                    self.data.add_inside_cell(cell_id)
                    continue
                
                # Check if cell intersects the circle boundary
                if self._cell_intersects_circle(cx, cy):
                    segment_data = self._circle_segment_in_cell(cx, cy)
                    if segment_data:
                        start_x, start_y, end_x, end_y, nx, ny = segment_data
                        segment = Segment(
                            cell_id=cell_id,
                            start_x=start_x,
                            start_y=start_y,
                            end_x=end_x,
                            end_y=end_y,
                            normal_x=nx,
                            normal_y=ny
                        )
                        self.data.add_segment(segment)
        
        return self.data


def main():
    """Main entry point."""
    parser = create_base_parser("Generate circle geometry for DSMC solver")
    parser.add_argument('--cx', type=float, required=True,
                        help='Circle center X coordinate')
    parser.add_argument('--cy', type=float, required=True,
                        help='Circle center Y coordinate')
    parser.add_argument('--radius', type=float, required=True,
                        help='Circle radius')
    
    args = parser.parse_args()
    
    grid = grid_from_args(args)
    
    generator = CircleGenerator(
        grid=grid,
        center_x=args.cx,
        center_y=args.cy,
        radius=args.radius,
    )
    
    generator.generate()
    generator.save(
        args.output,
        description=f"Circle: center=({args.cx}, {args.cy}), radius={args.radius}"
    )


if __name__ == '__main__':
    main()
