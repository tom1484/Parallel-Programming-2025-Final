"""
Ellipse geometry generator for DSMC solver.

Generates geometry data for a solid ellipse, including boundary segments
and cells that are entirely inside the ellipse.

Usage:
    python scripts/geometry/ellipse.py \
            -o output.dat \
            --nx 20 --ny 20 \
            --lx 0.3 --ly 0.3 \
            --cx 0.15 --cy 0.15 --a 0.08 --b 0.05
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


class EllipseGenerator(GeometryGenerator):
    """Generator for solid ellipse geometry."""

    def __init__(
        self,
        grid: GridConfig,
        center_x: float,
        center_y: float,
        semi_major: float,
        semi_minor: float,
    ):
        """
        Initialize the ellipse generator.

        Args:
            grid: Grid configuration
            center_x: X coordinate of ellipse center
            center_y: Y coordinate of ellipse center
            semi_major: Semi-major axis (horizontal radius)
            semi_minor: Semi-minor axis (vertical radius)
        """
        super().__init__(grid)
        self.center_x = center_x
        self.center_y = center_y
        self.semi_major = semi_major
        self.semi_minor = semi_minor

    def _point_in_ellipse(self, x: float, y: float) -> bool:
        """Check if a point is inside the ellipse."""
        dx = x - self.center_x
        dy = y - self.center_y
        return (dx * dx) / (self.semi_major ** 2) + (dy * dy) / (self.semi_minor ** 2) < 1.0

    def _cell_fully_inside(self, cx: int, cy: int) -> bool:
        """Check if a cell is entirely inside the ellipse."""
        x_min, y_min, x_max, y_max = self.grid.cell_bounds(cx, cy)
        # Check all four corners
        corners = [
            (x_min, y_min),
            (x_max, y_min),
            (x_min, y_max),
            (x_max, y_max),
        ]
        return all(self._point_in_ellipse(x, y) for x, y in corners)

    def _cell_intersects_ellipse(self, cx: int, cy: int) -> bool:
        """Check if a cell intersects the ellipse boundary."""
        x_min, y_min, x_max, y_max = self.grid.cell_bounds(cx, cy)
        
        # Find closest point on cell to ellipse center
        closest_x = max(x_min, min(self.center_x, x_max))
        closest_y = max(y_min, min(self.center_y, y_max))
        
        # Check if closest point is inside ellipse
        closest_inside = self._point_in_ellipse(closest_x, closest_y)
        
        # Check corners
        corners = [
            (x_min, y_min),
            (x_max, y_min),
            (x_min, y_max),
            (x_max, y_max),
        ]
        corners_inside = [self._point_in_ellipse(x, y) for x, y in corners]
        
        # Cell intersects if some corners are inside and some are outside
        # or if the closest point is inside but not all corners are inside
        if any(corners_inside) and not all(corners_inside):
            return True
        if closest_inside and not all(corners_inside):
            return True
        return False

    def _ellipse_segment_in_cell(
        self, cx: int, cy: int
    ) -> Tuple[float, float, float, float, float, float] | None:
        """
        Calculate the ellipse segment within a cell.

        Returns:
            Tuple of (start_x, start_y, end_x, end_y, normal_x, normal_y)
            or None if no segment exists in this cell.
        """
        x_min, y_min, x_max, y_max = self.grid.cell_bounds(cx, cy)
        
        # Find intersection points of ellipse with cell edges
        intersections: List[Tuple[float, float]] = []
        
        # Check each edge of the cell
        edges = [
            ((x_min, y_min), (x_max, y_min)),  # Bottom
            ((x_max, y_min), (x_max, y_max)),  # Right
            ((x_max, y_max), (x_min, y_max)),  # Top
            ((x_min, y_max), (x_min, y_min)),  # Left
        ]
        
        for (ex1, ey1), (ex2, ey2) in edges:
            pts = self._line_ellipse_intersections(ex1, ey1, ex2, ey2)
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
        
        # Order points counter-clockwise around the ellipse
        # Use parametric angle: theta = atan2(dy/b, dx/a)
        angle1 = math.atan2(
            (p1[1] - self.center_y) / self.semi_minor,
            (p1[0] - self.center_x) / self.semi_major
        )
        angle2 = math.atan2(
            (p2[1] - self.center_y) / self.semi_minor,
            (p2[0] - self.center_x) / self.semi_major
        )
        
        if angle1 > angle2:
            p1, p2 = p2, p1
        
        # Calculate outward normal at midpoint
        # For ellipse: (x-cx)²/a² + (y-cy)²/b² = 1
        # Gradient: (2(x-cx)/a², 2(y-cy)/b²) points outward
        mid_x = (p1[0] + p2[0]) / 2
        mid_y = (p1[1] + p2[1]) / 2
        normal_x = (mid_x - self.center_x) / (self.semi_major ** 2)
        normal_y = (mid_y - self.center_y) / (self.semi_minor ** 2)
        norm_len = math.sqrt(normal_x ** 2 + normal_y ** 2)
        if norm_len > 1e-9:
            normal_x /= norm_len
            normal_y /= norm_len
        
        return (p1[0], p1[1], p2[0], p2[1], normal_x, normal_y)

    def _line_ellipse_intersections(
        self, x1: float, y1: float, x2: float, y2: float
    ) -> List[Tuple[float, float]]:
        """
        Find intersection points of a line segment with the ellipse.

        The ellipse equation: (x-cx)²/a² + (y-cy)²/b² = 1
        Parameterize line: x = x1 + t*dx, y = y1 + t*dy

        Args:
            x1, y1: Start of line segment
            x2, y2: End of line segment

        Returns:
            List of intersection points
        """
        dx = x2 - x1
        dy = y2 - y1
        
        # Translate to ellipse center
        fx = x1 - self.center_x
        fy = y1 - self.center_y
        
        a2 = self.semi_major ** 2
        b2 = self.semi_minor ** 2
        
        # Substitute into ellipse equation and solve quadratic
        # (fx + t*dx)²/a² + (fy + t*dy)²/b² = 1
        # Expanding: A*t² + B*t + C = 0
        A = dx * dx / a2 + dy * dy / b2
        B = 2 * (fx * dx / a2 + fy * dy / b2)
        C = fx * fx / a2 + fy * fy / b2 - 1
        
        discriminant = B * B - 4 * A * C
        
        if discriminant < 0 or A < 1e-12:
            return []
        
        intersections = []
        sqrt_disc = math.sqrt(discriminant)
        
        t1 = (-B - sqrt_disc) / (2 * A)
        t2 = (-B + sqrt_disc) / (2 * A)
        
        for t in [t1, t2]:
            if -1e-9 <= t <= 1 + 1e-9:
                px = x1 + t * dx
                py = y1 + t * dy
                intersections.append((px, py))
        
        return intersections

    def generate(self) -> GeometryData:
        """Generate ellipse geometry data."""
        # Iterate over all cells
        for cy in range(self.grid.ny):
            for cx in range(self.grid.nx):
                cell_id = self.grid.cell_id(cx, cy)
                
                # Check if cell is fully inside
                if self._cell_fully_inside(cx, cy):
                    self.data.add_inside_cell(cell_id)
                    continue
                
                # Check if cell intersects the ellipse boundary
                if self._cell_intersects_ellipse(cx, cy):
                    segment_data = self._ellipse_segment_in_cell(cx, cy)
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
    parser = create_base_parser("Generate ellipse geometry for DSMC solver")
    parser.add_argument('--cx', type=float, required=True,
                        help='Ellipse center X coordinate')
    parser.add_argument('--cy', type=float, required=True,
                        help='Ellipse center Y coordinate')
    parser.add_argument('--a', type=float, required=True,
                        help='Semi-major axis (horizontal radius)')
    parser.add_argument('--b', type=float, required=True,
                        help='Semi-minor axis (vertical radius)')
    
    args = parser.parse_args()
    
    grid = grid_from_args(args)
    
    generator = EllipseGenerator(
        grid=grid,
        center_x=args.cx,
        center_y=args.cy,
        semi_major=args.a,
        semi_minor=args.b,
    )
    
    generator.generate()
    generator.save(
        args.output,
        description=f"Ellipse: center=({args.cx}, {args.cy}), a={args.a}, b={args.b}"
    )


if __name__ == '__main__':
    main()


