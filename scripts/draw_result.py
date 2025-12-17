#!/usr/bin/env python3
"""
Draw cell statistics heatmaps from DSMC simulation output.
Plots density, temperature, and particle count as 2D heatmaps.
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle


def parse_args():
    parser = argparse.ArgumentParser(
        description="Draw cell statistics heatmaps from DSMC simulation output"
    )
    parser.add_argument(
        "-i", "--input",
        type=str,
        required=True,
        help="Input directory containing cell.dat (or path to cell.dat file directly)"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output image filename (default: <input_dir>/heatmaps.png)"
    )
    parser.add_argument(
        "-c", "--config",
        type=str,
        default=None,
        help="Config YAML file to read grid dimensions (required if not inferrable)"
    )
    parser.add_argument(
        "-g", "--geometry",
        type=str,
        default=None,
        help="Geometry file (.dat) to overlay solid objects (optional)"
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="DPI for output image (default: 150)"
    )
    parser.add_argument(
        "--log-scale",
        action="store_true",
        help="Use logarithmic scale for density"
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the plot interactively"
    )
    return parser.parse_args()


def load_cell_data(filepath):
    """Load cell data from a dump file."""
    data = np.loadtxt(filepath, comments="#")
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return {
        "id": data[:, 0].astype(int),
        "particle_count": data[:, 1].astype(int),
        "offset": data[:, 2].astype(int),
        "density": data[:, 3],
        "temperature": data[:, 4],
    }


def load_config(config_path):
    """Load grid dimensions from config YAML."""
    try:
        import yaml
    except ImportError:
        print("Error: PyYAML not installed. Install with: pip install pyyaml")
        return None
    
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        return {
            "nx": config["grid"]["nx"],
            "ny": config["grid"]["ny"],
            "lx": config["grid"]["lx"],
            "ly": config["grid"]["ly"],
        }
    except Exception as e:
        print(f"Error loading config: {e}")
        return None


def load_segments(geometry_path):
    """Load segment data from geometry .dat file."""
    if geometry_path is None or not os.path.exists(geometry_path):
        return None
    
    try:
        segments = []
        inside_cells = []
        with open(geometry_path, "r") as f:
            header_read = False
            
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                
                parts = line.split()
                
                # First non-comment line is header: nx ny lx ly
                if not header_read:
                    header_read = True
                    continue
                
                if len(parts) < 2:
                    continue
                
                cell_id = int(parts[0])
                record_type = int(parts[1])
                
                if record_type == 0 and len(parts) >= 8:
                    segments.append({
                        "cell_id": cell_id,
                        "start_x": float(parts[2]),
                        "start_y": float(parts[3]),
                        "end_x": float(parts[4]),
                        "end_y": float(parts[5]),
                        "normal_x": float(parts[6]),
                        "normal_y": float(parts[7]),
                    })
                elif record_type == 1:
                    inside_cells.append(cell_id)
        
        return {
            "segments": segments,
            "inside_cells": np.array(inside_cells) if inside_cells else np.array([], dtype=int),
        }
    except Exception as e:
        print(f"Warning: Could not load geometry file: {e}")
        return None


def infer_grid_size(num_cells):
    """Try to infer grid dimensions from total cell count (assumes square grid)."""
    sqrt_n = int(np.sqrt(num_cells))
    if sqrt_n * sqrt_n == num_cells:
        return sqrt_n, sqrt_n
    
    # Try common aspect ratios
    for ny in range(1, int(np.sqrt(num_cells)) + 1):
        if num_cells % ny == 0:
            nx = num_cells // ny
            if abs(nx - ny) <= max(nx, ny) // 2:  # Reasonable aspect ratio
                return nx, ny
    
    return sqrt_n, sqrt_n  # Fallback


def reshape_to_grid(data, nx, ny):
    """Reshape 1D cell data to 2D grid. Cell ordering: cell_id = cy * nx + cx."""
    grid = np.zeros((ny, nx))
    for i, val in enumerate(data):
        cx = i % nx
        cy = i // nx
        if cy < ny:
            grid[cy, cx] = val
    return grid


def draw_heatmaps(cell_data, config, geometry=None, output_file=None, dpi=150, 
                  log_scale=False, show=False):
    """Draw heatmaps for density, temperature, and particle count."""
    
    # Get grid dimensions
    num_cells = len(cell_data["id"])
    if config:
        nx, ny = config["nx"], config["ny"]
        lx, ly = config["lx"], config["ly"]
    else:
        nx, ny = infer_grid_size(num_cells)
        lx, ly = nx, ny  # Use cell units if no config
        print(f"Inferred grid size: {nx} x {ny}")
    
    # Reshape data to 2D grids
    density_grid = reshape_to_grid(cell_data["density"], nx, ny)
    temperature_grid = reshape_to_grid(cell_data["temperature"], nx, ny)
    particle_count_grid = reshape_to_grid(cell_data["particle_count"], nx, ny)
    
    # Handle inside cells (solid objects) - set to NaN for visualization
    if geometry and len(geometry["inside_cells"]) > 0:
        for cell_id in geometry["inside_cells"]:
            cx = cell_id % nx
            cy = cell_id // nx
            if cy < ny:
                density_grid[cy, cx] = np.nan
                temperature_grid[cy, cx] = np.nan
                particle_count_grid[cy, cx] = np.nan
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Extent for proper axis labeling
    extent = [0, lx, 0, ly]
    
    # 1. Density heatmap
    ax1 = axes[0]
    if log_scale and np.nanmax(density_grid) > 0:
        # Use log scale, handle zeros
        density_plot = np.where(density_grid > 0, density_grid, np.nan)
        im1 = ax1.imshow(density_plot, origin="lower", extent=extent, 
                         cmap="viridis", norm=LogNorm(), aspect="equal")
    else:
        im1 = ax1.imshow(density_grid, origin="lower", extent=extent, 
                         cmap="viridis", aspect="equal")
    cbar1 = fig.colorbar(im1, ax=ax1, shrink=0.8)
    cbar1.set_label("Number Density (m⁻³)")
    ax1.set_xlabel("X (m)")
    ax1.set_ylabel("Y (m)")
    ax1.set_title("Density")
    
    # 2. Temperature heatmap
    ax2 = axes[1]
    im2 = ax2.imshow(temperature_grid, origin="lower", extent=extent, 
                     cmap="hot", aspect="equal")
    cbar2 = fig.colorbar(im2, ax=ax2, shrink=0.8)
    cbar2.set_label("Temperature (K)")
    ax2.set_xlabel("X (m)")
    ax2.set_ylabel("Y (m)")
    ax2.set_title("Temperature")
    
    # 3. Particle count heatmap
    ax3 = axes[2]
    im3 = ax3.imshow(particle_count_grid, origin="lower", extent=extent, 
                     cmap="Blues", aspect="equal")
    cbar3 = fig.colorbar(im3, ax=ax3, shrink=0.8)
    cbar3.set_label("Particle Count")
    ax3.set_xlabel("X (m)")
    ax3.set_ylabel("Y (m)")
    ax3.set_title("Particles per Cell")
    
    # Draw geometry segments on all plots
    if geometry and geometry["segments"]:
        for seg in geometry["segments"]:
            for ax in axes:
                ax.plot(
                    [seg["start_x"], seg["end_x"]],
                    [seg["start_y"], seg["end_y"]],
                    color="white", linewidth=2, solid_capstyle="round"
                )
    
    # Add statistics text
    valid_density = density_grid[~np.isnan(density_grid)]
    valid_temp = temperature_grid[~np.isnan(temperature_grid)]
    valid_count = particle_count_grid[~np.isnan(particle_count_grid)]
    
    stats_text = (
        f"Density: min={np.min(valid_density):.2e}, max={np.max(valid_density):.2e}, mean={np.mean(valid_density):.2e}\n"
        f"Temperature: min={np.min(valid_temp):.1f}K, max={np.max(valid_temp):.1f}K, mean={np.mean(valid_temp):.1f}K\n"
        f"Particles: total={int(np.sum(valid_count))}, per cell: min={int(np.min(valid_count))}, max={int(np.max(valid_count))}, mean={np.mean(valid_count):.1f}"
    )
    fig.text(0.5, 0.02, stats_text, ha="center", fontsize=9, family="monospace",
             bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    
    # Save or show
    if output_file:
        plt.savefig(output_file, dpi=dpi, bbox_inches="tight")
        print(f"Saved heatmaps to {output_file}")
    
    if show:
        plt.show()
    else:
        plt.close(fig)


def main():
    args = parse_args()
    
    # Determine cell data file path
    if os.path.isfile(args.input):
        cell_file = args.input
        input_dir = os.path.dirname(args.input) or "."
    else:
        cell_file = os.path.join(args.input, "cell.dat")
        input_dir = args.input
        if not os.path.exists(cell_file):
            print(f"Error: Could not find cell.dat in {args.input}")
            return
    
    # Determine output file
    if args.output:
        output_file = args.output
    else:
        output_file = os.path.join(input_dir, "heatmaps.png")
    
    # Load data
    print(f"Loading cell data from {cell_file}...")
    cell_data = load_cell_data(cell_file)
    print(f"Loaded {len(cell_data['id'])} cells")
    
    # Load config if provided
    config = None
    if args.config:
        config = load_config(args.config)
    
    # Load geometry if provided
    geometry = None
    if args.geometry:
        geometry = load_segments(args.geometry)
        if geometry:
            print(f"Loaded {len(geometry['segments'])} segments, {len(geometry['inside_cells'])} inside cells")
    
    # Draw heatmaps
    draw_heatmaps(
        cell_data=cell_data,
        config=config,
        geometry=geometry,
        output_file=output_file,
        dpi=args.dpi,
        log_scale=args.log_scale,
        show=args.show,
    )


if __name__ == "__main__":
    main()
