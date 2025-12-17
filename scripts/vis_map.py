#!/usr/bin/env python3
"""
Create animated GIFs of density and temperature heatmaps from DSMC visualization data.
Uses PyTorch for accelerated computation.
"""

import argparse
import glob
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LogNorm
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create animated GIFs of density and temperature heatmaps"
    )
    parser.add_argument(
        "-i", "--input",
        type=str,
        required=True,
        help="Input directory containing visualization cell data (*-cell.dat files)"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output directory for GIFs (default: same as input)"
    )
    parser.add_argument(
        "-c", "--config",
        type=str,
        default=None,
        help="Config YAML file to read grid dimensions"
    )
    parser.add_argument(
        "-g", "--geometry",
        type=str,
        default=None,
        help="Geometry file (.dat) to overlay solid objects (optional)"
    )
    parser.add_argument(
        "--downsample",
        type=int,
        default=1,
        help="Downsample factor for cells (default: 1, no downsampling)"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=10,
        help="Frames per second for GIF (default: 10)"
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=100,
        help="DPI for frames (default: 100)"
    )
    parser.add_argument(
        "--log-scale",
        action="store_true",
        help="Use logarithmic scale for density"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for computation (default: cuda if available, else cpu)"
    )
    return parser.parse_args()


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
    """Try to infer grid dimensions from total cell count."""
    sqrt_n = int(np.sqrt(num_cells))
    if sqrt_n * sqrt_n == num_cells:
        return sqrt_n, sqrt_n
    
    # Try common aspect ratios
    for ny in range(1, int(np.sqrt(num_cells)) + 1):
        if num_cells % ny == 0:
            nx = num_cells // ny
            if abs(nx - ny) <= max(nx, ny) // 2:
                return nx, ny
    
    return sqrt_n, sqrt_n


def find_cell_files(input_dir):
    """Find all cell data files and sort by timestep."""
    pattern = os.path.join(input_dir, "*-cell.dat")
    files = glob.glob(pattern)
    
    # Extract timestep from filename and sort
    def extract_timestep(filepath):
        basename = os.path.basename(filepath)
        match = re.match(r"(\d+)-cell\.dat", basename)
        if match:
            return int(match.group(1))
        return 0
    
    files.sort(key=extract_timestep)
    timesteps = [extract_timestep(f) for f in files]
    
    return files, timesteps


def load_all_cell_data_torch(files, device):
    """Load all cell data files into PyTorch tensors."""
    all_density = []
    all_temperature = []
    all_particle_count = []
    
    for filepath in files:
        data = np.loadtxt(filepath, comments="#")
        if data.ndim == 1:
            data = data.reshape(1, -1)
        
        # columns: id, particle_count, offset, density, temperature
        all_particle_count.append(data[:, 1])
        all_density.append(data[:, 3])
        all_temperature.append(data[:, 4])
    
    # Stack into tensors: shape (num_timesteps, num_cells)
    density_tensor = torch.tensor(np.stack(all_density), dtype=torch.float32, device=device)
    temperature_tensor = torch.tensor(np.stack(all_temperature), dtype=torch.float32, device=device)
    particle_count_tensor = torch.tensor(np.stack(all_particle_count), dtype=torch.float32, device=device)
    
    return density_tensor, temperature_tensor, particle_count_tensor


def reshape_to_grid_torch(data, nx, ny):
    """Reshape 1D cell data to 2D grid using PyTorch. 
    Cell ordering: cell_id = cy * nx + cx.
    Input shape: (num_timesteps, num_cells) or (num_cells,)
    Output shape: (num_timesteps, ny, nx) or (ny, nx)
    """
    if data.dim() == 1:
        return data.view(ny, nx)
    else:
        num_timesteps = data.shape[0]
        return data.view(num_timesteps, ny, nx)


def downsample_grid_torch(grid, factor):
    """Downsample 2D or 3D grid by averaging blocks.
    Uses average pooling for downsampling.
    Input shape: (num_timesteps, ny, nx) or (ny, nx)
    """
    if factor <= 1:
        return grid
    
    if grid.dim() == 2:
        # Add batch and channel dims for avg_pool2d
        grid = grid.unsqueeze(0).unsqueeze(0)
        pooled = torch.nn.functional.avg_pool2d(grid, kernel_size=factor, stride=factor)
        return pooled.squeeze(0).squeeze(0)
    else:
        # Shape: (num_timesteps, ny, nx) -> (num_timesteps, 1, ny, nx)
        grid = grid.unsqueeze(1)
        pooled = torch.nn.functional.avg_pool2d(grid, kernel_size=factor, stride=factor)
        return pooled.squeeze(1)


def apply_inside_mask_torch(grid, inside_cells, nx, ny, downsample_factor=1):
    """Set inside cells to NaN after downsampling."""
    if inside_cells is None or len(inside_cells) == 0:
        return grid
    
    # Create mask at original resolution
    mask = torch.zeros(ny, nx, dtype=torch.bool, device=grid.device)
    for cell_id in inside_cells:
        cx = cell_id % nx
        cy = cell_id // nx
        if cy < ny and cx < nx:
            mask[cy, cx] = True
    
    # Downsample mask (any True in block -> True)
    if downsample_factor > 1:
        mask_float = mask.float().unsqueeze(0).unsqueeze(0)
        mask_pooled = torch.nn.functional.max_pool2d(mask_float, kernel_size=downsample_factor, stride=downsample_factor)
        mask = mask_pooled.squeeze(0).squeeze(0) > 0.5
    
    # Apply mask
    if grid.dim() == 2:
        grid[mask] = float('nan')
    else:
        grid[:, mask] = float('nan')
    
    return grid


def compute_global_range(tensor, percentile_low=1, percentile_high=99):
    """Compute global value range across all timesteps, ignoring NaN."""
    valid = tensor[~torch.isnan(tensor)]
    if len(valid) == 0:
        return 0, 1
    
    # Move to CPU for percentile calculation
    valid_cpu = valid.cpu().numpy()
    vmin = np.percentile(valid_cpu, percentile_low)
    vmax = np.percentile(valid_cpu, percentile_high)
    
    return vmin, vmax


def create_frame(grid_data, timestep, title, cmap, vmin, vmax, extent, 
                 geometry=None, log_scale=False, dpi=100, lx=1.0, ly=1.0, 
                 downsample_factor=1):
    """Create a single frame as PIL Image."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Convert to numpy for plotting
    grid_np = grid_data.cpu().numpy()
    
    # Adjust extent for downsampled grid
    if downsample_factor > 1:
        # extent stays the same (physical coordinates), but grid resolution changes
        pass
    
    if log_scale:
        # Handle zeros/negatives for log scale
        grid_plot = np.where(grid_np > 0, grid_np, np.nan)
        im = ax.imshow(grid_plot, origin="lower", extent=extent,
                       cmap=cmap, norm=LogNorm(vmin=max(vmin, 1e-10), vmax=vmax), 
                       aspect="equal")
    else:
        im = ax.imshow(grid_np, origin="lower", extent=extent,
                       cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal")
    
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(f"{title} - Timestep {timestep}")
    
    # Draw geometry segments
    if geometry and geometry["segments"]:
        for seg in geometry["segments"]:
            ax.plot(
                [seg["start_x"], seg["end_x"]],
                [seg["start_y"], seg["end_y"]],
                color="white", linewidth=2, solid_capstyle="round"
            )
    
    plt.tight_layout()
    
    # Convert to PIL Image
    fig.canvas.draw()
    img = Image.frombuffer('RGB', fig.canvas.get_width_height(), 
                           fig.canvas.buffer_rgba(), 'raw', 'RGBA', 0, 1).convert('RGB')
    plt.close(fig)
    
    return img


def create_gif(grids, timesteps, output_path, title, cmap, vmin, vmax, extent,
               geometry=None, log_scale=False, fps=10, dpi=100, lx=1.0, ly=1.0,
               downsample_factor=1):
    """Create animated GIF from grid data."""
    frames = []
    
    print(f"Creating {title} GIF with {len(timesteps)} frames...")
    
    for i, (grid, ts) in enumerate(zip(grids, timesteps)):
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  Processing frame {i + 1}/{len(timesteps)}...")
        
        frame = create_frame(
            grid_data=grid,
            timestep=ts,
            title=title,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            extent=extent,
            geometry=geometry,
            log_scale=log_scale,
            dpi=dpi,
            lx=lx,
            ly=ly,
            downsample_factor=downsample_factor
        )
        frames.append(frame)
    
    # Save as GIF
    duration = int(1000 / fps)  # milliseconds per frame
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=0
    )
    print(f"Saved GIF: {output_path}")


def main():
    args = parse_args()
    
    # Setup device
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Find cell files
    print(f"Searching for cell files in {args.input}/visualization...")
    files, timesteps = find_cell_files(os.path.join(args.input, "visualization"))
    
    if not files:
        print(f"Error: No cell data files found in {args.input}")
        return
    
    print(f"Found {len(files)} cell data files")
    
    # Determine output directory
    output_dir = args.output if args.output else args.input
    os.makedirs(output_dir, exist_ok=True)
    
    # Load config
    config = None
    if args.config:
        config = load_config(args.config)
    
    # Load geometry
    geometry = load_segments(args.geometry)
    if geometry:
        print(f"Loaded {len(geometry['segments'])} segments, {len(geometry['inside_cells'])} inside cells")
    
    # Load all data into tensors
    print("Loading cell data...")
    density, temperature, particle_count = load_all_cell_data_torch(files, device)
    print(f"Loaded data shape: {density.shape}")
    
    # Determine grid dimensions
    num_cells = density.shape[1]
    if config:
        nx, ny = config["nx"], config["ny"]
        lx, ly = config["lx"], config["ly"]
    else:
        nx, ny = infer_grid_size(num_cells)
        lx, ly = float(nx), float(ny)
        print(f"Inferred grid size: {nx} x {ny}")
    
    # Reshape to grids
    print("Reshaping to grids...")
    density_grids = reshape_to_grid_torch(density, nx, ny)
    temperature_grids = reshape_to_grid_torch(temperature, nx, ny)
    
    # Downsample if requested
    if args.downsample > 1:
        print(f"Downsampling by factor {args.downsample}...")
        density_grids = downsample_grid_torch(density_grids, args.downsample)
        temperature_grids = downsample_grid_torch(temperature_grids, args.downsample)
        
        # Update grid dimensions for display
        nx_ds = nx // args.downsample
        ny_ds = ny // args.downsample
        print(f"Downsampled grid size: {nx_ds} x {ny_ds}")
    
    # Apply inside cell mask
    if geometry and len(geometry["inside_cells"]) > 0:
        print("Applying geometry mask...")
        density_grids = apply_inside_mask_torch(density_grids, geometry["inside_cells"], 
                                                 nx, ny, args.downsample)
        temperature_grids = apply_inside_mask_torch(temperature_grids, geometry["inside_cells"], 
                                                     nx, ny, args.downsample)
    
    # Compute global ranges for consistent coloring
    print("Computing value ranges...")
    density_vmin, density_vmax = compute_global_range(density_grids)
    temp_vmin, temp_vmax = compute_global_range(temperature_grids)
    
    print(f"Density range: {density_vmin:.2e} - {density_vmax:.2e}")
    print(f"Temperature range: {temp_vmin:.1f} - {temp_vmax:.1f}")
    
    # Extent for plotting (physical coordinates)
    extent = [0, lx, 0, ly]
    
    # Create density GIF
    density_output = os.path.join(output_dir, "density.gif")
    create_gif(
        grids=density_grids,
        timesteps=timesteps,
        output_path=density_output,
        title="Density",
        cmap="viridis",
        vmin=density_vmin,
        vmax=density_vmax,
        extent=extent,
        geometry=geometry,
        log_scale=args.log_scale,
        fps=args.fps,
        dpi=args.dpi,
        lx=lx,
        ly=ly,
        downsample_factor=args.downsample
    )
    
    # Create temperature GIF
    temperature_output = os.path.join(output_dir, "temperature.gif")
    create_gif(
        grids=temperature_grids,
        timesteps=timesteps,
        output_path=temperature_output,
        title="Temperature",
        cmap="hot",
        vmin=temp_vmin,
        vmax=temp_vmax,
        extent=extent,
        geometry=geometry,
        log_scale=False,  # Temperature typically doesn't need log scale
        fps=args.fps,
        dpi=args.dpi,
        lx=lx,
        ly=ly,
        downsample_factor=args.downsample
    )
    
    print("Done!")


if __name__ == "__main__":
    main()
