#!/usr/bin/env python3
"""
Visualization script for DSMC simulation output.
Reads particle and cell data dumps, creates plots, and generates GIF animations.
"""

import argparse
import glob
import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize DSMC simulation output and create GIF animation"
    )
    parser.add_argument(
        "-i", "--input",
        type=str,
        default="outputs",
        help="Input directory containing dump files (default: outputs)"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="simulation.gif",
        help="Output GIF filename (default: simulation.gif)"
    )
    parser.add_argument(
        "-c", "--config",
        type=str,
        default=None,
        help="Config YAML file to read domain dimensions (optional)"
    )
    parser.add_argument(
        "-g", "--geometry",
        type=str,
        default=None,
        help="Geometry file (.dat) containing solid object segments (optional)"
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
        help="DPI for output images (default: 100)"
    )
    parser.add_argument(
        "--show-grid",
        action="store_true",
        help="Show cell grid lines"
    )
    parser.add_argument(
        "--show-velocity",
        action="store_true",
        help="Show velocity vectors (quiver plot)"
    )
    parser.add_argument(
        "--color-by",
        choices=["cell", "speed", "species"],
        default="speed",
        help="Color particles by: cell, speed, or species (default: speed)"
    )
    return parser.parse_args()


def load_particle_data(filepath):
    """Load particle data from a dump file."""
    data = np.loadtxt(filepath, comments="#")
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return {
        "id": data[:, 0].astype(int),
        "pos_x": data[:, 1],
        "pos_y": data[:, 2],
        "vel_x": data[:, 3],
        "vel_y": data[:, 4],
        "vel_z": data[:, 5],
        "species": data[:, 6].astype(int),
        "cell_id": data[:, 7].astype(int),
    }


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


def load_segments(geometry_path):
    """Load segment data from geometry .dat file."""
    if geometry_path is None or not os.path.exists(geometry_path):
        return None
    
    try:
        segments = []
        with open(geometry_path, "r") as f:
            # Skip header line (nx ny lx ly)
            header = f.readline()
            
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) >= 7:
                    segments.append({
                        "cell_id": int(parts[0]),
                        "start_x": float(parts[1]),
                        "start_y": float(parts[2]),
                        "end_x": float(parts[3]),
                        "end_y": float(parts[4]),
                        "normal_x": float(parts[5]),
                        "normal_y": float(parts[6]),
                    })
        
        if not segments:
            return None
        
        # Convert to numpy arrays
        return {
            "cell_id": np.array([s["cell_id"] for s in segments]),
            "start_x": np.array([s["start_x"] for s in segments]),
            "start_y": np.array([s["start_y"] for s in segments]),
            "end_x": np.array([s["end_x"] for s in segments]),
            "end_y": np.array([s["end_y"] for s in segments]),
            "normal_x": np.array([s["normal_x"] for s in segments]),
            "normal_y": np.array([s["normal_y"] for s in segments]),
        }
    except Exception as e:
        print(f"Warning: Could not load geometry file: {e}")
        return None


def find_timesteps(input_dir):
    """Find all timesteps in the input directory."""
    pattern = os.path.join(input_dir, "*-particle.dat")
    files = glob.glob(pattern)
    
    timesteps = []
    for f in files:
        basename = os.path.basename(f)
        match = re.match(r"(\d+)-particle\.dat", basename)
        if match:
            timesteps.append(int(match.group(1)))
    
    return sorted(timesteps)


def load_config(config_path):
    """Load domain dimensions from config YAML."""
    try:
        import yaml
    except ImportError:
        print("Warning: PyYAML not installed, cannot read config file")
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
        print(f"Warning: Could not load config: {e}")
        return None


def infer_domain(particles_list):
    """Infer domain dimensions from particle positions."""
    all_x = np.concatenate([p["pos_x"] for p in particles_list])
    all_y = np.concatenate([p["pos_y"] for p in particles_list])
    
    margin = 0.05
    x_range = all_x.max() - all_x.min()
    y_range = all_y.max() - all_y.min()
    
    return {
        "lx": all_x.max() + margin * x_range,
        "ly": all_y.max() + margin * y_range,
        "nx": None,
        "ny": None,
    }


def create_animation(input_dir, output_file, config=None, geometry=None, fps=10, dpi=100,
                     show_grid=False, show_velocity=False, color_by="speed"):
    """Create GIF animation from dump files."""
    
    # Find all timesteps
    timesteps = find_timesteps(input_dir)
    if not timesteps:
        print(f"Error: No dump files found in {input_dir}")
        return
    
    print(f"Found {len(timesteps)} timesteps: {timesteps[0]} to {timesteps[-1]}")
    
    # Load all data
    particles_list = []
    cells_list = []
    for ts in timesteps:
        particle_file = os.path.join(input_dir, f"{ts:08d}-particle.dat")
        cell_file = os.path.join(input_dir, f"{ts:08d}-cell.dat")
        
        particles_list.append(load_particle_data(particle_file))
        cells_list.append(load_cell_data(cell_file))
    
    # Get domain dimensions
    if config:
        domain = load_config(config)
    else:
        domain = infer_domain(particles_list)
    
    if domain is None:
        domain = infer_domain(particles_list)
    
    lx, ly = domain["lx"], domain["ly"]
    nx, ny = domain.get("nx"), domain.get("ny")
    
    print(f"Domain: {lx} x {ly}")
    if nx and ny:
        print(f"Grid: {nx} x {ny} cells")
    
    # Calculate speed range for consistent coloring
    all_speeds = []
    for p in particles_list:
        speeds = np.sqrt(p["vel_x"]**2 + p["vel_y"]**2 + p["vel_z"]**2)
        all_speeds.extend(speeds)
    speed_min, speed_max = min(all_speeds), max(all_speeds)
    
    # Load segments (static geometry)
    segments = load_segments(geometry)
    if segments is not None:
        print(f"Loaded {len(segments['cell_id'])} segments from geometry file")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 8))
    
    def update(frame):
        ax.clear()
        
        particles = particles_list[frame]
        cells = cells_list[frame]
        ts = timesteps[frame]
        
        # Draw grid if requested
        if show_grid and nx and ny:
            dx, dy = lx / nx, ly / ny
            for i in range(nx + 1):
                ax.axvline(i * dx, color="gray", linewidth=0.5, alpha=0.5)
            for j in range(ny + 1):
                ax.axhline(j * dy, color="gray", linewidth=0.5, alpha=0.5)
        
        # Draw segments (solid objects)
        if segments is not None:
            for i in range(len(segments["cell_id"])):
                ax.plot(
                    [segments["start_x"][i], segments["end_x"][i]],
                    [segments["start_y"][i], segments["end_y"][i]],
                    color="black", linewidth=2, solid_capstyle="round"
                )
        
        # Determine colors
        if color_by == "speed":
            speeds = np.sqrt(particles["vel_x"]**2 + particles["vel_y"]**2 + particles["vel_z"]**2)
            colors = speeds
            cmap = "coolwarm"
            vmin, vmax = speed_min, speed_max
        elif color_by == "cell":
            colors = particles["cell_id"]
            cmap = "tab20"
            vmin, vmax = 0, max(1, cells["id"].max())
        else:  # species
            colors = particles["species"]
            cmap = "Set1"
            vmin, vmax = 0, max(1, particles["species"].max())
        
        # Plot particles
        scatter = ax.scatter(
            particles["pos_x"], particles["pos_y"],
            c=colors, cmap=cmap, vmin=vmin, vmax=vmax,
            s=20, alpha=0.7, edgecolors="none"
        )
        
        # Draw velocity vectors if requested
        if show_velocity:
            # Subsample for clarity
            n = len(particles["pos_x"])
            step = max(1, n // 50)
            ax.quiver(
                particles["pos_x"][::step], particles["pos_y"][::step],
                particles["vel_x"][::step], particles["vel_y"][::step],
                color="black", alpha=0.5, scale=5000, width=0.003
            )
        
        # Set limits and labels
        ax.set_xlim(0, lx)
        ax.set_ylim(0, ly)
        ax.set_aspect("equal")
        ax.set_xlabel("X position (m)")
        ax.set_ylabel("Y position (m)")
        ax.set_title(f"DSMC Simulation - Timestep {ts}\n({len(particles['id'])} particles)")
        
        return [scatter]
    
    # Create animation
    print("Creating animation...")
    anim = FuncAnimation(fig, update, frames=len(timesteps), blit=False)
    
    # Save as GIF
    print(f"Saving to {output_file}...")
    writer = PillowWriter(fps=fps)
    anim.save(output_file, writer=writer, dpi=dpi)
    
    plt.close(fig)
    print(f"Done! Saved animation to {output_file}")


def main():
    args = parse_args()
    
    create_animation(
        input_dir=args.input,
        output_file=args.output,
        config=args.config,
        geometry=args.geometry,
        fps=args.fps,
        dpi=args.dpi,
        show_grid=args.show_grid,
        show_velocity=args.show_velocity,
        color_by=args.color_by,
    )


if __name__ == "__main__":
    main()
