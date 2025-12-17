#!/usr/bin/env python3
"""
Generate particle emission schedules for DSMC simulations.

This script reads a source config file to get total_particles,
then generates an emission schedule based on the specified profile.

Usage:
    python source.py -c source_config.yaml -o schedule.dat --start 0 --end 5000 --profile uniform
    python source.py -c source_config.yaml -o schedule.dat --start 0 --end 1000 --profile ramp_up
    python source.py -c source_config.yaml -o schedule.dat --start 0 --end 1000 --profile burst --burst-timesteps 0,100,500
"""

import argparse
import yaml
import numpy as np
from pathlib import Path


def load_source_config(path: str) -> dict:
    """Load source config YAML and return total_particles."""
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def generate_uniform(total_particles: int, start: int, end: int) -> list[tuple[int, int]]:
    """
    Generate uniform emission schedule.
    Distributes particles evenly across all timesteps from start to end-1.
    """
    num_timesteps = end - start
    if num_timesteps <= 0:
        raise ValueError(f"Invalid range: start={start}, end={end}")
    
    # Calculate base particles per timestep and remainder
    base_count = total_particles // num_timesteps
    remainder = total_particles % num_timesteps
    
    schedule = []
    for i, ts in enumerate(range(start, end)):
        # Distribute remainder across first 'remainder' timesteps
        count = base_count + (1 if i < remainder else 0)
        if count > 0:
            schedule.append((ts, count))
    
    return schedule


def generate_ramp_up(total_particles: int, start: int, end: int) -> list[tuple[int, int]]:
    """
    Generate ramp-up emission schedule.
    Emission rate increases linearly from start to end.
    """
    num_timesteps = end - start
    if num_timesteps <= 0:
        raise ValueError(f"Invalid range: start={start}, end={end}")
    
    # Linear weights: 1, 2, 3, ..., num_timesteps
    weights = np.arange(1, num_timesteps + 1, dtype=float)
    weights /= weights.sum()
    
    # Distribute particles according to weights
    counts = np.round(weights * total_particles).astype(int)
    
    # Adjust for rounding errors
    diff = total_particles - counts.sum()
    if diff > 0:
        # Add to the last timesteps
        for i in range(diff):
            counts[-(i+1) % num_timesteps] += 1
    elif diff < 0:
        # Remove from the first timesteps with counts > 0
        for i in range(-diff):
            for j in range(num_timesteps):
                if counts[j] > 0:
                    counts[j] -= 1
                    break
    
    schedule = []
    for i, ts in enumerate(range(start, end)):
        if counts[i] > 0:
            schedule.append((ts, int(counts[i])))
    
    return schedule


def generate_ramp_down(total_particles: int, start: int, end: int) -> list[tuple[int, int]]:
    """
    Generate ramp-down emission schedule.
    Emission rate decreases linearly from start to end.
    """
    num_timesteps = end - start
    if num_timesteps <= 0:
        raise ValueError(f"Invalid range: start={start}, end={end}")
    
    # Linear weights: num_timesteps, num_timesteps-1, ..., 1
    weights = np.arange(num_timesteps, 0, -1, dtype=float)
    weights /= weights.sum()
    
    # Distribute particles according to weights
    counts = np.round(weights * total_particles).astype(int)
    
    # Adjust for rounding errors
    diff = total_particles - counts.sum()
    if diff > 0:
        for i in range(diff):
            counts[i % num_timesteps] += 1
    elif diff < 0:
        for i in range(-diff):
            for j in range(num_timesteps - 1, -1, -1):
                if counts[j] > 0:
                    counts[j] -= 1
                    break
    
    schedule = []
    for i, ts in enumerate(range(start, end)):
        if counts[i] > 0:
            schedule.append((ts, int(counts[i])))
    
    return schedule


def generate_burst(total_particles: int, start: int, end: int, 
                   burst_timesteps: list[int] = None) -> list[tuple[int, int]]:
    """
    Generate burst emission schedule.
    All particles are emitted at specific timesteps (bursts).
    """
    if burst_timesteps is None or len(burst_timesteps) == 0:
        # Default: single burst at start
        burst_timesteps = [start]
    
    # Filter to valid range
    burst_timesteps = [ts for ts in burst_timesteps if start <= ts < end]
    if len(burst_timesteps) == 0:
        raise ValueError(f"No burst timesteps in range [{start}, {end})")
    
    num_bursts = len(burst_timesteps)
    base_count = total_particles // num_bursts
    remainder = total_particles % num_bursts
    
    schedule = []
    for i, ts in enumerate(sorted(burst_timesteps)):
        count = base_count + (1 if i < remainder else 0)
        if count > 0:
            schedule.append((ts, count))
    
    return schedule


def generate_gaussian(total_particles: int, start: int, end: int,
                      center: float = None, sigma: float = None) -> list[tuple[int, int]]:
    """
    Generate Gaussian emission schedule.
    Emission follows a Gaussian profile centered at 'center' with std 'sigma'.
    """
    num_timesteps = end - start
    if num_timesteps <= 0:
        raise ValueError(f"Invalid range: start={start}, end={end}")
    
    if center is None:
        center = (start + end) / 2
    if sigma is None:
        sigma = num_timesteps / 6  # ~99.7% within range
    
    # Generate Gaussian weights
    timesteps = np.arange(start, end)
    weights = np.exp(-0.5 * ((timesteps - center) / sigma) ** 2)
    weights /= weights.sum()
    
    # Distribute particles
    counts = np.round(weights * total_particles).astype(int)
    
    # Adjust for rounding errors
    diff = total_particles - counts.sum()
    center_idx = int(center - start)
    center_idx = max(0, min(center_idx, num_timesteps - 1))
    
    if diff > 0:
        for i in range(diff):
            idx = (center_idx + i) % num_timesteps
            counts[idx] += 1
    elif diff < 0:
        for i in range(-diff):
            for j in range(num_timesteps):
                idx = (j) % num_timesteps
                if counts[idx] > 0:
                    counts[idx] -= 1
                    break
    
    schedule = []
    for i, ts in enumerate(range(start, end)):
        if counts[i] > 0:
            schedule.append((ts, int(counts[i])))
    
    return schedule


def generate_periodic(total_particles: int, start: int, end: int,
                      period: int = 10) -> list[tuple[int, int]]:
    """
    Generate periodic emission schedule.
    Emits particles at regular intervals (every 'period' timesteps).
    """
    num_timesteps = end - start
    if num_timesteps <= 0:
        raise ValueError(f"Invalid range: start={start}, end={end}")
    
    # Emission timesteps
    emit_timesteps = list(range(start, end, period))
    num_emissions = len(emit_timesteps)
    
    if num_emissions == 0:
        raise ValueError(f"No emission timesteps with period={period}")
    
    base_count = total_particles // num_emissions
    remainder = total_particles % num_emissions
    
    schedule = []
    for i, ts in enumerate(emit_timesteps):
        count = base_count + (1 if i < remainder else 0)
        if count > 0:
            schedule.append((ts, count))
    
    return schedule


def write_schedule(schedule: list[tuple[int, int]], output_path: str, 
                   total_particles: int, profile: str):
    """Write schedule to .dat file."""
    with open(output_path, 'w') as f:
        f.write(f"# Emission schedule generated by source.py\n")
        f.write(f"# Profile: {profile}\n")
        f.write(f"# Total particles: {total_particles}\n")
        f.write(f"# Emission events: {len(schedule)}\n")
        f.write(f"# Format: timestep count\n")
        
        for ts, count in schedule:
            f.write(f"{ts} {count}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Generate particle emission schedules for DSMC simulations.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available profiles:
  uniform    - Emit particles evenly across all timesteps
  ramp_up    - Linearly increasing emission rate
  ramp_down  - Linearly decreasing emission rate
  burst      - Emit all particles at specific timesteps (use --burst-timesteps)
  gaussian   - Gaussian distribution centered at midpoint (or --center)
  periodic   - Emit at regular intervals (use --period)

Examples:
  # Uniform emission from timestep 0 to 4999
  python source.py -c config.yaml -o schedule.dat --start 0 --end 5000 --profile uniform

  # Burst emission at timesteps 0, 1000, 2000
  python source.py -c config.yaml -o schedule.dat --start 0 --end 5000 --profile burst --burst-timesteps 0,1000,2000

  # Periodic emission every 100 timesteps
  python source.py -c config.yaml -o schedule.dat --start 0 --end 5000 --profile periodic --period 100
        """
    )
    
    parser.add_argument('-c', '--config', required=True,
                        help="Path to source config YAML file")
    parser.add_argument('-o', '--output', required=True,
                        help="Output schedule file path (.dat)")
    parser.add_argument('--start', type=int, default=0,
                        help="Start timestep (inclusive, default: 0)")
    parser.add_argument('--end', type=int, required=True,
                        help="End timestep (exclusive)")
    parser.add_argument('--profile', choices=['uniform', 'ramp_up', 'ramp_down', 
                                               'burst', 'gaussian', 'periodic'],
                        default='uniform',
                        help="Emission profile (default: uniform)")
    
    # Profile-specific options
    parser.add_argument('--burst-timesteps', type=str, default=None,
                        help="Comma-separated list of burst timesteps (for burst profile)")
    parser.add_argument('--center', type=float, default=None,
                        help="Center timestep for Gaussian profile")
    parser.add_argument('--sigma', type=float, default=None,
                        help="Standard deviation for Gaussian profile")
    parser.add_argument('--period', type=int, default=10,
                        help="Period for periodic profile (default: 10)")
    
    # Override total_particles
    parser.add_argument('--total', type=int, default=None,
                        help="Override total_particles from config")
    
    args = parser.parse_args()
    
    # Load config
    config = load_source_config(args.config)
    total_particles = args.total if args.total is not None else config.get('total_particles', 0)
    
    if total_particles <= 0:
        print(f"Error: total_particles must be > 0, got {total_particles}")
        return 1
    
    print(f"Source config: {args.config}")
    print(f"Total particles: {total_particles}")
    print(f"Timestep range: [{args.start}, {args.end})")
    print(f"Profile: {args.profile}")
    
    # Generate schedule
    if args.profile == 'uniform':
        schedule = generate_uniform(total_particles, args.start, args.end)
    elif args.profile == 'ramp_up':
        schedule = generate_ramp_up(total_particles, args.start, args.end)
    elif args.profile == 'ramp_down':
        schedule = generate_ramp_down(total_particles, args.start, args.end)
    elif args.profile == 'burst':
        burst_timesteps = None
        if args.burst_timesteps:
            burst_timesteps = [int(x.strip()) for x in args.burst_timesteps.split(',')]
        schedule = generate_burst(total_particles, args.start, args.end, burst_timesteps)
    elif args.profile == 'gaussian':
        schedule = generate_gaussian(total_particles, args.start, args.end,
                                     args.center, args.sigma)
    elif args.profile == 'periodic':
        schedule = generate_periodic(total_particles, args.start, args.end, args.period)
    else:
        print(f"Unknown profile: {args.profile}")
        return 1
    
    # Verify total
    schedule_sum = sum(count for _, count in schedule)
    if schedule_sum != total_particles:
        print(f"Warning: Schedule sum ({schedule_sum}) != total_particles ({total_particles})")
    
    # Write output
    write_schedule(schedule, args.output, total_particles, args.profile)
    print(f"Generated schedule with {len(schedule)} emission events")
    print(f"Output: {args.output}")
    
    return 0


if __name__ == '__main__':
    exit(main())
