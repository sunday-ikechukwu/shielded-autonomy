#!/usr/bin/env python3
"""
assess_dataset.py
Assesses a ROS2 imitation learning dataset (HDF5 files or DAgger recoveries)
for training behavior cloning models. Computes statistics, checks for data corruption,
counts potential preprocessing filters, and generates analysis plots.
"""

import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import argparse

def assess_dataset(data_dir, output_plot_path):
    print("=" * 60)
    print(f"ASSESSING DATASET IN: {data_dir}")
    print("=" * 60)

    if not os.path.exists(data_dir):
        print(f"Error: Directory {data_dir} does not exist!")
        return

    files = sorted([f for f in os.listdir(data_dir) if f.endswith('.hdf5') or f.endswith('.h5')])
    if not files:
        # Check if it's a DAgger recoveries file directly
        if os.path.isfile(data_dir) and (data_dir.endswith('.hdf5') or data_dir.endswith('.h5')):
            files = [os.path.basename(data_dir)]
            filepaths = [data_dir]
            is_single_file_db = True
            print("Detected single HDF5 database file (likely DAgger recoveries).")
        else:
            print("No HDF5 files found in the directory.")
            return
    else:
        filepaths = [os.path.join(data_dir, f) for f in files]
        is_single_file_db = False
        print(f"Found {len(files)} HDF5 files.")

    # Data collection buffers
    all_scans = []
    all_odom_vel = []
    all_goal_local = []
    all_actions = []
    
    # Episode metadata
    outcomes = {}
    total_timesteps = 0
    mismatched_files = 0
    nan_inf_count = 0
    
    # Preprocessing counts
    idle_count = 0
    spinning_count = 0
    reversing_count = 0

    # Load data
    if is_single_file_db:
        # Single database containing multiple group keys (recoveries)
        with h5py.File(filepaths[0], 'r') as f:
            keys = list(f.keys())
            print(f"Found {len(keys)} recovery chunks inside the database.")
            for key in keys:
                group = f[key]
                outcome = group.attrs.get('outcome', 'unknown')
                trigger_reason = group.attrs.get('trigger_reason', 'unknown')
                outcomes[outcome] = outcomes.get(outcome, 0) + 1
                
                scans = group['scans'][:]
                odom_vel = group['odom_vel'][:]
                # Some datasets might have 3D goal local (x, y, yaw), we only need 2D (dist, angle)
                goal_local = group['goal_local'][:, :2]
                actions = group['actions'][:]
                
                n = len(actions)
                total_timesteps += n
                
                # Check for NaNs/Infs
                if np.isnan(scans).any() or np.isinf(scans).any() or np.isnan(actions).any() or np.isinf(actions).any():
                    nan_inf_count += 1
                
                all_scans.append(scans)
                all_odom_vel.append(odom_vel)
                all_goal_local.append(goal_local)
                all_actions.append(actions)
    else:
        # Multiple episode files
        for filepath in filepaths:
            with h5py.File(filepath, 'r') as f:
                outcome = f.attrs.get('outcome', 'unknown')
                outcomes[outcome] = outcomes.get(outcome, 0) + 1
                
                if 'scans' not in f or 'actions' not in f or 'goal_local' not in f:
                    print(f"Warning: File {os.path.basename(filepath)} is missing expected datasets. Skipping.")
                    continue
                    
                scans = f['scans'][:]
                odom_vel = f['odom_vel'][:] if 'odom_vel' in f else np.zeros((len(scans), 2))
                goal_local = f['goal_local'][:, :2]
                actions = f['actions'][:]
                
                if len(scans) != len(actions):
                    mismatched_files += 1
                    continue
                
                n = len(actions)
                total_timesteps += n
                
                # Check for NaNs/Infs
                if np.isnan(scans).any() or np.isinf(scans).any() or np.isnan(actions).any() or np.isinf(actions).any():
                    nan_inf_count += 1
                
                all_scans.append(scans)
                all_odom_vel.append(odom_vel)
                all_goal_local.append(goal_local)
                all_actions.append(actions)

    if total_timesteps == 0:
        print("No valid timesteps loaded. Exiting.")
        return

    # Concatenate all steps into large arrays
    scans_flat = np.concatenate(all_scans, axis=0)
    odom_flat = np.concatenate(all_odom_vel, axis=0)
    goal_flat = np.concatenate(all_goal_local, axis=0)
    actions_flat = np.concatenate(all_actions, axis=0)

    # Preprocessing counts analysis
    goal_dist = goal_flat[:, 0]
    linear_cmd = actions_flat[:, 0]
    angular_cmd = actions_flat[:, 1]
    
    # 1. Idle/Stationary check: velocities very close to 0 while far from goal
    idle_mask = (np.abs(linear_cmd) < 0.005) & (np.abs(angular_cmd) < 0.005) & (goal_dist > 0.15)
    idle_count = np.sum(idle_mask)
    
    # 2. Spinning in place: linear velocity is low but angular is high
    spinning_mask = (np.abs(linear_cmd) < 0.01) & (np.abs(angular_cmd) > 0.3)
    spinning_count = np.sum(spinning_mask)
    
    # 3. Reversing check: robot is backing up
    reversing_mask = linear_cmd < -0.01
    reversing_count = np.sum(reversing_mask)

    # Print summary metrics
    print("\n--- EPISODE METADATA ---")
    for outcome, count in outcomes.items():
        print(f"  Outcome '{outcome}': {count} episodes")
    if mismatched_files > 0:
        print(f"  WARNING: {mismatched_files} files skipped due to step mismatches.")
    if nan_inf_count > 0:
        print(f"  WARNING: {nan_inf_count} files contain NaN or Inf values!")

    print("\n--- TIMESTEP STATISTICS ---")
    print(f"  Total Timesteps: {total_timesteps}")
    print(f"  Min LiDAR Range: {scans_flat.min():.4f} m (Warning threshold: 0.35m, Panic: 0.22m)")
    print(f"  Mean Goal Dist:  {goal_dist.mean():.4f} m  (Std: {goal_dist.std():.4f} m)")
    print(f"  Max Goal Dist:   {goal_dist.max():.4f} m  (Min: {goal_dist.min():.4f} m)")
    
    print("\n--- ACTION LIMITS & DISTRIBUTIONS ---")
    print(f"  Linear Speed (vx)  | Mean: {linear_cmd.mean():.4f} | Std: {linear_cmd.std():.4f} | Min: {linear_cmd.min():.4f} | Max: {linear_cmd.max():.4f}")
    print(f"  Angular Speed (wz) | Mean: {angular_cmd.mean():.4f} | Std: {angular_cmd.std():.4f} | Min: {angular_cmd.min():.4f} | Max: {angular_cmd.max():.4f}")

    print("\n--- PREPROCESSING DIAGNOSTICS ---")
    print(f"  Idle/Stationary Steps: {idle_count:6d} ({idle_count/total_timesteps*100:5.2f}%)  [Rec: filter out if v & w ~ 0 and dist > 0.15m]")
    print(f"  Spinning-in-Place:     {spinning_count:6d} ({spinning_count/total_timesteps*100:5.2f}%)  [Rec: DO NOT purge in novice training]")
    print(f"  Reversing Steps:       {reversing_count:6d} ({reversing_count/total_timesteps*100:5.2f}%)  [Rec: verify if MPPI actually needed to back up]")

    # Plotting
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Dataset Assessment: {os.path.basename(data_dir)}", fontsize=15, fontweight='bold')

    # Plot 1: Goal Distance Distribution
    axes[0, 0].hist(goal_dist, bins=50, color='#2563EB', edgecolor='black', alpha=0.7)
    axes[0, 0].axvline(1.5, color='#DC2626', linestyle='--', linewidth=2, label='Carrot Lookahead (1.5m)')
    axes[0, 0].set_title('Goal Distance Distribution')
    axes[0, 0].set_xlabel('Relative Goal Distance (meters)')
    axes[0, 0].set_ylabel('Timesteps Count')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    # Add status note
    if goal_dist.std() < 0.05:
        axes[0, 0].text(0.1, 0.5, "WARNING: Stuck Goal Detected!\nAll distances cluster at 1.5m.\nCheck plan_callback logic.", 
                        color='red', transform=axes[0, 0].transAxes, fontsize=10, bbox=dict(facecolor='white', alpha=0.8, edgecolor='red'))
    else:
        axes[0, 0].text(0.1, 0.5, "Goal Distance is varying.\nThis is correct!", 
                        color='green', transform=axes[0, 0].transAxes, fontsize=10, bbox=dict(facecolor='white', alpha=0.8, edgecolor='green'))

    # Plot 2: Minimum LiDAR Scan Values (Inflation Breach indicator)
    min_scans = scans_flat.min(axis=1) if len(scans_flat.shape) > 1 else scans_flat
    axes[0, 1].hist(min_scans, bins=50, color='#10B981', edgecolor='black', alpha=0.7)
    axes[0, 1].axvline(0.22, color='red', linestyle='-', linewidth=1.5, label='Panic Collision (0.22m)')
    axes[0, 1].axvline(0.35, color='orange', linestyle='-', linewidth=1.5, label='Inflation Warning (0.35m)')
    axes[0, 1].set_title('Minimum LiDAR Distance Distribution')
    axes[0, 1].set_xlabel('LiDAR Reading (meters)')
    axes[0, 1].set_ylabel('Timesteps Count')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()

    # Plot 3: 2D Action distribution
    hb = axes[1, 0].hexbin(linear_cmd, angular_cmd, gridsize=30, cmap='YlOrRd', mincnt=1)
    fig.colorbar(hb, ax=axes[1, 0], label='Timestep Count')
    axes[1, 0].set_title('Action command distribution (vx vs wz)')
    axes[1, 0].set_xlabel('Linear Velocity (vx) [m/s]')
    axes[1, 0].set_ylabel('Angular Velocity (wz) [rad/s]')
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: Action distributions (histograms)
    axes[1, 1].hist(linear_cmd, bins=40, color='#F59E0B', alpha=0.6, label='Linear (vx)', edgecolor='black')
    axes[1, 1].hist(angular_cmd, bins=40, color='#8B5CF6', alpha=0.6, label='Angular (wz)', edgecolor='black')
    axes[1, 1].set_title('Action Histograms')
    axes[1, 1].set_xlabel('Command Value')
    axes[1, 1].set_ylabel('Timesteps Count')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig(output_plot_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved analysis plots to: {output_plot_path}")
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Assess dataset and preprocessing checklist.')
    parser.add_argument('--dir', type=str, default='imitation_data/MPPI_imitation_data',
                        help='Path to dataset directory or recoveries file')
    parser.add_argument('--output', type=str, default='dataset_analysis.png',
                        help='Filename to save output analysis plot')
    args = parser.parse_args()

    assess_dataset(args.dir, args.output)

if __name__ == '__main__':
    main()
