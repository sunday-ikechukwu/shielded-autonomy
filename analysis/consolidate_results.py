#!/usr/bin/env python3
"""
consolidate_results.py
Reads raw CSV files from results/ folder and aggregates them into consolidated summaries
under the analysis/ directory. Does not modify any raw data.
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np

def find_results_dir():
    """Locate the results directory relative to the repository root."""
    cwd_results = Path("results")
    if cwd_results.is_dir():
        return cwd_results.resolve()
    
    script_results = Path(__file__).parent.parent / "results"
    if script_results.is_dir():
        return script_results.resolve()
    
    raise FileNotFoundError("Could not find the 'results/' directory. Please run the script from the repository root.")

def find_analysis_dir():
    """Locate or create the analysis directory relative to the repository root."""
    # Find repository root
    try:
        results_dir = find_results_dir()
        analysis_dir = results_dir.parent / "analysis"
    except FileNotFoundError:
        # Fallback to current working directory
        analysis_dir = Path("analysis")
        
    analysis_dir.mkdir(parents=True, exist_ok=True)
    return analysis_dir.resolve()

def filter_and_load_data(files):
    """
    Loads list of CSV files, applies strict inspectable filtering,
    and returns a concatenated DataFrame.
    """
    dfs = []
    for filepath in files:
        df = pd.read_csv(filepath)
        initial_len = len(df)
        
        # Check if rows are valid: must have policy, environment, scenario, run
        valid_mask = (
            df['policy'].notna() & 
            df['environment'].notna() & 
            df['scenario'].notna() & 
            df['run'].notna()
        )
        
        # Invalidate positive checks on run
        valid_mask &= (df['run'] > 0)
        
        excluded_df = df[~valid_mask]
        clean_df = df[valid_mask]
        
        if len(excluded_df) > 0:
            print(f"WARNING: Excluded {len(excluded_df)} row(s) from {filepath.name}:")
            for idx, row in excluded_df.iterrows():
                print(f"  Row {idx + 2}: {row.to_dict()} (Reason: Missing key identifiers or invalid run ID)")
                
        dfs.append(clean_df)
        
    if not dfs:
        return pd.DataFrame()
        
    return pd.concat(dfs, ignore_index=True)

def aggregate_experiment(df):
    """
    Aggregates dataframe by environment, policy, and scenario.
    Computes mean, std (sample), min, and max for each numeric metric.
    Computes successful_runs, failed_runs, and success_rate if 'result' column exists.
    """
    if df.empty:
        return pd.DataFrame()
        
    # Group by key dimensions
    group_cols = ['environment', 'policy', 'scenario']
    grouped = df.groupby(group_cols)
    
    # 1. Calculate count of runs
    n_runs = grouped.size().rename('n_runs')
    
    # 2. Success/failure stats (if result column is present)
    if 'result' in df.columns:
        # Check result value counts
        successful_runs = grouped['result'].apply(lambda s: (s == 'success').sum()).rename('successful_runs')
        failed_runs = grouped['result'].apply(lambda s: (s == 'aborted').sum()).rename('failed_runs')
        success_rate = (successful_runs / n_runs).rename('success_rate')
        
        summary_df = pd.concat([n_runs, successful_runs, failed_runs, success_rate], axis=1)
    else:
        summary_df = n_runs.to_frame()
        
    # 3. Numeric metrics
    # Get all numeric columns except 'run'
    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != 'run']
    
    for col in numeric_cols:
        summary_df[f"mean_{col}"] = grouped[col].mean()
        # pandas .std() uses ddof=1 (sample standard deviation) by default
        # and returns NaN when count < 2 (which is correct for n_runs = 1)
        summary_df[f"std_{col}"] = grouped[col].std()
        summary_df[f"min_{col}"] = grouped[col].min()
        summary_df[f"max_{col}"] = grouped[col].max()
        
    return summary_df.reset_index()

def main():
    try:
        results_dir = find_results_dir()
        analysis_dir = find_analysis_dir()
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
        
    print(f"Results Directory: {results_dir}")
    print(f"Analysis Directory: {analysis_dir}")
    print()
    
    # ------------------ Experiment A Consolidation ------------------
    exp_a_files = sorted(list((results_dir / "experiment_A").glob("*.csv")))
    print("Consolidating Experiment A...")
    print(f"Using raw files: {[f.name for f in exp_a_files]}")
    df_a = filter_and_load_data(exp_a_files)
    summary_a = aggregate_experiment(df_a)
    
    dest_a = analysis_dir / "experiment_A_summary.csv"
    summary_a.to_csv(dest_a, index=False)
    print(f"Saved Experiment A summary to: {dest_a.relative_to(analysis_dir.parent.parent)}")
    print(f"Columns: {list(summary_a.columns)}")
    print(f"Total summary rows: {len(summary_a)}")
    print()
    
    # ------------------ Experiment B Consolidation ------------------
    exp_b_files = sorted(list((results_dir / "experiment_B").glob("*.csv")))
    print("Consolidating Experiment B...")
    print(f"Using raw files: {[f.name for f in exp_b_files]}")
    df_b = filter_and_load_data(exp_b_files)
    summary_b = aggregate_experiment(df_b)
    
    dest_b = analysis_dir / "experiment_B_summary.csv"
    summary_b.to_csv(dest_b, index=False)
    print(f"Saved Experiment B summary to: {dest_b.relative_to(analysis_dir.parent.parent)}")
    print(f"Columns: {list(summary_b.columns)}")
    print(f"Total summary rows: {len(summary_b)}")
    print()
    
    # ------------------ Experiment C Consolidation ------------------
    exp_c_files = sorted(list((results_dir / "runtime_shared_autonomy").glob("*.csv")))
    print("Consolidating Experiment C...")
    print(f"Using raw files: {[f.name for f in exp_c_files]}")
    df_c = filter_and_load_data(exp_c_files)
    summary_c = aggregate_experiment(df_c)
    
    dest_c = analysis_dir / "experiment_C_summary.csv"
    summary_c.to_csv(dest_c, index=False)
    print(f"Saved Experiment C summary to: {dest_c.relative_to(analysis_dir.parent.parent)}")
    print(f"Columns: {list(summary_c.columns)}")
    print(f"Total summary rows: {len(summary_c)}")
    print()
    
    print("Consolidation pipeline complete.")

if __name__ == "__main__":
    main()
