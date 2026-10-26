#!/usr/bin/env python3
"""
validate_results.py
Performs integrity checks and sanity validation on raw CSV result files 
for Experiment A, Experiment B, and Experiment C (Runtime Shared Autonomy).
Does not modify any raw data.
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np

# Define expected labels
EXPECTED_POLICIES = {
    'A': {'bc', 'mppi', 'safe_dagger', 'safe_dagger_v1'},
    'B': {'bc_model', 'safe_dagger'},
    'C': {'shared_autonomy'}
}

EXPECTED_SCENARIOS = {
    'known': {
        'A1_short_distance', 'A2_long_corridor', 'A3_goal_between_obstacles', 
        'A3_goal_betweem_obstacles', 'A4_goal_behind_robot', 'A5_left_turn_to_goal', 
        'A5_left_turn', 'A6_right_turn_to_goal', 'A6_right_turn', 'A7_thin_u_turn', 
        'A8_wide_u_turn'
    },
    'unknown': {
        'B1_west_corridor_straight_line_run', 'B2_T_junction_left_turn', 
        'B3_T_junction_right_turn', 'B4_blind_corner_east_turn', 
        'B5_room_to_room_transition', 'B6_couch_s_curve_avoidance', 
        'B7_bottleneck_traversal', 'B8_long_horizon_generalization'
    }
}

EXPECTED_ENVIRONMENTS = {
    'A': {'known_environment'},
    'B': {'unknown_environment'},
    'C': {'known_environment', 'unknown_environment'}
}

# Metrics to check for non-negativity
NON_NEGATIVE_METRICS = [
    'navigation_time', 'planned_path_length', 'actual_path_length', 
    'path_efficiency', 'collision_count', 'near_miss_count', 
    'minimum_obstacle_distance', 'average_linear_velocity', 
    'linear_velocity_smoothness', 'angular_velocity_smoothness'
]

RUNTIME_NON_NEGATIVE_METRICS = [
    'intervention_count', 'stable_recovery_count', 'navigation_time', 
    'expert_control_time', 'average_intervention_duration', 
    'maximum_intervention_duration', 'distance_under_expert_control', 
    'expert_distance_percentage', 'expert_time_percentage', 
    'autonomy_time_percentage', 'autonomy_distance_percentage', 
    'mean_time_between_interventions', 'average_takeover_obstacle_distance', 
    'minimum_takeover_obstacle_distance', 'maximum_takeover_obstacle_distance'
]

def find_results_dir():
    """Locate the results directory relative to the repository root."""
    # Option 1: check current working directory
    cwd_results = Path("results")
    if cwd_results.is_dir():
        return cwd_results.resolve()
    
    # Option 2: check parent of this script
    script_results = Path(__file__).parent.parent / "results"
    if script_results.is_dir():
        return script_results.resolve()
    
    raise FileNotFoundError("Could not find the 'results/' directory. Please run the script from the repository root.")

def validate_file_general(file_path):
    """General checks applicable to any CSV file."""
    issues = []
    errors = 0
    warnings = 0
    
    # Try reading the file
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        issues.append({
            'type': 'ERROR',
            'row': 'N/A',
            'values': str(e),
            'reason': f"Failed to read CSV file: {e}"
        })
        return None, issues, 1, 0
    
    num_rows = len(df)
    columns = list(df.columns)
    
    # Check for empty/missing values
    nan_mask = df.isna()
    if nan_mask.any().any():
        nan_indices = np.argwhere(nan_mask.values)
        for row_idx, col_idx in nan_indices:
            col_name = columns[col_idx]
            val = df.iloc[row_idx, col_idx]
            issues.append({
                'type': 'WARNING',
                'row': f"Index {row_idx} (Line {row_idx + 2})",
                'values': f"{col_name} = {val}",
                'reason': f"Missing value in column '{col_name}'"
            })
            warnings += 1

    # Check for duplicate rows
    dup_mask = df.duplicated(keep=False)
    if dup_mask.any():
        dup_indices = df[dup_mask].index
        for idx in dup_indices:
            row_vals = df.iloc[idx].to_dict()
            issues.append({
                'type': 'WARNING',
                'row': f"Index {idx} (Line {idx + 2})",
                'values': str(row_vals),
                'reason': "Duplicate row found in CSV"
            })
            warnings += 1
            
    return df, issues, errors, warnings

def validate_experiment_a_b(df, file_path, exp_type):
    """Validation checks specifically for Experiment A and Experiment B."""
    issues = []
    errors = 0
    warnings = 0
    
    if df is None:
        return issues, errors, warnings
    
    # Verify expected columns are present
    required_cols = ['policy', 'environment', 'scenario', 'run', 'result', 'navigation_time']
    for col in required_cols:
        if col not in df.columns:
            issues.append({
                'type': 'ERROR',
                'row': 'Header',
                'values': f"Missing: {col}",
                'reason': f"Required column '{col}' is missing in CSV header"
            })
            errors += 1
            return issues, errors, warnings
            
    # Check rows
    seen_runs = set()
    for idx, row in df.iterrows():
        row_str = f"Index {idx} (Line {idx + 2})"
        
        # Method / Policy validation
        policy = row.get('policy')
        if policy not in EXPECTED_POLICIES[exp_type]:
            issues.append({
                'type': 'WARNING',
                'row': row_str,
                'values': f"policy = {policy}",
                'reason': f"Unexpected policy label '{policy}'. Expected one of: {EXPECTED_POLICIES[exp_type]}"
            })
            warnings += 1
            
        # Environment validation
        env = row.get('environment')
        if env not in EXPECTED_ENVIRONMENTS[exp_type]:
            issues.append({
                'type': 'WARNING',
                'row': row_str,
                'values': f"environment = {env}",
                'reason': f"Unexpected environment label '{env}'. Expected one of: {EXPECTED_ENVIRONMENTS[exp_type]}"
            })
            warnings += 1
            
        # Scenario validation
        scenario = row.get('scenario')
        scen_set = EXPECTED_SCENARIOS['known'] if exp_type == 'A' else EXPECTED_SCENARIOS['unknown']
        if scenario not in scen_set:
            issues.append({
                'type': 'WARNING',
                'row': row_str,
                'values': f"scenario = {scenario}",
                'reason': f"Unexpected scenario label '{scenario}' for Experiment {exp_type}"
            })
            warnings += 1
            
        # Run validation
        run = row.get('run')
        try:
            run_val = int(run)
            if run_val <= 0:
                issues.append({
                    'type': 'WARNING',
                    'row': row_str,
                    'values': f"run = {run}",
                    'reason': f"Run identifier should be positive integer, got '{run}'"
                })
                warnings += 1
        except (ValueError, TypeError):
            issues.append({
                'type': 'WARNING',
                'row': row_str,
                'values': f"run = {run}",
                'reason': f"Non-integer run identifier '{run}'"
            })
            warnings += 1
            run_val = None
            
        # Duplicate runs check (policy, environment, scenario, run)
        if run_val is not None:
            run_key = (policy, env, scenario, run_val)
            if run_key in seen_runs:
                issues.append({
                    'type': 'WARNING',
                    'row': row_str,
                    'values': f"Key: {run_key}",
                    'reason': f"Duplicate run record for {policy}/{scenario}/run {run_val}"
                })
                warnings += 1
            else:
                seen_runs.add(run_key)
                
        # Result validation
        result = row.get('result')
        if result not in {'success', 'aborted'}:
            issues.append({
                'type': 'WARNING',
                'row': row_str,
                'values': f"result = {result}",
                'reason': f"Unexpected result value '{result}'. Expected 'success' or 'aborted'"
            })
            warnings += 1
            
        # Navigation time check
        nav_time = row.get('navigation_time')
        try:
            nav_time_val = float(nav_time)
            if nav_time_val <= 0:
                issues.append({
                    'type': 'WARNING',
                    'row': row_str,
                    'values': f"navigation_time = {nav_time}",
                    'reason': f"Navigation time must be positive, got {nav_time_val}"
                })
                warnings += 1
        except (ValueError, TypeError):
            issues.append({
                'type': 'WARNING',
                'row': row_str,
                'values': f"navigation_time = {nav_time}",
                'reason': f"Non-numeric navigation_time '{nav_time}'"
            })
            warnings += 1
            
        # Check all other numeric metrics
        for metric in NON_NEGATIVE_METRICS:
            if metric in df.columns and metric != 'navigation_time':
                val = row.get(metric)
                try:
                    val_float = float(val)
                    if val_float < 0:
                        issues.append({
                            'type': 'WARNING',
                            'row': row_str,
                            'values': f"{metric} = {val}",
                            'reason': f"Metric '{metric}' should be non-negative, got {val_float}"
                        })
                        warnings += 1
                except (ValueError, TypeError):
                    if pd.notna(val):
                        issues.append({
                            'type': 'WARNING',
                            'row': row_str,
                            'values': f"{metric} = {val}",
                            'reason': f"Non-numeric value in metric column '{metric}'"
                        })
                        warnings += 1
                        
    return issues, errors, warnings

def validate_experiment_c(df, file_path):
    """Validation checks specifically for Experiment C (Runtime Shared Autonomy)."""
    issues = []
    errors = 0
    warnings = 0
    
    if df is None:
        return issues, errors, warnings
        
    # Verify expected columns are present
    required_cols = ['policy', 'environment', 'scenario', 'run', 'intervention_count', 'navigation_time']
    for col in required_cols:
        if col not in df.columns:
            issues.append({
                'type': 'ERROR',
                'row': 'Header',
                'values': f"Missing: {col}",
                'reason': f"Required column '{col}' is missing in CSV header"
            })
            errors += 1
            return issues, errors, warnings
            
    # Check rows
    seen_runs = set()
    for idx, row in df.iterrows():
        row_str = f"Index {idx} (Line {idx + 2})"
        
        # Method / Policy validation
        policy = row.get('policy')
        if policy not in EXPECTED_POLICIES['C']:
            issues.append({
                'type': 'WARNING',
                'row': row_str,
                'values': f"policy = {policy}",
                'reason': f"Unexpected policy label '{policy}'. Expected one of: {EXPECTED_POLICIES['C']}"
            })
            warnings += 1
            
        # Environment validation
        env = row.get('environment')
        if env not in EXPECTED_ENVIRONMENTS['C']:
            issues.append({
                'type': 'WARNING',
                'row': row_str,
                'values': f"environment = {env}",
                'reason': f"Unexpected environment label '{env}' for Experiment C. Expected one of: {EXPECTED_ENVIRONMENTS['C']}"
            })
            warnings += 1
            
        # Scenario validation
        scenario = row.get('scenario')
        scen_set = EXPECTED_SCENARIOS['known'] if env == 'known_environment' else EXPECTED_SCENARIOS['unknown']
        if scenario not in scen_set:
            issues.append({
                'type': 'WARNING',
                'row': row_str,
                'values': f"scenario = {scenario}",
                'reason': f"Unexpected scenario label '{scenario}' for environment '{env}'"
            })
            warnings += 1
            
        # Run validation
        run = row.get('run')
        try:
            run_val = int(run)
            if run_val <= 0:
                issues.append({
                    'type': 'WARNING',
                    'row': row_str,
                    'values': f"run = {run}",
                    'reason': f"Run identifier should be positive integer, got '{run}'"
                })
                warnings += 1
        except (ValueError, TypeError):
            issues.append({
                'type': 'WARNING',
                'row': row_str,
                'values': f"run = {run}",
                'reason': f"Non-integer run identifier '{run}'"
            })
            warnings += 1
            run_val = None
            
        # Duplicate runs check (policy, environment, scenario, run)
        if run_val is not None:
            run_key = (policy, env, scenario, run_val)
            if run_key in seen_runs:
                issues.append({
                    'type': 'WARNING',
                    'row': row_str,
                    'values': f"Key: {run_key}",
                    'reason': f"Duplicate run record for {policy}/{scenario}/run {run_val}"
                })
                warnings += 1
            else:
                seen_runs.add(run_key)
                
        # Validate stable_recovery_count <= intervention_count
        interventions = row.get('intervention_count')
        recoveries = row.get('stable_recovery_count')
        try:
            int_val = int(interventions)
            rec_val = int(recoveries)
            if rec_val > int_val:
                issues.append({
                    'type': 'WARNING',
                    'row': row_str,
                    'values': f"intervention_count = {int_val}, stable_recovery_count = {rec_val}",
                    'reason': "stable_recovery_count exceeds intervention_count"
                })
                warnings += 1
        except (ValueError, TypeError):
            # Non-integer warnings will be handled under numeric validations below
            pass
            
        # Validate expert_control_time <= navigation_time
        nav_time = row.get('navigation_time')
        exp_time = row.get('expert_control_time')
        try:
            nav_val = float(nav_time)
            exp_val = float(exp_time)
            if exp_val > nav_val:
                issues.append({
                    'type': 'WARNING',
                    'row': row_str,
                    'values': f"navigation_time = {nav_val}, expert_control_time = {exp_val}",
                    'reason': "expert_control_time exceeds navigation_time"
                })
                warnings += 1
        except (ValueError, TypeError):
            pass
            
        # Validate percentage sums: expert_time_percentage + autonomy_time_percentage ≈ 100
        t_exp_pct = row.get('expert_time_percentage')
        t_aut_pct = row.get('autonomy_time_percentage')
        try:
            t_exp_val = float(t_exp_pct)
            t_aut_val = float(t_aut_pct)
            time_sum = t_exp_val + t_aut_val
            if abs(time_sum - 100.0) > 0.1:
                issues.append({
                    'type': 'WARNING',
                    'row': row_str,
                    'values': f"expert_time_percentage = {t_exp_val}, autonomy_time_percentage = {t_aut_val}, sum = {time_sum}",
                    'reason': "expert_time_percentage + autonomy_time_percentage does not sum to 100"
                })
                warnings += 1
        except (ValueError, TypeError):
            pass
            
        # Validate percentage sums: expert_distance_percentage + autonomy_distance_percentage ≈ 100
        d_exp_pct = row.get('expert_distance_percentage')
        d_aut_pct = row.get('autonomy_distance_percentage')
        try:
            d_exp_val = float(d_exp_pct)
            d_aut_val = float(d_aut_pct)
            dist_sum = d_exp_val + d_aut_val
            if abs(dist_sum - 100.0) > 0.1:
                issues.append({
                    'type': 'WARNING',
                    'row': row_str,
                    'values': f"expert_distance_percentage = {d_exp_val}, autonomy_distance_percentage = {d_aut_val}, sum = {dist_sum}",
                    'reason': "expert_distance_percentage + autonomy_distance_percentage does not sum to 100"
                })
                warnings += 1
        except (ValueError, TypeError):
            pass
            
        # Validate non-negativity of all numeric columns in runtime_shared_autonomy
        for metric in RUNTIME_NON_NEGATIVE_METRICS:
            if metric in df.columns:
                val = row.get(metric)
                try:
                    val_float = float(val)
                    if val_float < 0:
                        issues.append({
                            'type': 'WARNING',
                            'row': row_str,
                            'values': f"{metric} = {val}",
                            'reason': f"Metric '{metric}' should be non-negative, got {val_float}"
                        })
                        warnings += 1
                except (ValueError, TypeError):
                    if pd.notna(val):
                        issues.append({
                            'type': 'WARNING',
                            'row': row_str,
                            'values': f"{metric} = {val}",
                            'reason': f"Non-numeric value in metric column '{metric}'"
                        })
                        warnings += 1
                        
    return issues, errors, warnings

def print_report(experiment_name, results):
    """Print the validation results for an experiment."""
    print("=" * 60)
    print(f"Validation Report for {experiment_name}")
    print("=" * 60)
    
    any_issues = False
    for filepath, issues in results.items():
        if issues:
            any_issues = True
            print(f"File: {filepath}")
            for issue in issues:
                print(f"  [{issue['type']}] {issue['row']}: {issue['reason']}")
                print(f"      Values: {issue['values']}")
            print()
            
    if not any_issues:
        print("No issues detected.")
        print()

def main():
    try:
        results_dir = find_results_dir()
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
        
    print(f"Results Directory Located: {results_dir}")
    print()
    
    exp_a_files = sorted(list((results_dir / "experiment_A").glob("*.csv")))
    exp_b_files = sorted(list((results_dir / "experiment_B").glob("*.csv")))
    exp_c_files = sorted(list((results_dir / "runtime_shared_autonomy").glob("*.csv")))
    
    total_files = 0
    total_rows = 0
    total_warnings = 0
    total_errors = 0
    
    report_a = {}
    report_b = {}
    report_c = {}
    
    # ------------------ Validate Experiment A ------------------
    for f in exp_a_files:
        rel_path = f.relative_to(results_dir.parent)
        print(f"Checking Experiment A file: {rel_path}")
        total_files += 1
        
        df, issues, errs, warns = validate_file_general(f)
        total_errors += errs
        total_warnings += warns
        
        if df is not None:
            total_rows += len(df)
            a_issues, a_errs, a_warns = validate_experiment_a_b(df, f, 'A')
            issues.extend(a_issues)
            total_errors += a_errs
            total_warnings += a_warns
            
        report_a[str(rel_path)] = issues
        
    # ------------------ Validate Experiment B ------------------
    for f in exp_b_files:
        rel_path = f.relative_to(results_dir.parent)
        print(f"Checking Experiment B file: {rel_path}")
        total_files += 1
        
        df, issues, errs, warns = validate_file_general(f)
        total_errors += errs
        total_warnings += warns
        
        if df is not None:
            total_rows += len(df)
            b_issues, b_errs, b_warns = validate_experiment_a_b(df, f, 'B')
            issues.extend(b_issues)
            total_errors += b_errs
            total_warnings += b_warns
            
        report_b[str(rel_path)] = issues
        
    # ------------------ Validate Experiment C ------------------
    for f in exp_c_files:
        rel_path = f.relative_to(results_dir.parent)
        print(f"Checking Experiment C file: {rel_path}")
        total_files += 1
        
        df, issues, errs, warns = validate_file_general(f)
        total_errors += errs
        total_warnings += warns
        
        if df is not None:
            total_rows += len(df)
            c_issues, c_errs, c_warns = validate_experiment_c(df, f)
            issues.extend(c_issues)
            total_errors += c_errs
            total_warnings += c_warns
            
        report_c[str(rel_path)] = issues
        
    print()
    print_report("Experiment A", report_a)
    print_report("Experiment B", report_b)
    print_report("Experiment C", report_c)
    
    print("=" * 60)
    print("Validation Run Summary")
    print("=" * 60)
    print(f"Files checked: {total_files}")
    print(f"Rows checked: {total_rows}")
    print(f"Warnings: {total_warnings}")
    print(f"Errors: {total_errors}")
    print("=" * 60)

if __name__ == "__main__":
    main()
