# Shielded DAgger: A Deterministic Fallback Architecture for Safe Imitation Learning in Robotics

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![ROS 2](https://img.shields.io/badge/ROS-2-22314E.svg)](https://www.ros.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.3-red.svg)](https://pytorch.org)
[![Nav2](https://img.shields.io/badge/Navigation-Nav2-purple.svg)](https://navigation.ros.org/)
[![Controller](https://img.shields.io/badge/Expert-MPPI-orange.svg)]()
[![Gazebo](https://img.shields.io/badge/Simulator-Gazebo-orange.svg)](https://gazebosim.org/)
[![ONNX](https://img.shields.io/badge/Inference-ONNX-green.svg)](https://onnx.ai/)

## Overview

This repository contains the implementation of **Shielded DAgger**, a safety-aware imitation learning framework for autonomous robot navigation in unstructured environments.

The project addresses one of the major limitations of standard imitation learning: **covariate shift**. Even when a behavior-cloned policy achieves low offline loss, small prediction errors can accumulate during deployment, pushing the robot into out-of-distribution (OOD) states where failures become likely.

Shielded DAgger is designed to mitigate this by combining:

- A fast learned policy for nominal control  
- A deterministic **MPPI expert planner** for recovery  
- A real-time **safety shield** for arbitration  
- A **Safe DAgger pipeline** for dataset aggregation  

When the learned policy approaches unsafe or unfamiliar states, control is transferred to the expert planner. Recovery trajectories are recorded and used to iteratively improve the policy on failure cases.

---

## Research Objective

Design a safe imitation learning pipeline that reduces collision rates by shielding learned policies from catastrophic out-of-distribution failures while allowing continuous adaptation through expert intervention.

**Manuscript in preparation:**  
**Ikechukwu S., Sawyerr B.A.**  
*Shielded DAgger: A Deterministic Fallback Architecture for Safe Imitation Learning in Robotics* (Under Review)

---

## Motivation: Why Shielded DAgger?

In standard Behavior Cloning (BC), a neural network learns a mapping:

**State → Action**

where state includes:

- LiDAR observations  
- Robot odometry  
- Local goal information  

and action includes:

- Linear velocity (`v`)
- Angular velocity (`ω`)

The problem is that BC assumes the robot remains within the same state distribution as the training data.

In reality, small prediction errors compound over time.

A slight steering mistake can produce a new state the policy has never seen before. Once outside the training distribution, predictions become unreliable and may lead to collisions.

This is the classic **covariate shift problem**.

---

## Core Idea

Instead of replacing classical planning entirely, Shielded DAgger treats the learned policy as a lightweight approximation of an expert planner.

- **Neural Policy:** fast inference (~O(1))
- **MPPI Expert:** robust but computationally expensive
- **Safety Shield:** monitors policy reliability

The safety shield ensures the robot remains within safe operational boundaries.

If the learned policy violates safety constraints, the expert takes over.

This creates a practical form of **safe dataset aggregation (DAgger)**.

---

## System Architecture

### 1. Behavioral Cloning Policy

A PyTorch MLP policy trained on navigation demonstrations.

**Input (40D observation vector):**

- 36 LiDAR beams  
- 2 odometry values  
- 2 local goal coordinates  

**Output:**

- Linear velocity  
- Angular velocity  

The model is exported to **ONNX** for low-latency inference at **10 Hz**.

---

### 2. Safety Monitor

The safety monitor runs continuously and publishes a binary risk signal (`safe` / `unsafe`) based on four hierarchical trigger conditions.

#### Trigger 1 — LiDAR Panic Check (Immediate Collision Risk)

Emergency takeover is triggered if the robot is dangerously close to an obstacle in any direction.

```python
latest_scan < lidar_threshold
```

This represents the highest-priority safety condition.

---

#### Trigger 2 — Reverse Collision Check

If the BC policy commands reverse motion, rear obstacle clearance is explicitly checked.

```python
bc_linear_velocity < -0.01 and rear_scan < rear_lidar_threshold
```

This prevents the policy from reversing into unseen obstacles.

---

#### Trigger 3 — Inflation Boundary Breach (Early Warning)

Takeover is triggered when the robot enters the inflated costmap safety margin, even before direct collision risk.

```python
latest_scan < warning_threshold
```

This acts as a predictive safety buffer.

---

#### Trigger 4 — Policy–Expert Disagreement

The shield compares intended actions from:

- Behavioral Cloning policy
- MPPI expert planner

Three disagreement modes are evaluated:

##### Angular disagreement
```python
abs(mppi_w - bc_w) > delta_w_threshold
```

##### Linear velocity disagreement
```python
abs(mppi_v - bc_v) > delta_v_threshold
```

##### Topological disagreement (opposite turning direction)
```python
sign(mppi_w) != sign(bc_w)
```

This detects situations where both policies propose fundamentally different recovery strategies.

Typical thresholds:

- `Δv > 0.25 m/s`
- `Δω > 1.0 rad/s`
- Obstacle distance `≤ 0.28 m`

---

### 3. Arbitrator State Machine

Handles control transfer between:

- **POLICY mode**
- **EXPERT mode**

The arbitrator ensures stable hand-offs and prevents oscillatory switching.

---

### 4. Safe DAgger Data Pipeline

During expert intervention:

- Sensor streams are synchronized  
- Recovery trajectories are recorded  
- Data stored as `.hdf5`  
- Recovery samples added to training set  
- Policy retrained on failure cases  

This allows the robot to improve precisely where it previously failed.

---

# Key Engineering Challenges

## 1. Arbitration Chattering (“Watchdog Starvation”)

### Problem

During tight turns and narrow corridor maneuvers, the system entered a loop:

`Expert → Policy → Expert`

The deadlock timer incorrectly interpreted temporary MPPI zero-velocity replanning as failure.

### Solution

Implemented a **continuous inactivity watchdog**.

The timer now only advances when the expert is truly stationary:

- `Δv ≈ 0`
- `Δω ≈ 0`

This eliminated arbitration chattering.

---

## 2. Hardware Asynchrony & Dataset Corruption

### Problem

Running simultaneously:

- Gazebo simulation  
- MPPI planner  
- ONNX inference  

caused CPU bottlenecks.

This produced “frozen frames” where:

- Command velocity was high  
- Actual odometry remained zero  

Such samples polluted DAgger data.

### Solution

Built a **rolling-window deadlock filter** in Python/NumPy.

The filter removes lag-induced corrupted samples by comparing:

- commanded velocity  
vs  
- actual odometry  

over a sliding time window.

This significantly improved dataset quality.

---

## 3. Overly Aggressive Safety Thresholds

### Problem

Initial thresholds caused unnecessary interventions even when the policy was safe but slower than MPPI.

### Solution

Performed empirical threshold tuning.

Final thresholds reduced:

- false positives  
- unnecessary takeovers  
- CPU overhead  

The shield became an emergency fallback rather than an overactive supervisor.

---

## 4. Global Planner Latency

### Problem

Low-frequency global planning caused outdated sub-goals.

This created a “sliding carrot” effect where the local policy chased stale targets.

### Solution

Implemented **dynamic Pure Pursuit lookahead** using `tf2`.

The robot continuously tracks a local sub-goal approximately **1.5 m ahead**, decoupling local control from global planning latency.

---

## Tech Stack

- Python  
- ROS 2  
- PyTorch  
- ONNX Runtime  
- Gazebo  
- NumPy  
- HDF5  
- MPPI Planner  
- Costmaps / tf2  

---

## Future Work

- Multi-modal policy architectures (CNN / Transformer)
- Uncertainty-aware shield triggering
- Reinforcement learning fine-tuning after DAgger
- Adaptive threshold learning
- Real-world deployment on physical mobile robot

---

## Citation

```bibtex
@misc{ikechukwu2026shieldeddagger,
  title={Shielded DAgger: A Deterministic Fallback Architecture for Safe Imitation Learning in Robotics},
  author={Sunday Ikechukwu and B. A. Sawyerr},
  year={2026},
  note={Under Review}
}
```