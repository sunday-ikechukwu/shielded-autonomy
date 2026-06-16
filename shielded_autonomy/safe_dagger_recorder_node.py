import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import Twist
from std_msgs.msg import String
from action_msgs.msg import GoalStatusArray

import tf2_ros
import tf2_geometry_msgs
from tf2_ros import TransformException
import numpy as np
import h5py
from datetime import datetime
import os
from collections import deque

class SafeDaggerRecorderNode(Node):
    def __init__(self):
        super().__init__('safe_dagger_recorder_node')

        # --- Parameters ---
        self.declare_parameter('save_directory', os.path.expanduser('~/nav2_ws/src/shielded_autonomy/imitation_data/dagger_data/'))
        self.declare_parameter('scan_range_max', 3.5)

        self.save_directory = self.get_parameter('save_directory').value
        self.scan_range_max = self.get_parameter('scan_range_max').value

        os.makedirs(self.save_directory, exist_ok=True)
        self.output_filepath = os.path.join(self.save_directory, 'dagger_recoveries.hdf5')

        # --- State Variables ---
        self.current_goal = None
        self.recording_active = False
        self.current_mode = "BC_POLICY"
        self.latest_trigger_reason = "UNKNOWN"

        # --- Volatile Recovery Chunk Buffers ---
        self.scans_buffer = []
        self.odom_vel_buffer = []
        self.goal_local_buffer = []
        self.actions_buffer = []
        self.time_buffer = []
        self.step_count = 0
       

        # --- The Pre-Trigger Memory Buffer (~1 second of context) ---
        self.pretrigger_buffer = deque(maxlen=10)

        # --- Volatile Storage for Synchronization ---
        self.latest_odom_vel = None
        self.latest_mppi_cmd = None

        # --- TF2 Setup ---
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # --- QoS ---
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            depth=5
        )

        # --- Subscribers ---
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, qos_profile=sensor_qos)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, qos_profile=sensor_qos)
        self.plan_sub = self.create_subscription(Path, '/plan', self.plan_callback, 10)
        self.mppi_sub = self.create_subscription(Twist, '/cmd_vel_mppi', self.mppi_callback, 10)
        self.mode_sub = self.create_subscription(String, '/active_mode', self.mode_callback, 10)
        self.status_sub = self.create_subscription(GoalStatusArray, '/navigate_to_pose/_action/status', self.status_callback, 10)
        self.reason_sub = self.create_subscription(String, '/trigger_reason', self.reason_callback, 10)

        self.get_logger().info(f'Safe DAgger Hybrid Recorder Ready.\nTarget File: {self.output_filepath}')
    
    def reason_callback(self, msg: String):
        self.latest_trigger_reason = msg.data

    def capture_frame(self, scan, goal_local):
        """Helper to package the current synced state."""
        return {
            'scan': scan.copy(),
            'odom': self.latest_odom_vel.copy(),
            'goal': goal_local.copy(),
            'action': self.latest_mppi_cmd.copy(),
            'time': self.get_clock().now().nanoseconds / 1e9
        }

    def plan_callback(self, msg: Path):
        if self.recording_active:
            return
        if not msg.poses:
            return

        self.current_goal = msg.poses[-1]
        self.recording_active = True
        
        self.reset_buffers()
        self.get_logger().info('New Nav2 plan detected. Active and monitoring for expert interventions...')

    def mode_callback(self, msg: String):
        new_mode = msg.data
        
        # BC -> EXPERT transition (Trigger Fired!)
        if self.current_mode != "EXPERT" and new_mode == "EXPERT":
            if len(self.pretrigger_buffer) > 0:
                self.get_logger().warn(
                    f"Expert takeover detected. Injecting {len(self.pretrigger_buffer)} pre-trigger frames."
                )
                
                # Unload the rolling memory into the permanent chunk buffers
                for frame in self.pretrigger_buffer:
                    self.scans_buffer.append(frame['scan'])
                    self.odom_vel_buffer.append(frame['odom'])
                    self.goal_local_buffer.append(frame['goal'])
                    self.actions_buffer.append(frame['action'])
                    self.time_buffer.append(frame['time'])
                    self.step_count += 1
            
            self.pretrigger_buffer.clear()

        # EXPERT -> BC transition (Recovery Complete)
        elif self.current_mode == "EXPERT" and new_mode != "EXPERT":
            if self.step_count > 0:
                self.get_logger().info('Shield deactivated control. Packaging continuous recovery chunk...')
                self.save_recovery_chunk()
                
        self.current_mode = new_mode

    def scan_callback(self, msg: LaserScan):
        # We process data as long as we are active and have a goal (Even in BC Mode!)
        if not self.recording_active or self.current_goal is None:
            return
        
        ranges = np.array(msg.ranges, dtype=np.float32)
        ranges = np.clip(ranges, 0.0, self.scan_range_max)
        ranges = np.nan_to_num(ranges, nan=self.scan_range_max)
        latest_scan = ranges[::10]

        goal_local = self.get_goal_in_robot_frame()
        if goal_local is None or self.latest_odom_vel is None or self.latest_mppi_cmd is None:
            return

        # Always capture the frame and add it to our rolling memory buffer
        frame = self.capture_frame(latest_scan, goal_local)
        self.pretrigger_buffer.append(frame)

        # ONLY commit to the permanent chunk buffers if the Expert is actively driving
        if self.current_mode != "EXPERT":
            return

        self.scans_buffer.append(frame['scan'])
        self.odom_vel_buffer.append(frame['odom'])
        self.goal_local_buffer.append(frame['goal'])
        self.actions_buffer.append(frame['action'])
        self.time_buffer.append(frame['time'])
        self.step_count += 1

    def odom_callback(self, msg: Odometry):
        self.latest_odom_vel = np.array([msg.twist.twist.linear.x, msg.twist.twist.angular.z], dtype=np.float32)

    def mppi_callback(self, msg: Twist):
        self.latest_mppi_cmd = np.array([msg.linear.x, msg.angular.z], dtype=np.float32)

    def get_goal_in_robot_frame(self):
        try:
            transform = self.tf_buffer.lookup_transform('base_link', self.current_goal.header.frame_id, rclpy.time.Time(), timeout=rclpy.duration.Duration(seconds=0.1))
            goal_robot = tf2_geometry_msgs.do_transform_pose_stamped(self.current_goal, transform)
            dx, dy = goal_robot.pose.position.x, goal_robot.pose.position.y
            return np.array([np.sqrt(dx**2 + dy**2), np.arctan2(dy, dx)], dtype=np.float32)
        except TransformException:
            return None

    def status_callback(self, msg: GoalStatusArray):
        if not self.recording_active or not msg.status_list:
            return

        latest_status_code = msg.status_list[-1].status
        if latest_status_code in [4, 5, 6]:
            self.get_logger().info(f'Nav2 navigation lifecycle ended (Status: {latest_status_code}). Closing active tracking.')
            if self.step_count > 0:
                self.save_recovery_chunk()
            self.recording_active = False
            self.current_goal = None

    def save_recovery_chunk(self):
        if self.step_count == 0:
            return

        if self.step_count < 5:
            self.get_logger().warn(
                f'Skipping chunk with only {self.step_count} steps — too few to be useful.'
            )
            self.reset_buffers()
            return

        with h5py.File(self.output_filepath, 'a') as f:
            # Safer HDF5 indexing based on sorted keys
            existing_keys = sorted(f.keys())
            if not existing_keys:
                next_index = 1
            else:
                last_index = int(existing_keys[-1].split('_')[-1])
                next_index = last_index + 1
                
            recovery_id = f"recovery_{next_index:04d}"
            
            group = f.create_group(recovery_id)
            group.create_dataset('scans', data=np.array(self.scans_buffer))
            group.create_dataset('odom_vel', data=np.array(self.odom_vel_buffer))
            group.create_dataset('goal_local', data=np.array(self.goal_local_buffer))
            group.create_dataset('actions', data=np.array(self.actions_buffer))
            group.create_dataset('timestamps_sec', data=np.array(self.time_buffer, dtype=np.float64))

            group.attrs['num_steps'] = self.step_count
            group.attrs['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            group.attrs['trigger_reason'] = self.latest_trigger_reason

        self.get_logger().info(f'Successfully consolidated {self.step_count} frames into internal group: [{recovery_id}]')
        self.reset_buffers()

    def reset_buffers(self):
        self.scans_buffer.clear()
        self.odom_vel_buffer.clear()
        self.goal_local_buffer.clear()
        self.actions_buffer.clear()
        self.time_buffer.clear()
        self.pretrigger_buffer.clear()
        self.step_count = 0
        self.latest_trigger_reason = "UNKNOWN"

def main(args=None):
    rclpy.init(args=args)
    node = SafeDaggerRecorderNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        if node.step_count > 0:
            node.get_logger().info('Shutdown signal caught. Committing outstanding buffer to disk...')
            node.save_recovery_chunk()
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

