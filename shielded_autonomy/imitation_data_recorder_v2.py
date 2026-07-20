"""
imitation_data_recorder_v2.py
Records (observation, action) pairs with DWA/MPPI as the expert policy.
Uses Local Carrot (~1.5m) and UUID state-machine tracking.
Saves to HDF5 for offline behavior cloning training.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import Twist, PoseStamped
from tf_transformations import euler_from_quaternion
from action_msgs.msg import GoalStatusArray

from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
import message_filters
from message_filters import ApproximateTimeSynchronizer

import tf2_ros
import tf2_geometry_msgs
from tf2_ros import TransformException

from rclpy.duration import Duration

import numpy as np
import h5py
from datetime import datetime
import os

class ImitationDataRecorderV2(Node):
    def __init__(self):
        super().__init__('imitation_data_recorder_v2')

        # Parameters
        self.declare_parameter('save_directory', os.path.expanduser('~/nav2_ws/src/shielded_autonomy/imitation_data/MPPI_expert_data_v2/'))
        self.declare_parameter('episode_name', 'episode')
        self.declare_parameter('max_steps', 10000)
        self.declare_parameter('scan_range_max', 3.5)  # TurtleBot3 LiDAR max range

        self.save_directory = self.get_parameter('save_directory').value
        self.episode_name = self.get_parameter('episode_name').value
        self.max_steps = self.get_parameter('max_steps').value
        self.scan_range_max = self.get_parameter('scan_range_max').value
        
        # Create save directory if it doesn't exist
        os.makedirs(self.save_directory, exist_ok=True)

        # State Machine Variables
        self.current_goal = None
        self.current_plan = []
        self.recording = False
        self.tracked_goal_id = None
        self.episode_outcome = None

        # TF buffer and listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Initialize data buffers
        self.reset_buffers()

        # QoS profile for sensor
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            depth=5
        )

        scan_sub = message_filters.Subscriber(
            self, LaserScan, '/scan', qos_profile=sensor_qos)
        odom_sub = message_filters.Subscriber(
            self, Odometry, '/odom', qos_profile=sensor_qos)

        self.sync = ApproximateTimeSynchronizer(
            [scan_sub, odom_sub], queue_size=10, slop=0.1)
        self.sync.registerCallback(self.sync_callback)

        self.plan_sub = self.create_subscription(Path, '/plan', self.plan_callback, 10)
        
        self.cmd_vel_sub = self.create_subscription(Twist, '/cmd_vel', self.cmd_vel_callback, 10)

        # Status subscriber to detect episode outcomes
        self.status_sub = self.create_subscription(GoalStatusArray, '/navigate_to_pose/_action/status', self.status_callback, 10)

        self.get_logger().info('V2 Imitation Data Recorder node ready. Send a goal in RViz to start recording.')

    def reset_buffers(self):
        """Clears all data arrays for a fresh recording session."""
        self.scans_buffer       = []
        self.odom_vel_buffer    = []
        self.goal_local_buffer  = []
        self.actions_buffer     = []
        self.time_buffer        = []
        self.step_count         = 0
        self.latest_cmd_vel     = None

    def plan_callback(self, msg: Path):
        """ Dynamically tracks a carrot 1.5m ahead of the robot's REAL-TIME position. """
        if not msg.poses:
            return

        self.current_plan = msg.poses

        # ── Step 1: Find closest path point to robot ──────────────────
        try:
            # Use the path's native frame (usually 'map') instead of hardcoding
            transform = self.tf_buffer.lookup_transform(
                msg.header.frame_id, 'base_link',
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.0) # 0.0 prevents blocking the callback
            )
            robot_x = transform.transform.translation.x
            robot_y = transform.transform.translation.y
        except TransformException as e:
            # If TF not ready, gracefully fall back to the start of the path
            robot_x = msg.poses[0].pose.position.x
            robot_y = msg.poses[0].pose.position.y

        min_dist = float('inf')
        closest_idx = 0
        for i, pose in enumerate(msg.poses):
            dx = pose.pose.position.x - robot_x
            dy = pose.pose.position.y - robot_y
            dist = np.sqrt(dx**2 + dy**2)
            if dist < min_dist:
                min_dist = dist
                closest_idx = i

        # ── Step 2: Walk 1.5m forward from closest point ──────────────
        target_lookahead = 1.5
        cumulative_dist  = 0.0
        carrot_pose      = msg.poses[-1]  # default to final goal if path is short

        for i in range(closest_idx + 1, len(msg.poses)):
            p1 = msg.poses[i-1].pose.position
            p2 = msg.poses[i].pose.position
            cumulative_dist += np.sqrt((p2.x - p1.x)**2 + (p2.y - p1.y)**2)
            
            if cumulative_dist >= target_lookahead:
                carrot_pose = msg.poses[i]
                break

        self.current_goal = carrot_pose
    
    def status_callback(self, msg: GoalStatusArray):
        """
        LIFECYCLE NODE: Strictly manages UUIDs and Recording States.
        Never does geometric math.
        """
        if not msg.status_list:
            return

        # ── 1. START NEW EPISODE ──
        if self.tracked_goal_id is None:
            for status_item in msg.status_list:
                if status_item.status in [1, 2]:  # Accepted or Executing
                    self.tracked_goal_id = tuple(status_item.goal_info.goal_id.uuid)
                    self.get_logger().info(f"✅ New Goal UUID Accepted: {self.tracked_goal_id}. Starting Recording.")
                    
                    self.reset_buffers()
                    self.episode_outcome = None
                    self.recording = True
                    break

        # ── 2. MONITOR ACTIVE EPISODE ──
        if self.tracked_goal_id is not None:
            tracked_item = None
            for status_item in msg.status_list:
                if tuple(status_item.goal_info.goal_id.uuid) == self.tracked_goal_id:
                    tracked_item = status_item
                    break

            if tracked_item is not None:
                status_code = tracked_item.status
                
                # 4 = Succeeded, 5 = Canceled, 6 = Aborted
                if status_code in [4, 5, 6]:
                    self.get_logger().info(f"🛑 Goal Ended (Status: {status_code}). Handling outcome...")
                    
                    if self.step_count > 0:
                        if status_code == 4:
                            self.episode_outcome = 'success'
                        elif status_code == 5:
                            self.episode_outcome = 'canceled'
                        elif status_code == 6:
                            self.episode_outcome = 'aborted'
                            
                        self.save_episode()
                    else:
                        self.get_logger().info('Discarding empty episode (0 steps).')
                    
                    # ── RESET STATE MACHINE ──
                    self.recording = False
                    self.current_goal = None
                    self.current_plan = []
                    self.episode_outcome = None
                    self.tracked_goal_id = None
        
    def cmd_vel_callback(self, msg: Twist):
        self.latest_cmd_vel = msg

    def sync_callback(self, scan: LaserScan, odom: Odometry):
        # don't record if state machine is off or carrot isn't ready
        if not self.recording or self.current_goal is None:
            return
        
        if self.latest_cmd_vel is None:
            return

        # PROCESS LIDAR — downsample 360 → 36 rays
        ranges = np.array(scan.ranges, dtype=np.float32)
        ranges = np.clip(ranges, 0.0, self.scan_range_max)
        ranges = np.nan_to_num(ranges, nan=self.scan_range_max)
        ranges = ranges[::10]  # 360 → 36 rays

        # PROCESS ODOMETRY — extract velocity only
        odom_vel = np.array([
            odom.twist.twist.linear.x,
            odom.twist.twist.angular.z
        ], dtype=np.float32)

        # TRANSFORM GOAL TO ROBOT FRAME
        goal_local = self.get_goal_in_robot_frame(scan.header.stamp)
        if goal_local is None:
            return
        
        # EXPERT ACTION — what MPPI actually commanded
        action = np.array([
            self.latest_cmd_vel.linear.x,
            self.latest_cmd_vel.angular.z
        ], dtype=np.float32)

        # GET CURRENT TIME for timestamping
        current_time_sec = self.get_clock().now().nanoseconds / 1e9

        # APPEND TO BUFFERS
        self.scans_buffer.append(ranges)
        self.odom_vel_buffer.append(odom_vel)
        self.goal_local_buffer.append(goal_local)
        self.actions_buffer.append(action)
        self.time_buffer.append(current_time_sec)
        self.step_count += 1

        if self.step_count % 100 == 0:
            self.get_logger().info(f'Recorded {self.step_count} steps...')

        # Safety net — force save if episode runs too long
        if self.step_count >= self.max_steps:
            self.get_logger().warn('Max steps reached — force saving.')
            self.episode_outcome = 'max_steps'
            self.save_episode()
            
            # Reset state machine manually since Action Server didn't finish it
            self.recording = False
            self.current_goal = None
            self.episode_outcome = None
            self.tracked_goal_id = None
        
    def get_goal_in_robot_frame(self, timestamp):
        try:
            transform = self.tf_buffer.lookup_transform(
                'base_link',
                self.current_goal.header.frame_id,
                rclpy.time.Time(),
                timeout=Duration(seconds=0.1)
            )
            goal_robot_frame = tf2_geometry_msgs.do_transform_pose_stamped(
                self.current_goal, transform
            )

            dx = goal_robot_frame.pose.position.x      # forward distance to goal
            dy = goal_robot_frame.pose.position.y      # lateral distance to goal

            # How far is the goal
            goal_distance = np.sqrt(dx**2 + dy**2)

            # # Which direction to turn to FACE the goal right now
            # goal_angle = np.arctan2(dy, dx)
            # Deadband guard: if we are very close to the goal, clamp angle to 0
            # to prevent origin mathematical singularity and angular oscillations.
            if goal_distance < 0.15:
                goal_angle = 0.0
            else:
                goal_angle = np.arctan2(dy, dx)

            return np.array([goal_distance, goal_angle], dtype=np.float32)
        
        except TransformException as e:
            self.get_logger().warn(
                f'TF not ready: {e}', throttle_duration_sec=2.0)
            return None
            
    def save_episode(self):
        self.recording = False
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = os.path.join(
            self.save_directory,
            f'{self.episode_name}_{self.episode_outcome}_{timestamp}.hdf5'
        )

        with h5py.File(filename, 'w') as f:
            f.create_dataset('scans',      data=np.array(self.scans_buffer))
            f.create_dataset('odom_vel',   data=np.array(self.odom_vel_buffer))
            f.create_dataset('goal_local', data=np.array(self.goal_local_buffer))
            f.create_dataset('actions',    data=np.array(self.actions_buffer))
            
            # add timestamp to calculate effective polling rate later if needed
            f.create_dataset('timestamps_sec', data=np.array(self.time_buffer, dtype=np.float64))

            # Metadata — outcome flag is the critical one.
            f.attrs['episode_name']   =         self.episode_name
            f.attrs['outcome']        =         self.episode_outcome
            f.attrs['num_steps']      =         self.step_count
            f.attrs['obs_dim']        =         '36 (scan) + 2 (odom_vel) + ' \
                                                '2 (goal_distance, goal_angle) = 40'
            f.attrs['action_dim']     =         '2: [linear_x, angular_z]'
        
        self.get_logger().info(
            f'Episode saved [{self.episode_outcome}]: {self.step_count} steps → {filename}'
        )

        # Clear RAM 
        self.reset_buffers()


def main(args=None):
    rclpy.init(args=args)
    node = ImitationDataRecorderV2()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        if node.step_count > 0:
            node.get_logger().info('Keyboard interrupt — saving current episode before exiting.')
            node.episode_outcome = 'interrupted'
            node.save_episode()
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()