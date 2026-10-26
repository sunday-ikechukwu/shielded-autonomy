import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import Twist
from action_msgs.msg import GoalStatusArray

import tf2_ros
import tf2_geometry_msgs
from tf2_ros import TransformException

import numpy as np
import onnxruntime as ort
import os

class SafeDaggerInferenceNode(Node):
    def __init__(self):
        super().__init__('safe_dagger_inference_node')

        # ── Parameters ────────────────────────────────────────────────────
        self.declare_parameter('model_path',
            os.path.expanduser('~/nav2_ws/src/shielded_autonomy/models/dagger_model_v2/bc_dagger_1x.onnx'))
        self.declare_parameter('norm_stats_path',
            os.path.expanduser('~/nav2_ws/src/shielded_autonomy/models/dagger_model_v2/bc_dagger_1x_norm_stats.npz'))
        self.declare_parameter('scan_range_max', 3.5)
        self.declare_parameter('inference_hz', 10.0)

        # Safety limits
        self.declare_parameter('max_linear_x',  0.26)
        self.declare_parameter('max_angular_z', 1.9)

        model_path          = self.get_parameter('model_path').value
        norm_stats_path     = self.get_parameter('norm_stats_path').value
        self.scan_range_max = self.get_parameter('scan_range_max').value
        inference_hz        = self.get_parameter('inference_hz').value
        self.max_linear_x   = self.get_parameter('max_linear_x').value
        self.max_angular_z  = self.get_parameter('max_angular_z').value

        # ── Load ONNX model ───────────────────────────────────────────────
        if not os.path.exists(model_path):
            self.get_logger().error(f'Model not found: {model_path}')
            raise FileNotFoundError(model_path)

        self.ort_session = ort.InferenceSession(model_path)
        self.input_name  = self.ort_session.get_inputs()[0].name
        self.get_logger().info(f'ONNX model loaded: {model_path}')

        # ── Load normalization statistics ─────────────────────────────────
        if not os.path.exists(norm_stats_path):
            self.get_logger().error(f'Norm stats not found: {norm_stats_path}')
            raise FileNotFoundError(norm_stats_path)

        stats         = np.load(norm_stats_path)
        self.obs_mean = stats['obs_mean'].astype(np.float32)  
        self.obs_std  = stats['obs_std'].astype(np.float32)   
        self.act_mean = stats['act_mean'].astype(np.float32)  
        self.act_std  = stats['act_std'].astype(np.float32)   
        self.get_logger().info('Normalization statistics loaded.')

        # ── TF2 ───────────────────────────────────────────────────────────
        self.tf_buffer   = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # ── Latest state storage ──────────────────────────────────────────
        self.latest_scan     = None
        self.latest_odom_vel = None
        self.current_goal    = None
        self.navigating      = False
        self.tracked_goal_id = None
        self.current_plan    = []     # Store full plan for debugging/replanning

        # ── QoS for sensor topics ─────────────────────────────────────────
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            depth=5
        )

        # ── Subscribers & Publishers ──────────────────────────────────────
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, qos_profile=sensor_qos)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, qos_profile=sensor_qos)
        self.plan_sub = self.create_subscription(Path, '/plan', self.plan_callback, 10)
        self.status_sub = self.create_subscription(GoalStatusArray, '/navigate_to_pose/_action/status', self.status_callback, 10)
        
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel_bc', 10)

        # ── Inference timer ───────────────────────────────────────────────
        timer_period = 1.0 / inference_hz
        self.timer = self.create_timer(timer_period, self.inference_loop)

        self.get_logger().info(f'SafeDAgger Inference Node ready at {inference_hz}Hz. Waiting for plan...')

    # ── Callbacks  ─────────────────────────────
    def scan_callback(self, msg: LaserScan):
        ranges = np.array(msg.ranges, dtype=np.float32)
        ranges = np.clip(ranges, 0.0, self.scan_range_max)
        ranges = np.nan_to_num(ranges, nan=self.scan_range_max)
        self.latest_scan = ranges[::10]  # downsample 360 → 36

    def odom_callback(self, msg: Odometry):
        self.latest_odom_vel = np.array([
            msg.twist.twist.linear.x,
            msg.twist.twist.angular.z
        ], dtype=np.float32)

    def plan_callback(self, msg: Path):
        """ Dynamically tracks a carrot 1.5m ahead of the robot's REAL-TIME position. """
        if not msg.poses:
            return
        
        # Store full plan for debugging/visualization purposes
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
        if not msg.status_list:
            return
            
        # If we are not tracking any goal, look for an active one (status 1 = Accepted, 2 = Executing)
        if self.tracked_goal_id is None:
            for status_item in msg.status_list:
                if status_item.status in [1, 2]:
                    self.tracked_goal_id = tuple(status_item.goal_info.goal_id.uuid)
                    self.get_logger().info(f"New Nav2 Goal Accepted! Tracking Goal ID: {self.tracked_goal_id}")
                    self.navigating = True
                    break

        # If we are tracking a goal, monitor its lifecycle
        if self.tracked_goal_id is not None:
            tracked_item = None
            for status_item in msg.status_list:
                if tuple(status_item.goal_info.goal_id.uuid) == self.tracked_goal_id:
                    tracked_item = status_item
                    break

            if tracked_item is not None:
                status_code = tracked_item.status
                if status_code in [4, 5, 6]:
                    self.get_logger().info(f"Tracked Goal Ended (Status: {status_code}). Stopping Robot.")
                    self.navigating = False
                    self.current_goal = None
                    self.current_plan    = []
                    self.tracked_goal_id = None
                    
                    # Hard stop the robot
                    stop = Twist()
                    self.cmd_pub.publish(stop)

    # ── Inference loop ────────────────────────────────────────────────────
    def inference_loop(self):
        if not self.navigating:
            return

        if (self.latest_scan is None or self.latest_odom_vel is None or self.current_goal is None):
            return

        # ── Get goal in robot frame ───────────────────────────────────────
        goal_local = self.get_goal_in_robot_frame()
        if goal_local is None:
            return

        # ── Build observation vector ──────────────────────────────────────
        obs = np.concatenate([
            self.latest_scan,
            self.latest_odom_vel,
            goal_local
        ]).astype(np.float32)

        # ── Normalize ─────────────────────────────────────────────────────
        obs_norm = (obs - self.obs_mean) / self.obs_std

        # ── ONNX inference ────────────────────────────────────────────────
        input_tensor = obs_norm.reshape(1, -1)
        ort_outputs  = self.ort_session.run(None, {self.input_name: input_tensor})

        # ── Denormalize output ────────────────────────────────────────────
        action_norm = ort_outputs[0][0]  
        action = (action_norm * self.act_std) + self.act_mean

        # ── Safety clipping ───────────────────────────────────────────────
        linear_x  = float(np.clip(action[0], -self.max_linear_x, self.max_linear_x))
        angular_z = float(np.clip(action[1], -self.max_angular_z, self.max_angular_z))

        # ── Publish ───────────────────────────────────────────────────────
        twist = Twist()
        twist.linear.x  = linear_x
        twist.angular.z = angular_z
        self.cmd_pub.publish(twist)

    def get_goal_in_robot_frame(self):
        try:
            transform = self.tf_buffer.lookup_transform(
                'base_link',
                self.current_goal.header.frame_id,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.1)
            )
            goal_robot_frame = tf2_geometry_msgs.do_transform_pose_stamped(
                self.current_goal, transform
            )

            dx = goal_robot_frame.pose.position.x
            dy = goal_robot_frame.pose.position.y

            goal_distance = np.sqrt(dx**2 + dy**2)

            # Deadband guard: if we are very close to the goal, clamp angle to 0
            # to prevent origin mathematical singularity and angular oscillations.
            if goal_distance < 0.15:
                goal_angle = 0.0
            else:
                goal_angle = np.arctan2(dy, dx)

            return np.array([goal_distance, goal_angle], dtype=np.float32)

        except TransformException as e:
            self.get_logger().warn(f'TF not ready: {e}', throttle_duration_sec=2.0)
            return None

def main(args=None):
    rclpy.init(args=args)
    node = SafeDaggerInferenceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        stop = Twist()
        node.cmd_pub.publish(stop)
        node.get_logger().info('Shutting down — robot stopped.')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()