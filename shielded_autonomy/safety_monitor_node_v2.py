import rclpy
from rclpy.node import Node 
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

from sensor_msgs.msg    import LaserScan
from std_msgs.msg       import Bool, String  # Added String for reason_pub
from geometry_msgs.msg  import Twist
from action_msgs.msg    import GoalStatusArray

import numpy as np 

class SafetyMonitorNodeV2(Node):
    def __init__(self):
        super().__init__('safety_monitor_node_v2')

        #___________Parameter_________________________
        self.declare_parameter('lidar_panic_threshold', 0.22)
        self.declare_parameter('lidar_warning_threshold', 0.35)
        self.declare_parameter('lidar_rear_panic_threshold', 0.25) # New parameter for rear LiDAR panic
        self.declare_parameter('action_delta_w_threshold', 0.6) #0.8
        self.declare_parameter('action_delta_v_threshold', 0.12) #0.15
        self.declare_parameter('min_turn_to_compare', 0.2) # 0.15 Sign mismatch limit
        self.declare_parameter('check_hz', 10.0)

        self.lidar_threshold = self.get_parameter('lidar_panic_threshold').value
        self.warning_threshold = self.get_parameter('lidar_warning_threshold').value
        self.rear_lidar_threshold = self.get_parameter('lidar_rear_panic_threshold').value
        self.delta_w_threshold = self.get_parameter('action_delta_w_threshold').value
        self.delta_v_threshold = self.get_parameter('action_delta_v_threshold').value
        self.min_turn_to_compare = self.get_parameter('min_turn_to_compare').value
        check_hz             = self.get_parameter('check_hz').value

        #___________State storages________________________
        self.latest_scan     = None
        self.latest_rear_scan = None
        self.latest_bc_v     = None
        self.latest_bc_w     = None     
        self.latest_mppi_v   = None
        self.latest_mppi_w   = None    
        self.tracked_goal_id = None
        

        #____________Qos______________________________
        sensor_qos      = QoSProfile(
            reliability = ReliabilityPolicy.BEST_EFFORT,
            durability  = DurabilityPolicy.VOLATILE,
            depth       = 5
        )

        #____________Subscribers____________________________
        self.scan_sub  = self.create_subscription(LaserScan, '/scan', self.scan_callback, qos_profile=sensor_qos)
        
        self.bc_sub   = self.create_subscription(Twist, '/cmd_vel_bc', self.bc_callback, 10)
        self.mppi_sub = self.create_subscription(Twist, '/cmd_vel_mppi', self.mppi_callback, 10)
        self.status_sub = self.create_subscription(GoalStatusArray, '/navigate_to_pose/_action/status', self.status_callback, 10)

        #___________Publishers_________________________________
        self.safety_pub = self.create_publisher(Bool, '/safety_status', 10)
        self.reason_pub = self.create_publisher(String, '/trigger_reason', 10) 

        #___________Evaluation Timer_________________________
        self.timer = self.create_timer(1.0 / check_hz, self.safety_check_loop)

        self.get_logger().info(f'Safety Monitor Active. Limits -> w_Delta: {self.delta_w_threshold} rad/s, v_Delta: {self.delta_v_threshold} m/s, LiDAR: {self.lidar_threshold}m')

    def scan_callback(self, msg: LaserScan):
        ranges = np.array(msg.ranges, dtype=np.float32)
        valid_ranges = ranges[np.isfinite(ranges) & (ranges > 0.0)]

        if len(valid_ranges) > 0:
            self.latest_scan = np.min(valid_ranges) 
            
            # Rear rays are typically around 180 degrees (indices 150 to 210 in waffle scan of 360 rays)
            num_rays = len(msg.ranges)
            idx_150 = int(150 / 360 * num_rays)
            idx_210 = int(210 / 360 * num_rays)
            rear_ranges = ranges[idx_150:idx_210]
            valid_rear = rear_ranges[np.isfinite(rear_ranges) & (rear_ranges > 0.0)]
            self.latest_rear_scan = np.min(valid_rear) if len(valid_rear) > 0 else float('inf')
        else:
            self.latest_scan = float('inf')
            self.latest_rear_scan = float('inf')

    def bc_callback(self, msg: Twist):
        self.latest_bc_v = msg.linear.x
        self.latest_bc_w = msg.angular.z

    def mppi_callback(self, msg: Twist):
        self.latest_mppi_v = msg.linear.x
        self.latest_mppi_w = msg.angular.z

    def status_callback(self, msg: GoalStatusArray):
        if not msg.status_list:
            return

        # If we are not tracking any goal, look for an active one (status 1 = Accepted, 2 = Executing)
        if self.tracked_goal_id is None:
            for status_item in msg.status_list:
                if status_item.status in [1, 2]:
                    self.tracked_goal_id = tuple(int(x) for x in status_item.goal_info.goal_id.uuid)
                    self.get_logger().info(f"New Nav2 Goal Accepted! Tracking Goal ID in Safety Monitor: {self.tracked_goal_id}")
                    break

        # If we are tracking a goal, monitor its lifecycle
        if self.tracked_goal_id is not None:
            tracked_item = None
            for status_item in msg.status_list:
                if tuple(int(x) for x in status_item.goal_info.goal_id.uuid) == self.tracked_goal_id:
                    tracked_item = status_item
                    break

            if tracked_item is not None:
                status_code = tracked_item.status
                if status_code in [4, 5, 6]:
                    self.get_logger().info(f"Tracked Goal Ended (Status: {status_code}). Resetting Safety Monitor stale values.")
                    # Reset values to None to prevent safety checks using stale velocities
                    self.latest_bc_v = None
                    self.latest_bc_w = None
                    self.latest_mppi_v = None
                    self.latest_mppi_w = None
                    self.latest_scan = None
                    self.latest_rear_scan = None
                    self.tracked_goal_id = None

    def safety_check_loop(self):
        if (self.tracked_goal_id is None or
            self.latest_scan is None or 
            self.latest_bc_w is None or 
            self.latest_mppi_w is None or 
            self.latest_bc_v is None or 
            self.latest_mppi_v is None):
            # Publish safe status by default when not navigating
            msg = Bool()
            msg.data = False
            self.safety_pub.publish(msg)
            return
        
        risk_detected = False
        trigger_reason = ""

        # 1. CHECK LIDAR PANIC (Any direction)
        if self.latest_scan < self.lidar_threshold:
            risk_detected = True
            trigger_reason = f"LIDAR PANIC ({self.latest_scan:.2f}m < {self.lidar_threshold}m)"

        # 2. CHECK REVERSING COLLISION RISK (Check obstacles behind robot if BC is reversing)
        elif self.latest_bc_v < -0.01 and self.latest_rear_scan < self.rear_lidar_threshold:
            risk_detected = True
            trigger_reason = f"REVERSING RISK (v={self.latest_bc_v:.2f} m/s, rear={self.latest_rear_scan:.2f}m < {self.rear_lidar_threshold}m)"

        # 3. CHECK EARLY WARNING (Inflation Breach)
        elif self.latest_scan < self.warning_threshold:
            risk_detected = True
            trigger_reason = f"INFLATION BREACH ({self.latest_scan:.2f}m < {self.warning_threshold}m)"

        # 4. CHECK BRAIN DISAGREEMENT (Action Delta & Sign Mismatch & Linear Velocity Disagreement)
        else:
            delta_w = abs(self.latest_mppi_w - self.latest_bc_w)
            delta_v = abs(self.latest_mppi_v - self.latest_bc_v)
            
            # Evaluate topological disagreement
            opposite_sign = (
                np.sign(self.latest_mppi_w) != np.sign(self.latest_bc_w)
                and abs(self.latest_mppi_w) > self.min_turn_to_compare
                and abs(self.latest_bc_w) > self.min_turn_to_compare
            )

            if delta_w > self.delta_w_threshold:
                risk_detected = True
                trigger_reason = f"POLICY ANGLE DISAGREEMENT (Delta W: {delta_w:.2f} > {self.delta_w_threshold})"
            elif delta_v > self.delta_v_threshold:
                risk_detected = True
                trigger_reason = f"POLICY SPEED DISAGREEMENT (Delta V: {delta_v:.2f} > {self.delta_v_threshold})"
            elif opposite_sign:
                risk_detected = True
                trigger_reason = f"TURN SIGN MISMATCH (BC={self.latest_bc_w:.2f}, MPPI={self.latest_mppi_w:.2f})"

        # --- PUBLISH STATUS ---
        msg = Bool()
        msg.data = risk_detected
        self.safety_pub.publish(msg)

        # --- PUBLISH REASON AND LOG ---
        if risk_detected:
            reason_msg = String()
            reason_msg.data = trigger_reason
            self.reason_pub.publish(reason_msg)
            self.get_logger().warn(f"SHIELD TRIGGERED: {trigger_reason}", throttle_duration_sec=0.5)

def main(args=None):
    rclpy.init(args=args)
    node = SafetyMonitorNodeV2()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()