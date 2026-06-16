import rclpy
from   rclpy.node       import Node
from   geometry_msgs.msg  import Twist
from   std_msgs.msg       import Bool, String


class ArbitrationNode(Node):
    def __init__(self):
        super().__init__('arbitration_node')
    
        #___________Parameter______________________
        #Timer Hysteresis: How long to wait after the last detected risk before handing control back to BC?
        self.declare_parameter('recovery_duration', 3.0) 
        self.recovery_duration = self.get_parameter('recovery_duration').value

        #___________State Variable____________________
        self.mode              = 'BC_POLICY'
        self.previous_mode     = 'BC_POLICY'
        self.recovery_end_time = 0.0
        
        #___________Deadlock State Variables_____________
        self.expert_start_time   = 0.0
        self.deadlock_timeout    = 12.0 #4.0 8.0
        self.stuck_vel_threshold = 0.015 #0.02
        self.latest_mppi_v       = 0.0
        self.latest_mppi_w       = 0.0
       

        #___________Subscriber________________________
        self.bc_sub     = self.create_subscription(Twist, '/cmd_vel_bc', self.bc_callback, 10)
        self.mppi_sub   = self.create_subscription(Twist, '/cmd_vel_mppi', self.mppi_callback, 10)
        self.safety_sub = self.create_subscription(Bool, '/safety_status', self.safety_callback, 10)

        #___________Publisher_________________________
        self.cmd_pub  =  self.create_publisher(Twist, '/cmd_vel', 10)
        self.mode_pub =  self.create_publisher(String, '/active_mode', 10)

        # Announce initial state immediately
        self.publish_mode()
        self.get_logger().info("Arbitration Node Active. Defaulting to BC_POLICY.")

    def publish_mode(self):
        msg = String()
        msg.data = self.mode
        self.mode_pub.publish(msg)

    def safety_callback(self, msg: Bool):
        is_risk_detected = msg.data

        current_time = self.get_clock().now().nanoseconds / 1e9 #get time in seconds

        if is_risk_detected:
            #Continuously push the hysteresis timer forward as long as danger exists.
            # The countdown will only begin once is_risk_detected becomes False.
            self.recovery_end_time = current_time + self.recovery_duration

            if self.mode == 'EXPERT':
                # Evaluate Deadlock: Has it been too long AND is the expert paralyzed?
                time_in_expert = current_time - self.expert_start_time
                is_stuck = (abs(self.latest_mppi_v) < self.stuck_vel_threshold 
                        and abs(self.latest_mppi_w) < self.stuck_vel_threshold)
                
                if time_in_expert > self.deadlock_timeout and is_stuck:
                    self.get_logger().error("!!! EXPERT DEADLOCK DETECTED (0 vel). Forcing BC Handoff !!!")
                    self.mode = "BC_POLICY"
                    self.recovery_end_time = current_time + 1.5
            
            else:
                # SHIELD TRIGGERED! Lock into EXPERT mode and reset the countdown timer.
                self.mode = 'EXPERT'
                self.expert_start_time = current_time
        else:
            # Kinematic Hysteresis: sure robot is not longer turning, and is moving forward, before handing control back to BC.
            maneuver_complete = (
                abs(self.latest_mppi_w) < 0.15
                and self.latest_mppi_v > 0.05  # Make sure MPPI is moving forward, not deadlocked
            )

            # No risk currently detected. Check if our recovery timer has expired.
            if current_time > self.recovery_end_time and maneuver_complete:
                self.mode = "BC_POLICY"

        if self.mode != self.previous_mode:
            if self.mode == 'EXPERT':
                self.get_logger().warn(">>> TAKEOVER: MPPI Expert is executing recovery! <<<")
            else:
                self.get_logger().info("<<< HANDOFF: BC Policy has resumed control. >>>")
            self.publish_mode()
            self.previous_mode = self.mode
    
    def bc_callback(self, msg: Twist):
        if self.mode == 'BC_POLICY':
            self.cmd_pub.publish(msg)

    
    def mppi_callback(self, msg: Twist):
        # Always track what the expert wants to do
        self.latest_mppi_v = msg.linear.x
        self.latest_mppi_w = msg.angular.z

        if self.mode == 'EXPERT':
            self.cmd_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = ArbitrationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        stop = Twist()
        node.cmd_pub.publish(stop)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()