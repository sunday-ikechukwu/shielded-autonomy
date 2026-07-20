import rclpy
from   rclpy.node       import Node
from   geometry_msgs.msg  import Twist
from   std_msgs.msg       import Bool, String
from   action_msgs.msg    import GoalStatusArray

from   action_msgs.srv    import CancelGoal
from   unique_identifier_msgs.msg import UUID


class ArbitrationNodeV2(Node):
    def __init__(self):
        super().__init__('arbitration_node_v2')
    
        #___________Parameter______________________
        #Timer Hysteresis: How long to wait after the last detected risk before handing control back to BC?
        self.declare_parameter('recovery_duration', 3.0) 
        self.recovery_duration = self.get_parameter('recovery_duration').value

        #___________State Variable____________________
        self.mode              = 'BC_POLICY'
        self.previous_mode     = 'BC_POLICY'
        self.recovery_end_time = 0.0
        self.bc_recovery_end_time = 0.0
        self.tracked_goal_id   = None
        
        #___________Deadlock State Variables_____________
        # self.expert_start_time   = 0.0
        self.expert_stuck_start_time = None
        self.deadlock_timeout    = 4.0 #12.0  8.0
        self.stuck_vel_threshold = 0.015 #0.02
        self.latest_mppi_v       = 0.0
        self.latest_mppi_w       = 0.0
       

        #___________Subscriber________________________
        self.bc_sub     = self.create_subscription(Twist, '/cmd_vel_bc', self.bc_callback, 10)
        self.mppi_sub   = self.create_subscription(Twist, '/cmd_vel_mppi', self.mppi_callback, 10)
        self.safety_sub = self.create_subscription(Bool, '/safety_status', self.safety_callback, 10)
        self.status_sub = self.create_subscription(GoalStatusArray, '/navigate_to_pose/_action/status', self.status_callback, 10)

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
        ##ADDED
        if self.tracked_goal_id is None:
            # If not tracking any goal, ensure we stay in BC_POLICY and reset state variables
            self.mode = 'BC_POLICY'
            self.previous_mode = 'BC_POLICY'
            self.expert_stuck_start_time = None
            self.bc_recovery_end_time = 0.0
            return

        is_risk_detected = msg.data
        current_time = self.get_clock().now().nanoseconds / 1e9 #get time in seconds

        if is_risk_detected:
            #Continuously push the hysteresis timer forward as long as danger exists.
            # The countdown will only begin once is_risk_detected becomes False.
            self.recovery_end_time = current_time + self.recovery_duration

            ##ADDED
            if self.mode == 'BC_RECOVERY':
                # During recovery mode, we ignore new takeover requests and let BC run for 1.5s
                if current_time > self.bc_recovery_end_time:
                    # After 1.5s, check if we successfully escaped deadlock (mppi velocity is non-zero)
                    is_still_stuck = (abs(self.latest_mppi_v) < self.stuck_vel_threshold 
                                      and abs(self.latest_mppi_w) < self.stuck_vel_threshold)
                    if is_still_stuck:
                        self.get_logger().error("!!! BC Recovery failed to break deadlock. Cancelling active goal request !!!")
                        self.cancel_current_goal()
                        stop_cmd = Twist()
                        self.cmd_pub.publish(stop_cmd)
                        self.mode = 'BC_POLICY'
                    else:
                        self.get_logger().info("<<< BC Recovery broke the deadlock. Returning to EXPERT recovery mode. >>>")
                        self.mode = 'EXPERT'
                        self.expert_stuck_start_time = None


            elif self.mode == 'EXPERT':
                # Evaluate Deadlock: Has it been too long AND is the expert paralyzed?
                # time_in_expert = current_time - self.expert_start_time
                is_stuck = (abs(self.latest_mppi_v) < self.stuck_vel_threshold 
                        and abs(self.latest_mppi_w) < self.stuck_vel_threshold)
                
                # if time_in_expert > self.deadlock_timeout and is_stuck:
                #     self.get_logger().error("!!! EXPERT DEADLOCK DETECTED (0 vel). Forcing BC Handoff !!!")
                #     self.mode = "BC_POLICY"
                #     self.recovery_end_time = current_time + 1.5
                
                if is_stuck:
                    if self.expert_stuck_start_time is None:
                        self.expert_stuck_start_time = current_time

                    # elif current_time - self.expert_stuck_start_time > self.deadlock_timeout:
                    #     self.get_logger().error("!!! EXPERT DEADLOCK DETECTED (0 vel). Forcing BC Handoff !!!")
                    #     self.mode = "BC_POLICY"
                    #     self.recovery_end_time = current_time + 1.5
                    #     self.expert_stuck_start_time = None
                    elif current_time - self.expert_stuck_start_time > self.deadlock_timeout:
                        # self.get_logger().error("!!! EXPERT DEADLOCK DETECTED (0 vel). Cancelling active goal request !!!")
                        # self.cancel_current_goal()
                        
                        # # Stop the robot immediately
                        # stop_cmd = Twist()
                        # self.cmd_pub.publish(stop_cmd)
                        
                        # self.expert_stuck_start_time = None
                        ###ADDED
                        self.get_logger().error("!!! EXPERT DEADLOCK DETECTED (0 vel). Switching to BC Recovery (Shake-Up) !!!")
                        self.mode = 'BC_RECOVERY'
                        self.bc_recovery_end_time = current_time + 3 #1.5
                        self.expert_stuck_start_time = None

                else:
                    self.expert_stuck_start_time = None
            
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

            ###ADDED
            if self.mode == 'BC_RECOVERY':
                # If risk cleared during BC recovery, we can switch back to BC_POLICY!
                is_still_stuck = (abs(self.latest_mppi_v) < self.stuck_vel_threshold 
                                  and abs(self.latest_mppi_w) < self.stuck_vel_threshold)
                if not is_still_stuck or current_time > self.bc_recovery_end_time:
                    self.mode = "BC_POLICY"
                    self.get_logger().info("<<< BC Recovery cleared the risk/deadlock. Resuming normal BC_POLICY control. >>>")
            

            # No risk currently detected. Check if our recovery timer has expired.
            elif current_time > self.recovery_end_time and maneuver_complete:
                self.mode = "BC_POLICY"

        if self.mode != self.previous_mode:
            if self.mode == 'EXPERT':
                self.get_logger().warn(">>> TAKEOVER: MPPI Expert is executing recovery! <<<")
            ##ADDED
            elif self.mode == 'BC_RECOVERY':
                self.get_logger().warn(">>> RECOVERY SHAKE-UP: Forcing BC Policy to break deadlock! <<<")
            else:
                self.get_logger().info("<<< HANDOFF: BC Policy has resumed control. >>>")
            
            self.publish_mode()
            self.previous_mode = self.mode
            
    def status_callback(self, msg: GoalStatusArray):
        if not msg.status_list:
            return

        # If we are not tracking any goal, look for an active one (status 1 = Accepted, 2 = Executing)
        if self.tracked_goal_id is None:
            for status_item in msg.status_list:
                if status_item.status in [1, 2]:
                    # self.tracked_goal_id = tuple(status_item.goal_info.goal_id.uuid)
                    self.tracked_goal_id = tuple(int(x) for x in status_item.goal_info.goal_id.uuid)
                    self.get_logger().info(f"✅ New Nav2 Goal Accepted! Tracking Goal ID in Arbitrator: {self.tracked_goal_id}")
                    break

        # If we are tracking a goal, monitor its lifecycle
        if self.tracked_goal_id is not None:
            tracked_item = None
            for status_item in msg.status_list:
                # if tuple(status_item.goal_info.goal_id.uuid) == self.tracked_goal_id:
                if tuple(int(x) for x in status_item.goal_info.goal_id.uuid) == self.tracked_goal_id:
                    tracked_item = status_item
                    break

            if tracked_item is not None:
                status_code = tracked_item.status
                if status_code in [4, 5, 6]:
                    self.get_logger().info(f"🛑 Tracked Goal Ended (Status: {status_code}). Resetting Arbitrator to BC_POLICY and clearing stale expert velocities.")
                    # Reset Mode and expert velocity state variables to clear stale data
                    self.mode = 'BC_POLICY'
                    self.previous_mode = 'BC_POLICY'
                    self.recovery_end_time = 0.0
                    self.bc_recovery_end_time = 0.0
                    self.latest_mppi_v = 0.0
                    self.latest_mppi_w = 0.0
                    self.publish_mode()
                    self.tracked_goal_id = None
                    self.expert_stuck_start_time = None
    
    def cancel_current_goal(self):
        if self.tracked_goal_id is None:
            return

        self.get_logger().info("Sending cancel request for the active goal...")
        
        if not hasattr(self, 'cancel_client'):
            self.cancel_client = self.create_client(CancelGoal, '/navigate_to_pose/_action/cancel_goal')

        if not self.cancel_client.service_is_ready():
            self.get_logger().warn("Cancel goal service not ready, waiting...")
            self.cancel_client.wait_for_service(timeout_sec=1.0)

        if self.cancel_client.service_is_ready():
            req = CancelGoal.Request()
            # req.goal_info.goal_id.uuid = list(self.tracked_goal_id)
            req.goal_info.goal_id.uuid = [int(x) for x in self.tracked_goal_id]
            self.cancel_client.call_async(req)
            self.get_logger().info("Cancel request sent successfully.")
        else:
            self.get_logger().error("Failed to call cancel service: service not available.")

    
    def bc_callback(self, msg: Twist):
        # if self.mode == 'BC_POLICY':
        if self.mode in ['BC_POLICY', 'BC_RECOVERY']:
            self.cmd_pub.publish(msg)

    
    def mppi_callback(self, msg: Twist):
        # Always track what the expert wants to do
        self.latest_mppi_v = msg.linear.x
        self.latest_mppi_w = msg.angular.z

        if self.mode == 'EXPERT':
            self.cmd_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = ArbitrationNodeV2()
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