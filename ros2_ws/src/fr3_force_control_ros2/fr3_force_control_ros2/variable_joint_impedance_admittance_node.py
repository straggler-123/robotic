#!/usr/bin/env python3
import threading

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64
from std_msgs.msg import Float64MultiArray
from std_srvs.srv import SetBool
from std_srvs.srv import Trigger

from .controller_core import VariableJointImpedanceAdmittanceController


class VariableJointImpedanceAdmittanceNode(Node):
    def __init__(self):
        super().__init__("variable_joint_impedance_admittance_node")

        self.declare_parameter("model_path", "")
        self.declare_parameter("urdf_path", "")
        self.declare_parameter("ee_body_name", "fr3v2_link7")
        self.declare_parameter("ee_frame_name", "fr3v2_link8")
        self.declare_parameter("motion_mask", [1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
        self.declare_parameter("debug_print", True)
        self.declare_parameter("debug_interval", 0.2)
        self.declare_parameter("publish_rate", 20.0)

        model_path = self.get_parameter("model_path").get_parameter_value().string_value or None
        urdf_path = self.get_parameter("urdf_path").get_parameter_value().string_value or None
        ee_body_name = self.get_parameter("ee_body_name").value
        ee_frame_name = self.get_parameter("ee_frame_name").value
        motion_mask = self.get_parameter("motion_mask").value

        self.controller = VariableJointImpedanceAdmittanceController(
            model_path=model_path,
            urdf_path=urdf_path,
            ee_body_name=ee_body_name,
            ee_frame_name=ee_frame_name,
            motion_mask=motion_mask,
        )
        self.controller.set_reference_from_current()

        self.debug_print = bool(self.get_parameter("debug_print").value)
        self.debug_interval = float(self.get_parameter("debug_interval").value)
        self.last_debug_time = -1e9
        self.lock = threading.Lock()
        self.last_result = None

        self.q_ref_pub = self.create_publisher(Float64MultiArray, "/fr3_force_control/state/q_ref", 10)
        self.kp_pub = self.create_publisher(Float64MultiArray, "/fr3_force_control/state/kp", 10)
        self.kd_pub = self.create_publisher(Float64MultiArray, "/fr3_force_control/state/kd", 10)
        self.wrench_pub = self.create_publisher(Float64MultiArray, "/fr3_force_control/state/wrench", 10)
        self.adm_pos_pub = self.create_publisher(Float64MultiArray, "/fr3_force_control/state/adm_pos", 10)

        self.create_subscription(
            Float64MultiArray,
            "/fr3_force_control/command/motion_mask",
            self.on_motion_mask,
            10,
        )
        self.create_subscription(
            Float64MultiArray,
            "/fr3_force_control/command/admittance_gains",
            self.on_admittance_gains,
            10,
        )
        self.create_subscription(
            Float64,
            "/fr3_force_control/command/debug_interval",
            self.on_debug_interval,
            10,
        )

        self.create_service(Trigger, "/fr3_force_control/reset_reference", self.on_reset_reference)
        self.create_service(SetBool, "/fr3_force_control/set_debug_print", self.on_set_debug_print)

        publish_rate = float(self.get_parameter("publish_rate").value)
        self.timer = self.create_timer(1.0 / publish_rate, self.on_timer)
        self.get_logger().info("FR3 ROS 2 force-control node started.")

    def on_motion_mask(self, msg):
        data = np.asarray(msg.data, dtype=float)
        if data.size != 6:
            self.get_logger().warning("motion_mask expects 6 values: [x, y, z, rx, ry, rz]")
            return
        with self.lock:
            self.controller.set_motion_mask(data)

    def on_admittance_gains(self, msg):
        data = np.asarray(msg.data, dtype=float)
        if data.size != 18:
            self.get_logger().warning("admittance_gains expects 18 values: [M6, D6, K6]")
            return
        with self.lock:
            self.controller.set_admittance_gains(
                mass=data[0:6],
                damping=data[6:12],
                stiffness=data[12:18],
            )

    def on_debug_interval(self, msg):
        self.debug_interval = max(float(msg.data), 0.01)

    def on_reset_reference(self, request, response):
        del request
        with self.lock:
            self.controller.set_reference_from_current()
        response.success = True
        response.message = "Reference reset to current joint state."
        return response

    def on_set_debug_print(self, request, response):
        self.debug_print = bool(request.data)
        response.success = True
        response.message = f"debug_print={self.debug_print}"
        return response

    def publish_array(self, publisher, data):
        msg = Float64MultiArray()
        msg.data = np.asarray(data, dtype=float).tolist()
        publisher.publish(msg)

    def maybe_log_debug(self, result):
        sim_time = float(self.controller.data.time)
        if not self.debug_print:
            return
        if sim_time - self.last_debug_time < self.debug_interval:
            return
        self.last_debug_time = sim_time
        self.get_logger().info(
            "wrench=%s adm_pos=%s q_ref=%s kp=%s kd=%s"
            % (
                np.array2string(result["wrench_task"], precision=3, suppress_small=True),
                np.array2string(result["adm_pos"], precision=4, suppress_small=True),
                np.array2string(result["q_ref"], precision=3, suppress_small=True),
                np.array2string(result["kp"], precision=1, suppress_small=True),
                np.array2string(result["kd"], precision=1, suppress_small=True),
            )
        )

    def on_timer(self):
        with self.lock:
            result = self.controller.step()
            self.last_result = result

        self.publish_array(self.q_ref_pub, result["q_ref"])
        self.publish_array(self.kp_pub, result["kp"])
        self.publish_array(self.kd_pub, result["kd"])
        self.publish_array(self.wrench_pub, result["wrench_task"])
        self.publish_array(self.adm_pos_pub, result["adm_pos"])
        self.maybe_log_debug(result)


def main(args=None):
    rclpy.init(args=args)
    node = VariableJointImpedanceAdmittanceNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
