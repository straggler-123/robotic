#!/usr/bin/env python3
"""
表面力控制仿真 - 圆形擦拭任务（Pinocchio 修正版）

修复点：
1. Pinocchio 关节映射显式绑定 panda_joint1~7。
2. MuJoCo 状态 body 与 Pinocchio end-effector frame 默认统一为 panda_hand。

说明：
- 这个版本只修“关节映射 + 末端点一致性”。
- 如果 M / h / Lambda 仍与 MuJoCo 相差较大，主要原因通常是 XML 和 URDF 的惯量参数不一致。
"""

import mujoco
import mujoco.viewer
import numpy as np
from scipy.spatial.transform import Rotation
from dynamics_calculator_pinocchio_fixed import DynamicsCalculatorPinocchio


class WipeTableSimulationPinocchio:
    def __init__(
        self,
        mujoco_model_path: str = "surface_force_control_disk.xml",
        pinocchio_urdf_path: str = "panda_arm.urdf",
        mujoco_ee_body: str = "panda_hand",
        end_effector_frame: str = "panda_hand",
    ):
        self.model = mujoco.MjModel.from_xml_path(mujoco_model_path)
        self.data = mujoco.MjData(self.model)
        self.joint_names = [f"panda_joint{i + 1}" for i in range(7)]
        self.n_joints = 7
        self.mujoco_ee_body = mujoco_ee_body
        self.pinocchio_ee_frame = end_effector_frame

        self.joint_dof_indices = [self.model.joint(name).id for name in self.joint_names]
        self.dof_adr = [self.model.joint(idx).dofadr[0] for idx in self.joint_dof_indices]

        self.dynamics_calc = DynamicsCalculatorPinocchio(
            urdf_path=pinocchio_urdf_path,
            end_effector_frame=end_effector_frame,
            joint_names=self.joint_names,
        )

        q_init = np.array([0, -0.785, 0, -2.356, 0, 1.571, 0.785])
        for i, adr in enumerate(self.dof_adr):
            self.data.qpos[adr] = q_init[i]
        mujoco.mj_forward(self.model, self.data)

        self.setup_control_parameters()
        self.X_e_integral = np.zeros(6)
        self.F_e_integral = np.zeros(6)
        self.debug_counter = 0
        self.filtered_wrench = np.zeros(6)
        self.force_alpha = 0.1
        self.start_time = 0.0
        self.tau_max = 87.0
        self.tau_min = -87.0
        self.tau_warning_counter = 0
        self.tau_warning_interval = 100

        try:
            self.sensor_id_force = self.model.sensor("force_sensor_force").id
            self.sensor_id_torque = self.model.sensor("force_sensor_torque").id
            print("✓ 6维力传感器初始化成功")
            print(f"  Force传感器ID: {self.sensor_id_force}")
            print(f"  Torque传感器ID: {self.sensor_id_torque}")
        except (AttributeError, KeyError) as e:
            print(f"⚠ 力传感器未找到: {e}")
            self.sensor_id_force = None
            self.sensor_id_torque = None

        print("✓ 混合力位控制仿真（Pinocchio修正版）初始化完成")
        print(f"  MuJoCo 末端 body: {self.mujoco_ee_body}")
        print(f"  Pinocchio 末端 frame: {self.pinocchio_ee_frame}")

    def setup_control_parameters(self):
        self.K_p = np.diag([20.0, 20.0, 20.0, 100.0, 100.0, 100.0]) * 5
        self.K_d = np.diag([2.0, 2.0, 2.0, 10.0, 10.0, 10.0]) * 5
        self.K_i = np.diag([0.1, 0.1, 0.1, 1.0, 1.0, 1.0])
        self.K_fp = np.diag([0.5, 0.5, 0.5, 0.5, 0.5, 0.5]) * 2
        self.K_fi = np.diag([0.01, 0.01, 0.01, 0.01, 0.01, 0.01])
        self.F_desired_val = -30.0

    def compute_spatial_error(self, p, q, p_d, q_d):
        qc = np.roll(q, -1)
        qd = np.roll(q_d, -1)
        Rc = Rotation.from_quat(qc).as_matrix()
        Rd = Rotation.from_quat(qd).as_matrix()
        R_err = Rd @ Rc.T
        omega_e = Rotation.from_matrix(R_err).as_rotvec()
        pos_e = p_d - Rd @ Rc.T @ p
        return np.concatenate([omega_e, pos_e])

    def compute_projection_matrix(self, Lambda_s, A_s):
        try:
            Lambda_inv = np.linalg.inv(Lambda_s)
            inner = A_s @ Lambda_inv @ A_s.T
            inner_inv = np.linalg.inv(inner + 1e-6 * np.eye(A_s.shape[0]))
            P = np.eye(6) - A_s.T @ inner_inv @ A_s @ Lambda_inv
            return P
        except np.linalg.LinAlgError:
            return np.eye(6)

    def compute_adjoint_transformation(self, R, p):
        p_skew = np.array([
            [0, -p[2], p[1]],
            [p[2], 0, -p[0]],
            [-p[1], p[0], 0],
        ])
        Ad = np.zeros((6, 6))
        Ad[:3, :3] = R
        Ad[3:, 3:] = R
        Ad[3:, :3] = p_skew @ R
        return Ad

    def get_filtered_contact_wrench(self):
        if self.sensor_id_force is None or self.sensor_id_torque is None:
            return np.zeros(6)

        force_raw = self.data.sensordata[self.sensor_id_force:self.sensor_id_force + 3]
        torque_raw = self.data.sensordata[self.sensor_id_torque * 3:self.sensor_id_torque * 3 + 3]
        wrench_raw = np.concatenate([torque_raw, force_raw])
        wrench = -wrench_raw
        self.filtered_wrench = self.force_alpha * wrench + (1 - self.force_alpha) * self.filtered_wrench
        return self.filtered_wrench

    def generate_wipe_trajectory(self, t):
        center = np.array([0.55, 0.0])
        radius = 0.02
        freq = 1.0
        quat_d = Rotation.from_euler('xyz', [3.14159, 0, 0]).as_quat()
        quat_d = np.array([quat_d[3], quat_d[0], quat_d[1], quat_d[2]])

        if t < 2.0:
            pos_d = np.array([0.55, 0.0, 0.3])
            mode = "APPROACH"
        elif t < 4.0:
            alpha = (t - 2.0) / 2.0
            z_target = 0.3 * (1 - alpha) + 0.14 * alpha
            pos_d = np.array([0.5, 0.0, z_target])
            mode = "DESCEND"
        else:
            time_circle = t - 4.0
            x = center[0] + radius * np.cos(freq * time_circle)
            y = center[1] + radius * np.sin(freq * time_circle)
            pos_d = np.array([x, y, 0.14])
            mode = "WIPE"
        return pos_d, quat_d, mode

    def control_step(self, t, dt):
        q = np.array([self.data.qpos[adr] for adr in self.dof_adr])
        qdot = np.array([self.data.qvel[adr] for adr in self.dof_adr])
        pos = np.array(self.data.body(self.mujoco_ee_body).xpos)
        quat = np.array(self.data.body(self.mujoco_ee_body).xquat)

        J_s = self.dynamics_calc.compute_spatial_jacobian(q, 6)
        V_s = J_s @ qdot
        Lambda_s = self.dynamics_calc.compute_task_space_mass_matrix(q, 6)
        eta_s = self.dynamics_calc.compute_task_space_coriolis(q, V_s, 6)

        wrench_curr = self.get_filtered_contact_wrench()
        f_z_curr = wrench_curr[5]
        is_contact = abs(f_z_curr) > 10

        pos_d, quat_d, mode = self.generate_wipe_trajectory(t)
        enable_force_control = is_contact and (mode in ["DESCEND", "WIPE"])

        if enable_force_control:
            A_s = np.zeros((4, 6))
            A_s[0, 0] = 1.0
            A_s[1, 1] = 1.0
            A_s[2, 2] = 0.0  # 保持原逻辑，避免同时改太多控制行为
            A_s[3, 5] = 1.0
            P_s = self.compute_projection_matrix(Lambda_s, A_s)
        else:
            P_s = np.eye(6)

        X_e = self.compute_spatial_error(pos, quat, pos_d, quat_d)
        V_e = -V_s
        self.X_e_integral += X_e * dt
        self.X_e_integral = np.clip(self.X_e_integral, -0.5, 0.5)

        F_motion = P_s @ Lambda_s @ (self.K_p @ X_e + self.K_d @ V_e + self.K_i @ self.X_e_integral)

        F_force = np.zeros(6)
        if enable_force_control:
            site_pos_world = np.array(self.data.site("ft_sensor_site").xpos)
            site_mat_world = np.array(self.data.site("ft_sensor_site").xmat).reshape(3, 3)

            T_world_to_site = np.eye(4)
            T_world_to_site[:3, :3] = site_mat_world
            T_world_to_site[:3, 3] = site_pos_world
            T_site_to_world = np.linalg.inv(T_world_to_site)

            R_site_to_world = T_site_to_world[:3, :3]
            p_site_in_world = T_site_to_world[:3, 3]
            Ad_site_to_world = self.compute_adjoint_transformation(R_site_to_world, p_site_in_world)

            F_d_site = np.zeros(6)
            F_d_site[5] = self.F_desired_val

            F_e_world = Ad_site_to_world.T @ (F_d_site - wrench_curr * 0.001)
            self.F_e_integral += F_e_world * dt
            self.F_e_integral = np.clip(self.F_e_integral, -50.0, 50.0)

            F_d_world = Ad_site_to_world.T @ F_d_site
            F_cmd_world = F_d_world - self.K_fp @ F_e_world - self.K_fi @ self.F_e_integral
            F_force = (np.eye(6) - P_s) @ F_cmd_world

        F_total = F_motion + F_force + eta_s
        tau = J_s.T @ F_total

        tau_null = -2.0 * qdot
        tau += (np.eye(7) - J_s.T @ np.linalg.pinv(J_s.T)) @ tau_null

        tau_original = tau.copy()
        tau = np.clip(tau, self.tau_min, self.tau_max)

        exceeded_mask = (tau_original < self.tau_min) | (tau_original > self.tau_max)
        if np.any(exceeded_mask):
            self.tau_warning_counter += 1
            if self.tau_warning_counter >= self.tau_warning_interval:
                exceeded_joints = np.where(exceeded_mask)[0]
                for joint_idx in exceeded_joints:
                    joint_name = self.joint_names[joint_idx]
                    tau_original_val = tau_original[joint_idx]
                    tau_clipped_val = tau[joint_idx]
                    if tau_original_val > self.tau_max:
                        print(
                            f"⚠️  力矩超限警告 [t={t:.2f}s]: 关节 {joint_name} (索引 {joint_idx}) "
                            f"力矩 {tau_original_val:.2f} N·m 超出上限 {self.tau_max:.2f} N·m, "
                            f"已限制为 {tau_clipped_val:.2f} N·m"
                        )
                    elif tau_original_val < self.tau_min:
                        print(
                            f"⚠️  力矩超限警告 [t={t:.2f}s]: 关节 {joint_name} (索引 {joint_idx}) "
                            f"力矩 {tau_original_val:.2f} N·m 超出下限 {self.tau_min:.2f} N·m, "
                            f"已限制为 {tau_clipped_val:.2f} N·m"
                        )
                self.tau_warning_counter = 0

        self.data.ctrl[:] = tau

    def run(self):
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            while viewer.is_running():
                dt = self.model.opt.timestep
                self.control_step(self.data.time, dt)
                mujoco.mj_step(self.model, self.data)
                viewer.sync()


if __name__ == "__main__":
    print("=" * 60)
    print("混合力位控制仿真 - Pinocchio 动力学计算版本（修正版）")
    print("=" * 60)
    sim = WipeTableSimulationPinocchio(
        mujoco_model_path="surface_force_control_disk.xml",
        pinocchio_urdf_path="panda_arm.urdf",
        mujoco_ee_body="panda_hand",
        end_effector_frame="panda_hand",
    )
    sim.run()
