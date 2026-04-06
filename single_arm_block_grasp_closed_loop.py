#!/usr/bin/env python3
import mujoco
import mujoco.viewer
import numpy as np
from scipy.spatial.transform import Rotation
from dynamics_calculator_wv import DynamicsCalculator


class SingleArmBlockGraspClosedLoopV2:
    def __init__(self, model_path="single_arm_block_scene_closed_loop.xml"):
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)

        self.arm_joint_names = [f"panda_joint{i+1}" for i in range(7)]
        self.arm_dof_adr = [self.model.joint(name).dofadr[0] for name in self.arm_joint_names]

        # self.ee_body_name = "panda_hand"
        self.ee_body_name = "panda_grasp_center"
        self.block_body_name = "block_body"
        self.block_geom_name = "block_geom"
        self.surface_body_name = "work_surface"
        self.surface_geom_name = "surface"

        self.finger_act_id = self.model.actuator("panda_actuator_finger").id
        self.finger_qpos_adr = [
            self.model.joint("panda_finger_joint1").qposadr[0],
            self.model.joint("panda_finger_joint2").qposadr[0],
        ]

        self.dynamics = DynamicsCalculator(model_path, self.ee_body_name, self.arm_joint_names)

        self.q_init = np.array([0.0, -0.65, 0.0, -2.10, 0.0, 1.65, 0.78])
        for i, adr in enumerate(self.arm_dof_adr):
            self.data.qpos[adr] = self.q_init[i]
        mujoco.mj_forward(self.model, self.data)

        self.data.ctrl[self.finger_act_id] = 10.0
        for _ in range(200):
            mujoco.mj_step(self.model, self.data)

        self.home_quat = self.data.body(self.ee_body_name).xquat.copy()

        self.K_p = np.diag([120.0, 120.0, 140.0, 220.0, 220.0, 220.0])
        self.K_d = np.diag([24.0, 24.0, 28.0, 36.0, 36.0, 36.0])

        self.block_half_size = self.model.geom(self.block_geom_name).size.copy()
        self.block_half_height = float(self.block_half_size[2])

        self.hand_to_grasp_center_z = 0.001
        # self.hand_to_grasp_center_z = 0.085

        # =========================
        # 新增：panda_hand 的 XY 偏置（单位：米）
        # 例如：x 正方向偏 1 cm，y 负方向偏 5 mm
        # 按你的实际情况改这里
        # =========================
        self.hand_xy_bias = np.array([0.007, -0.001])
        # self.hand_xy_bias = np.array([-0.01, -0.005])


        self.pregrasp_above_center = 0.120
        self.lift_target_z = 0.34
        self.move_target_xyz = np.array([0.38, -0.15, 0.32])
        self.release_target_xyz = np.array([0.38, -0.15, 0.24])

        self.pregrasp_pos_tol = 0.012
        self.descend_xy_tol = 0.006
        self.descend_z_tol = 0.008
        self.move_pos_tol = 0.015
        self.release_pos_tol = 0.010

        self.finger_open_cmd = 10.0
        self.finger_close_cmd = -20.0
        self.finger_hold_cmd = -18.0
        self.force_target_close = 8.0
        self.force_target_hold = 6.0
        self.force_kp = 1.6

        self.secure_gap_low = 0.010
        self.secure_gap_high = 0.050
        self.contact_force_threshold = 1.0
        self.secure_force_threshold = 4.0

        self.state = "pregrasp"
        self.state_enter_time = 0.0
        self.last_print_time = -1.0
        self.block_init_z = self.block_pos()[2]

        self.target_pos = self.compute_pregrasp_target()
        self.target_quat = self.home_quat.copy()

        self.retry_counter = 0
        self.max_retries = 20

    def ee_pos(self):
        return self.data.body(self.ee_body_name).xpos.copy()

    def ee_quat(self):
        return self.data.body(self.ee_body_name).xquat.copy()

    def block_pos(self):
        return self.data.body(self.block_body_name).xpos.copy()

    def finger_gap(self):
        q1 = self.data.qpos[self.finger_qpos_adr[0]]
        q2 = self.data.qpos[self.finger_qpos_adr[1]]
        return float(q1 + q2)

    def finger_forces(self):
        f_l = self.data.sensor("panda_left_finger_force").data.copy()
        f_r = self.data.sensor("panda_right_finger_force").data.copy()
        return f_l, f_r

    def total_contact_force(self):
        f_l, f_r = self.finger_forces()
        return float(np.linalg.norm(f_l) + np.linalg.norm(f_r))

    def time_in_state(self):
        return self.data.time - self.state_enter_time

    def set_state(self, new_state):
        if new_state != self.state:
            print(f"[STATE] {self.state} -> {new_state}")
            self.state = new_state
            self.state_enter_time = self.data.time

    def compute_spatial_error(self, p, q, p_d, q_d):
        qc = np.roll(q, -1)
        qd = np.roll(q_d, -1)
        Rc = Rotation.from_quat(qc).as_matrix()
        Rd = Rotation.from_quat(qd).as_matrix()
        R_err = Rd @ Rc.T
        omega = Rotation.from_matrix(R_err).as_rotvec()
        pos_e = p_d - p
        return np.concatenate([omega, pos_e])

    def surface_top_z(self):
        geom = self.model.geom(self.surface_geom_name)
        body = self.data.body(self.surface_body_name)
        return float(body.xpos[2] + geom.size[2])

    def block_top_z(self):
        return float(self.block_pos()[2] + self.block_half_height)

    # 新增：统一处理 XY 偏置
    def apply_hand_xy_bias(self, pos):
        pos = pos.copy()
        pos[0] += self.hand_xy_bias[0]
        pos[1] += self.hand_xy_bias[1]
        return pos

    def block_center_target(self):
        p = self.block_pos()
        p = self.apply_hand_xy_bias(p)
        return np.array([p[0], p[1], p[2] + self.hand_to_grasp_center_z])

    def compute_pregrasp_target(self):
        p = self.block_pos()
        p = self.apply_hand_xy_bias(p)
        return np.array([p[0], p[1], p[2] + self.hand_to_grasp_center_z + self.pregrasp_above_center])

    def compute_descend_target(self):
        return self.block_center_target()

    def compute_lift_target(self):
        p = self.block_pos()
        p = self.apply_hand_xy_bias(p)
        return np.array([p[0], p[1], self.lift_target_z])

    def block_attached(self):
        bp = self.block_pos()
        ep = self.ee_pos()
        lateral = np.linalg.norm(bp[:2] - ep[:2])
        lifted = bp[2] > (self.surface_top_z() + self.block_half_height + 0.010)
        return lateral < 0.040 and lifted

    def grasp_success(self):
        gap = self.finger_gap()
        total_force = self.total_contact_force()
        bp = self.block_pos()

        grasped_by_gap = self.secure_gap_low < gap < self.secure_gap_high
        grasped_by_force = total_force > self.secure_force_threshold
        grasped_by_motion = bp[2] > (self.surface_top_z() + self.block_half_height + 0.006)
        return (grasped_by_gap and grasped_by_force) or grasped_by_motion

    def compute_gripper_ctrl(self):
        total_force = self.total_contact_force()
        gap = self.finger_gap()

        if self.state in ["pregrasp", "descend", "move", "release_descend", "done"]:
            if self.state == "done":
                return self.finger_open_cmd
            if self.state in ["move", "release_descend"] and self.block_attached():
                cmd = self.finger_hold_cmd + self.force_kp * (self.force_target_hold - total_force)
                return float(np.clip(cmd, -30.0, 30.0))
            return self.finger_open_cmd

        if self.state == "close":
            cmd = self.finger_close_cmd + self.force_kp * (self.force_target_close - total_force)
            return float(np.clip(cmd, -30.0, 30.0))

        if self.state == "lift":
            if gap < self.secure_gap_high or total_force > self.contact_force_threshold:
                cmd = self.finger_hold_cmd + self.force_kp * (self.force_target_hold - total_force)
                return float(np.clip(cmd, -30.0, 30.0))
            return self.finger_close_cmd

        if self.state == "open":
            return self.finger_open_cmd

        return 0.0

    def update_state_machine(self):
        ee = self.ee_pos()
        bp = self.block_pos()

        if self.state == "pregrasp":
            self.target_pos = self.compute_pregrasp_target()
            pos_err = np.linalg.norm(ee - self.target_pos)
            if pos_err < self.pregrasp_pos_tol:
                self.set_state("descend")

        elif self.state == "descend":
            self.target_pos = self.compute_descend_target()
            xy_err = np.linalg.norm(ee[:2] - self.target_pos[:2])
            z_err = abs(ee[2] - self.target_pos[2])
            if xy_err < self.descend_xy_tol and z_err < self.descend_z_tol:
                self.set_state("close")

        elif self.state == "close":
            self.target_pos = self.compute_descend_target()
            if self.grasp_success():
                self.set_state("lift")
            elif self.time_in_state() > 2.0:
                self.retry_counter += 1
                self.set_state("pregrasp")

        elif self.state == "lift":
            self.target_pos = self.compute_lift_target()
            lifted_enough = bp[2] > (self.surface_top_z() + self.block_half_height + 0.035)
            if self.block_attached() and lifted_enough:
                self.set_state("move")
            elif self.time_in_state() > 1.8 and not self.block_attached():
                self.retry_counter += 1
                self.set_state("pregrasp")

        elif self.state == "move":
            self.target_pos = self.move_target_xyz.copy()
            ee_err = np.linalg.norm(ee - self.target_pos)
            if ee_err < self.move_pos_tol:
                self.set_state("release_descend")
            elif self.time_in_state() > 2.0 and not self.block_attached():
                self.retry_counter += 1
                self.set_state("pregrasp")

        elif self.state == "release_descend":
            self.target_pos = self.release_target_xyz.copy()
            ee_err = np.linalg.norm(ee - self.target_pos)
            if ee_err < self.release_pos_tol:
                self.set_state("open")

        elif self.state == "open":
            self.target_pos = self.release_target_xyz.copy()
            if self.time_in_state() > 0.6:
                self.block_init_z = self.block_pos()[2]
                self.set_state("done")

        elif self.state == "done":
            self.target_pos = self.compute_pregrasp_target()

    def control_step(self, dt):
        q = np.array([self.data.qpos[adr] for adr in self.arm_dof_adr])
        qdot = np.array([self.data.qvel[adr] for adr in self.arm_dof_adr])

        self.update_state_machine()

        pos_d = self.target_pos
        quat_d = self.target_quat

        X_e = self.compute_spatial_error(self.ee_pos(), self.ee_quat(), pos_d, quat_d)
        J = self.dynamics.compute_spatial_jacobian(q, 6)
        V = J @ qdot
        Lambda = self.dynamics.compute_task_space_mass_matrix(q, 6)
        F_task = Lambda @ (self.K_p @ X_e + self.K_d @ (-V))
        tau = J.T @ F_task + self.dynamics.compute_coriolis_gravity(q, qdot)

        self.data.ctrl[:7] = tau
        self.data.ctrl[self.finger_act_id] = self.compute_gripper_ctrl()

        if self.data.time - self.last_print_time > 0.2:
            ee_err = np.linalg.norm(self.ee_pos() - self.target_pos)
            bp = self.block_pos()
            print(
                f"t={self.data.time:5.2f} "
                f"state={self.state:>14s} "
                f"ee_err={ee_err:.4f} "
                f"gap={self.finger_gap():.4f} "
                f"f={self.total_contact_force():.2f} "
                f"block_center=({bp[0]:.3f},{bp[1]:.3f},{bp[2]:.3f}) "
                f"block_top_z={self.block_top_z():.4f}"
            )
            self.last_print_time = self.data.time

    def run(self):
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            viewer.cam.trackbodyid = self.model.body(self.ee_body_name).id
            viewer.cam.distance = 1.05
            viewer.cam.elevation = -25
            viewer.cam.azimuth = 135
            while viewer.is_running():
                dt = self.model.opt.timestep
                self.control_step(dt)
                mujoco.mj_step(self.model, self.data)
                viewer.sync()


if __name__ == "__main__":
    sim = SingleArmBlockGraspClosedLoopV2()
    sim.run()