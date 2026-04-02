#!/usr/bin/env python3
import time
import mujoco
import mujoco.viewer
import numpy as np
from scipy.spatial.transform import Rotation
from dynamics_calculator_wv import DynamicsCalculator


class SingleArmBlockGraspClosedLoop:
    def __init__(self, model_path="single_arm_block_scene_closed_loop.xml"):
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)

        self.joint_names = [f"panda_joint{i+1}" for i in range(7)]
        self.dof_adr = [self.model.joint(name).dofadr[0] for name in self.joint_names]
        self.finger_act_id = self.model.actuator("panda_actuator_finger").id
        self.finger_qpos_adr = [
            self.model.joint("panda_finger_joint1").qposadr[0],
            self.model.joint("panda_finger_joint2").qposadr[0],
        ]

        self.dynamics_calc = DynamicsCalculator(model_path, "panda_hand", self.joint_names)

        # Initial joint posture
        self.q_init = np.array([0.0, -0.65, 0.0, -2.10, 0.0, 1.65, 0.78])
        for i, adr in enumerate(self.dof_adr):
            self.data.qpos[adr] = self.q_init[i]
        # open gripper initially
        self.data.ctrl[self.finger_act_id] = 10.0
        mujoco.mj_forward(self.model, self.data)

        self.home_quat = self.data.body("panda_hand").xquat.copy()

        # OSC gains
        self.K_p = np.diag([260, 260, 260, 140, 140, 140])
        self.K_d = np.diag([34, 34, 34, 20, 20, 20])

        # Mild translational admittance using contact wrench from fingers
        self.M_adm = np.diag([3.0, 3.0, 3.0, 1.0, 1.0, 1.0])
        self.D_adm = np.diag([80.0, 80.0, 80.0, 20.0, 20.0, 20.0])
        self.K_adm = np.diag([0.0, 0.0, 60.0, 0.0, 0.0, 0.0])
        self.adm_pos = np.zeros(6)
        self.adm_vel = np.zeros(6)
        self.max_pos_offset = np.array([0.01, 0.01, 0.02])

        # Task parameters
        self.block_name = "block_body"
        self.surface_name = "work_surface"
        self.pregrasp_height = 0.11
        self.descend_clearance = 0.032
        self.lift_height = 0.16
        self.place_xy = np.array([0.32, -0.18])
        self.place_height = 0.12
        self.release_height = 0.06

        # Gripper closed-loop params
        self.finger_open_cmd = 10.0
        self.finger_close_max = -28.0
        self.finger_hold_cmd = -16.0
        self.force_target_close = 7.0
        self.force_target_hold = 5.5
        self.force_kp = 2.2
        self.close_gap_threshold = 0.018
        self.contact_force_threshold = 1.0
        self.secure_force_threshold = 3.0

        # State machine
        self.state = "pregrasp"
        self.state_enter_time = 0.0
        self.last_print_time = -1.0
        self.block_init_z = self.block_pos()[2]
        self.target_pos = self.compute_pregrasp_target()
        self.target_quat = self.home_quat.copy()
        self.last_valid_grasp_xy = self.block_pos()[:2].copy()

    # ---------- Measurements ----------
    def ee_pos(self):
        return self.data.body("panda_hand").xpos.copy()

    def ee_quat(self):
        return self.data.body("panda_hand").xquat.copy()

    def block_pos(self):
        return self.data.body(self.block_name).xpos.copy()

    def finger_gap(self):
        q1 = self.data.qpos[self.finger_qpos_adr[0]]
        q2 = self.data.qpos[self.finger_qpos_adr[1]]
        return float(q1 + q2)

    def finger_forces(self):
        f_l = self.data.sensor("panda_left_finger_force").data.copy()
        f_r = self.data.sensor("panda_right_finger_force").data.copy()
        return f_l, f_r

    # ---------- Helpers ----------
    def compute_spatial_error(self, p, q, p_d, q_d):
        qc = np.roll(q, -1)
        qd = np.roll(q_d, -1)
        Rc = Rotation.from_quat(qc).as_matrix()
        Rd = Rotation.from_quat(qd).as_matrix()
        R_err = Rd @ Rc.T
        omega = Rotation.from_matrix(R_err).as_rotvec()
        pos_e = p_d - p
        return np.concatenate([omega, pos_e])

    def set_state(self, new_state):
        if new_state != self.state:
            print(f"[STATE] {self.state} -> {new_state}")
            self.state = new_state
            self.state_enter_time = self.data.time

    def time_in_state(self):
        return self.data.time - self.state_enter_time

    def compute_pregrasp_target(self):
        p = self.block_pos()
        self.last_valid_grasp_xy = p[:2].copy()
        return np.array([p[0], p[1], p[2] + self.pregrasp_height])

    def compute_descend_target(self):
        p = self.block_pos()
        target = np.array([p[0], p[1], p[2] + self.descend_clearance])
        self.last_valid_grasp_xy = p[:2].copy()
        return target

    def compute_lift_target(self):
        xy = self.last_valid_grasp_xy.copy()
        return np.array([xy[0], xy[1], self.block_init_z + self.lift_height])

    def compute_move_target(self):
        return np.array([self.place_xy[0], self.place_xy[1], self.place_height])

    def compute_release_target(self):
        return np.array([self.place_xy[0], self.place_xy[1], self.release_height])

    def block_attached(self):
        bp = self.block_pos()
        ep = self.ee_pos()
        lateral = np.linalg.norm(bp[:2] - ep[:2])
        return lateral < 0.035 and bp[2] > self.block_init_z + 0.018

    def surface_top_z(self):
        geom = self.model.geom("surface")
        body = self.data.body(self.surface_name)
        return body.xpos[2] + geom.size[2]

    def compute_gripper_ctrl(self):
        f_l, f_r = self.finger_forces()
        total_force = np.linalg.norm(f_l) + np.linalg.norm(f_r)
        gap = self.finger_gap()

        if self.state in ["pregrasp", "descend"]:
            return self.finger_open_cmd

        if self.state == "close":
            cmd = self.finger_close_max + self.force_kp * (self.force_target_close - total_force)
            return float(np.clip(cmd, -30.0, 30.0))

        if self.state in ["lift", "move", "release_descend"]:
            if gap < self.close_gap_threshold or total_force > self.contact_force_threshold:
                cmd = self.finger_hold_cmd + 1.2 * (self.force_target_hold - total_force)
                return float(np.clip(cmd, -30.0, 30.0))
            return self.finger_close_max

        if self.state == "open":
            return self.finger_open_cmd

        return 0.0

    def update_state_machine(self):
        ee = self.ee_pos()
        block = self.block_pos()
        gap = self.finger_gap()
        f_l, f_r = self.finger_forces()
        total_force = np.linalg.norm(f_l) + np.linalg.norm(f_r)

        if self.state == "pregrasp":
            self.target_pos = self.compute_pregrasp_target()
            pos_err = np.linalg.norm(ee - self.target_pos)
            if pos_err < 0.012:
                self.set_state("descend")

        elif self.state == "descend":
            self.target_pos = self.compute_descend_target()
            xy_err = np.linalg.norm(ee[:2] - self.target_pos[:2])
            z_err = abs(ee[2] - self.target_pos[2])
            if (xy_err < 0.008 and z_err < 0.010) or total_force > self.contact_force_threshold:
                self.set_state("close")

        elif self.state == "close":
            self.target_pos = self.compute_descend_target()
            grasped_by_force = total_force > self.secure_force_threshold
            grasped_by_gap = gap < self.close_gap_threshold
            grasped_by_motion = block[2] > self.block_init_z + 0.006
            if (grasped_by_force and grasped_by_gap) or grasped_by_motion:
                self.set_state("lift")
            elif self.time_in_state() > 2.5:
                self.set_state("pregrasp")

        elif self.state == "lift":
            self.target_pos = self.compute_lift_target()
            if self.block_attached() and block[2] > self.block_init_z + 0.055:
                self.set_state("move")
            elif self.time_in_state() > 1.5 and not self.block_attached():
                self.set_state("pregrasp")

        elif self.state == "move":
            self.target_pos = self.compute_move_target()
            ee_err = np.linalg.norm(ee - self.target_pos)
            if ee_err < 0.015:
                self.set_state("release_descend")
            elif not self.block_attached() and self.time_in_state() > 0.3:
                self.set_state("pregrasp")

        elif self.state == "release_descend":
            self.target_pos = self.compute_release_target()
            ee_err = np.linalg.norm(ee - self.target_pos)
            if ee_err < 0.010:
                self.set_state("open")

        elif self.state == "open":
            self.target_pos = self.compute_release_target()
            if self.time_in_state() > 0.6:
                self.block_init_z = self.block_pos()[2]
                self.set_state("done")

        elif self.state == "done":
            self.target_pos = self.compute_pregrasp_target()

    def control_step(self, dt):
        q = np.array([self.data.qpos[adr] for adr in self.dof_adr])
        qdot = np.array([self.data.qvel[adr] for adr in self.dof_adr])

        self.update_state_machine()

        # Admittance input from finger contact forces
        f_l, f_r = self.finger_forces()
        f_contact = f_l + f_r
        W_ext = np.array([0.0, 0.0, 0.0, f_contact[0], f_contact[1], f_contact[2]])
        acc = np.linalg.inv(self.M_adm) @ (W_ext - self.D_adm @ self.adm_vel - self.K_adm @ self.adm_pos)
        self.adm_vel += acc * dt
        self.adm_pos += self.adm_vel * dt
        self.adm_pos[:3] = np.clip(self.adm_pos[:3], -self.max_pos_offset, self.max_pos_offset)
        self.adm_pos[3:] = 0.0

        pos_d = self.target_pos + self.adm_pos[:3]
        quat_d = self.target_quat

        X_e = self.compute_spatial_error(self.ee_pos(), self.ee_quat(), pos_d, quat_d)
        J = self.dynamics_calc.compute_spatial_jacobian(q, 6)
        V = J @ qdot
        Lambda = self.dynamics_calc.compute_task_space_mass_matrix(q, 6)
        F_task = Lambda @ (self.K_p @ X_e + self.K_d @ (-V))
        tau = J.T @ F_task + self.dynamics_calc.compute_coriolis_gravity(q, qdot)

        self.data.ctrl[:7] = tau
        self.data.ctrl[self.finger_act_id] = self.compute_gripper_ctrl()

        # periodic debug
        if self.data.time - self.last_print_time > 0.2:
            gap = self.finger_gap()
            total_force = np.linalg.norm(f_l) + np.linalg.norm(f_r)
            bp = self.block_pos()
            ep = self.ee_pos()
            print(
                f"t={self.data.time:5.2f} state={self.state:>14s} "
                f"ee_err={np.linalg.norm(ep - self.target_pos):.4f} gap={gap:.4f} "
                f"f={total_force:.2f} block_z={bp[2]:.4f}"
            )
            self.last_print_time = self.data.time

    def run(self):
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            viewer.cam.trackbodyid = self.model.body("panda_hand").id
            viewer.cam.distance = 1.1
            viewer.cam.elevation = -25
            viewer.cam.azimuth = 135
            while viewer.is_running():
                dt = self.model.opt.timestep
                self.control_step(dt)
                mujoco.mj_step(self.model, self.data)
                viewer.sync()


if __name__ == "__main__":
    sim = SingleArmBlockGraspClosedLoop()
    sim.run()
