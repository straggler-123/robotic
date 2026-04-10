#!/usr/bin/env python3
from pathlib import Path
import sys

import mujoco
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[4]
MODEL_DIR = REPO_ROOT / "mujoco_menagerie-main" / "franka_fr3_v2"
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from dynamics_calculator_pinocchio import DynamicsCalculatorPinocchio


class VariableJointImpedanceAdmittanceController:
    def __init__(
        self,
        model_path=None,
        urdf_path=None,
        ee_body_name="fr3v2_link7",
        ee_frame_name="fr3v2_link8",
        motion_mask=None,
    ):
        model_path = Path(model_path) if model_path else MODEL_DIR / "scene.xml"
        urdf_path = Path(urdf_path) if urdf_path else MODEL_DIR / "fr3v2.urdf"

        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.data = mujoco.MjData(self.model)

        self.joint_names = [f"fr3v2_joint{i + 1}" for i in range(7)]
        self.qpos_adr = [self.model.joint(name).qposadr[0] for name in self.joint_names]
        self.qvel_adr = [self.model.joint(name).dofadr[0] for name in self.joint_names]
        self.ee_body_id = self.model.body(ee_body_name).id

        self.dynamics_calc = DynamicsCalculatorPinocchio(
            str(urdf_path),
            ee_frame_name,
            self.joint_names,
        )

        self.q_init = self.model.keyframe("home").qpos.copy()[:7]
        for i, adr in enumerate(self.qpos_adr):
            self.data.qpos[adr] = self.q_init[i]
            self.data.qvel[self.qvel_adr[i]] = 0.0
        mujoco.mj_forward(self.model, self.data)

        self.q_ref = self.q_init.copy()

        self.K_adm = np.diag([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.M_adm = np.diag([5.0, 5.0, 5.0, 1.0, 1.0, 1.0])
        self.D_adm = np.diag([80.0, 80.0, 80.0, 20.0, 20.0, 20.0])
        self.adm_pos = np.zeros(6)
        self.adm_vel = np.zeros(6)
        self.max_pos_offset = 0.10
        self.max_rot_offset = 0.50
        self.force_deadband = 1e-3
        self.ik_damping = 1e-4
        self.qref_rate_limit = np.array([0.8, 0.8, 0.8, 1.0, 1.2, 1.2, 1.5], dtype=float)
        self.was_admittance_active = False

        self.kp_min = np.array([20.0, 20.0, 20.0, 15.0, 10.0, 8.0, 5.0], dtype=float)
        self.kp_max = np.array([4000.0, 4000.0, 3500.0, 2500.0, 2000.0, 1500.0, 1000.0], dtype=float)
        self.kd_min = np.array([3.0, 3.0, 3.0, 2.0, 1.5, 1.0, 0.8], dtype=float)
        self.kd_max = np.array([220.0, 220.0, 200.0, 160.0, 120.0, 90.0, 60.0], dtype=float)
        self.dq_floor = np.array([0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002], dtype=float)
        self.dq_limit = np.array([0.20, 0.20, 0.20, 0.22, 0.25, 0.25, 0.30], dtype=float)
        self.zeta = np.array([1.2, 1.2, 1.2, 1.1, 1.0, 1.0, 1.0], dtype=float)
        self.gain_filter_alpha = 0.15
        self.kp_diag = self.kp_min.copy()
        self.kd_diag = self.kd_min.copy()

        self.tau_limit = self._get_tau_limit()
        self.set_motion_mask(motion_mask)

    @staticmethod
    def _admittance_mask_to_task_mask(mask):
        mask = np.asarray(mask, dtype=float).reshape(6)
        return np.array([mask[3], mask[4], mask[5], mask[0], mask[1], mask[2]], dtype=float)

    def _get_tau_limit(self):
        limit = []
        for name in self.joint_names:
            act_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            if act_id < 0 or not self.model.actuator_ctrllimited[act_id]:
                limit.append(np.inf)
            else:
                limit.append(abs(self.model.actuator_ctrlrange[act_id, 1]))
        return np.asarray(limit, dtype=float)

    def set_motion_mask(self, mask=None):
        if mask is None:
            mask = np.ones(6)
        mask_array = np.asarray(mask, dtype=float).reshape(6)
        self.motion_mask = np.clip(mask_array, 0.0, 1.0)
        self.task_motion_mask = self._admittance_mask_to_task_mask(self.motion_mask)

    def set_admittance_gains(self, mass=None, damping=None, stiffness=None):
        if mass is not None:
            self.M_adm = np.diag(np.asarray(mass, dtype=float).reshape(6))
        if damping is not None:
            self.D_adm = np.diag(np.asarray(damping, dtype=float).reshape(6))
        if stiffness is not None:
            self.K_adm = np.diag(np.asarray(stiffness, dtype=float).reshape(6))

    def set_reference_from_current(self):
        q, _ = self.get_joint_state()
        self.q_ref = q.copy()
        self.adm_pos[:] = 0.0
        self.adm_vel[:] = 0.0

    def get_joint_state(self):
        q = np.array([self.data.qpos[adr] for adr in self.qpos_adr], dtype=float)
        qdot = np.array([self.data.qvel[adr] for adr in self.qvel_adr], dtype=float)
        return q, qdot

    def read_external_wrench_task(self):
        fm = self.data.xfrc_applied[self.ee_body_id].copy()
        task_wrench = np.hstack([fm[3:], fm[:3]])
        return task_wrench * self.task_motion_mask

    def read_external_wrench_admittance(self):
        fm = self.data.xfrc_applied[self.ee_body_id].copy()
        adm_wrench = np.hstack([fm[:3], fm[3:]])
        return adm_wrench * self.motion_mask

    def update_admittance(self, dt, wrench_adm):
        if np.linalg.norm(wrench_adm) <= self.force_deadband:
            if self.was_admittance_active:
                q, _ = self.get_joint_state()
                self.q_ref = q.copy()
            self.was_admittance_active = False
            self.adm_vel[:] = 0.0
            self.adm_pos[:] = 0.0
            return

        self.was_admittance_active = True
        adm_acc = np.linalg.solve(
            self.M_adm,
            wrench_adm - self.D_adm @ self.adm_vel - self.K_adm @ self.adm_pos,
        )
        self.adm_vel += adm_acc * dt
        self.adm_pos += self.adm_vel * dt
        self.adm_pos[:3] = np.clip(self.adm_pos[:3], -self.max_pos_offset, self.max_pos_offset)
        self.adm_pos[3:] = np.clip(self.adm_pos[3:], -self.max_rot_offset, self.max_rot_offset)
        self.adm_vel *= self.motion_mask
        self.adm_pos *= self.motion_mask

    def update_joint_reference(self, q, dt):
        task_twist_ref = np.array(
            [self.adm_vel[3], self.adm_vel[4], self.adm_vel[5], self.adm_vel[0], self.adm_vel[1], self.adm_vel[2]],
            dtype=float,
        )
        J = self.dynamics_calc.compute_spatial_jacobian(q, 6)
        jj_t = J @ J.T + self.ik_damping * np.eye(6)
        J_pinv = J.T @ np.linalg.inv(jj_t)
        qref_dot = np.clip(J_pinv @ task_twist_ref, -self.qref_rate_limit, self.qref_rate_limit)
        self.q_ref += qref_dot * dt

    def solve_variable_gains(self, q, wrench_task):
        dq = np.clip(q - self.q_ref, -self.dq_limit, self.dq_limit)
        J = self.dynamics_calc.compute_spatial_jacobian(q, 6)
        tau_ext = J.T @ wrench_task
        kp_raw = np.clip(np.abs(tau_ext) / np.maximum(np.abs(dq), self.dq_floor), self.kp_min, self.kp_max)
        M = self.dynamics_calc.compute_mass_matrix(q)
        M_diag = np.maximum(np.diag(M), 1e-6)
        kd_raw = np.clip(2.0 * self.zeta * np.sqrt(kp_raw * M_diag), self.kd_min, self.kd_max)
        self.kp_diag = (1.0 - self.gain_filter_alpha) * self.kp_diag + self.gain_filter_alpha * kp_raw
        self.kd_diag = (1.0 - self.gain_filter_alpha) * self.kd_diag + self.gain_filter_alpha * kd_raw
        return dq, tau_ext

    def compute_control(self, dt):
        q, qdot = self.get_joint_state()
        wrench_adm = self.read_external_wrench_admittance()
        wrench_task = self.read_external_wrench_task()
        self.update_admittance(dt, wrench_adm)
        self.update_joint_reference(q, dt)
        dq, tau_ext = self.solve_variable_gains(q, wrench_task)
        tau_imp = -self.kp_diag * dq - self.kd_diag * qdot
        h = self.dynamics_calc.compute_coriolis_gravity(q, qdot)
        tau = np.clip(tau_imp + h, -self.tau_limit, self.tau_limit)
        return {
            "tau": tau,
            "dq": dq,
            "wrench_task": wrench_task,
            "tau_ext": tau_ext,
            "q_ref": self.q_ref.copy(),
            "kp": self.kp_diag.copy(),
            "kd": self.kd_diag.copy(),
            "adm_pos": self.adm_pos.copy(),
        }

    def step(self):
        dt = self.model.opt.timestep
        result = self.compute_control(dt)
        self.data.ctrl[:7] = result["tau"]
        mujoco.mj_step(self.model, self.data)
        return result
