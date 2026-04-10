#!/usr/bin/env python3
import mujoco
import mujoco.viewer
import numpy as np

from dynamics_calculator_pinocchio import DynamicsCalculatorPinocchio


class VariableJointImpedanceMouse:
    """
    Variable joint-space impedance driven by:
    1) small joint displacement dq = q - q_ref
    2) external end-effector wrench mapped to joint torque tau_ext = J^T wrench_ext

    Inference used in this script:
    kp_i ~= |tau_ext_i| / max(|dq_i|, dq_floor)
    kd_i = 2 * zeta * sqrt(kp_i * M_ii)

    Control law:
    tau = -Kp * dq - Kd * qdot + h
    """

    WRENCH_AXES = ("fx", "fy", "fz", "mx", "my", "mz")
    SPATIAL_WRENCH_AXES = ("mx", "my", "mz", "fx", "fy", "fz")

    def __init__(
        self,
        model_path="scene.xml",
        urdf_path="fr3v2.urdf",
        ee_body_name="fr3v2_link7",
        ee_frame_name="fr3v2_link8",
        wrench_mask=None,
    ):
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)

        self.joint_names = [f"fr3v2_joint{i + 1}" for i in range(7)]
        self.qpos_adr = [self.model.joint(name).qposadr[0] for name in self.joint_names]
        self.qvel_adr = [self.model.joint(name).dofadr[0] for name in self.joint_names]

        self.ee_body_name = ee_body_name
        self.ee_frame_name = ee_frame_name
        self.ee_body_id = self.model.body(ee_body_name).id

        self.dynamics_calc = DynamicsCalculatorPinocchio(
            urdf_path,
            ee_frame_name,
            self.joint_names,
        )

        self.q_init = self.model.keyframe("home").qpos.copy()[:7]
        for i, adr in enumerate(self.qpos_adr):
            self.data.qpos[adr] = self.q_init[i]
            self.data.qvel[self.qvel_adr[i]] = 0.0
        mujoco.mj_forward(self.model, self.data)

        self.q_ref = self.q_init.copy()

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
        self.set_wrench_mask(wrench_mask)

    # -------------------- public interface --------------------
    def set_reference(self, q_ref):
        self.q_ref = np.asarray(q_ref, dtype=float).reshape(7).copy()

    def set_reference_from_current(self):
        q, _ = self.get_joint_state()
        self.q_ref = q.copy()

    def set_wrench_mask(self, mask=None):
        if mask is None:
            mask = np.ones(6)
        mask_array = np.asarray(mask, dtype=float).reshape(6)
        self.wrench_mask = np.clip(mask_array, 0.0, 1.0)
        self.spatial_wrench_mask = np.array(
            [
                self.wrench_mask[3],
                self.wrench_mask[4],
                self.wrench_mask[5],
                self.wrench_mask[0],
                self.wrench_mask[1],
                self.wrench_mask[2],
            ],
            dtype=float,
        )

    def set_wrench_direction_mask(
        self,
        x=1.0,
        y=1.0,
        z=1.0,
        rx=1.0,
        ry=1.0,
        rz=1.0,
    ):
        self.set_wrench_mask([x, y, z, rx, ry, rz])

    def set_gain_limits(self, kp_min=None, kp_max=None, kd_min=None, kd_max=None):
        if kp_min is not None:
            self.kp_min = np.asarray(kp_min, dtype=float).reshape(7)
        if kp_max is not None:
            self.kp_max = np.asarray(kp_max, dtype=float).reshape(7)
        if kd_min is not None:
            self.kd_min = np.asarray(kd_min, dtype=float).reshape(7)
        if kd_max is not None:
            self.kd_max = np.asarray(kd_max, dtype=float).reshape(7)
        self.kp_diag = np.clip(self.kp_diag, self.kp_min, self.kp_max)
        self.kd_diag = np.clip(self.kd_diag, self.kd_min, self.kd_max)

    def get_gain_state(self):
        return {
            "kp": self.kp_diag.copy(),
            "kd": self.kd_diag.copy(),
            "q_ref": self.q_ref.copy(),
            "wrench_mask_axes": ("x", "y", "z", "rx", "ry", "rz"),
            "wrench_mask": self.wrench_mask.copy(),
        }

    # -------------------- helpers --------------------
    def _get_tau_limit(self):
        limit = []
        for name in self.joint_names:
            act_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            if act_id < 0 or not self.model.actuator_ctrllimited[act_id]:
                limit.append(np.inf)
            else:
                limit.append(abs(self.model.actuator_ctrlrange[act_id, 1]))
        return np.asarray(limit, dtype=float)

    def get_joint_state(self):
        q = np.array([self.data.qpos[adr] for adr in self.qpos_adr], dtype=float)
        qdot = np.array([self.data.qvel[adr] for adr in self.qvel_adr], dtype=float)
        return q, qdot

    def read_external_wrench(self):
        fm = self.data.xfrc_applied[self.ee_body_id].copy()
        spatial_wrench = np.hstack([fm[3:], fm[:3]])
        return spatial_wrench * self.spatial_wrench_mask

    def solve_variable_gains(self, q, wrench_ext):
        dq = np.clip(q - self.q_ref, -self.dq_limit, self.dq_limit)
        J = self.dynamics_calc.compute_spatial_jacobian(q, 6)
        tau_ext = J.T @ wrench_ext

        kp_raw = np.abs(tau_ext) / np.maximum(np.abs(dq), self.dq_floor)
        kp_raw = np.clip(kp_raw, self.kp_min, self.kp_max)

        M = self.dynamics_calc.compute_mass_matrix(q)
        M_diag = np.maximum(np.diag(M), 1e-6)
        kd_raw = 2.0 * self.zeta * np.sqrt(kp_raw * M_diag)
        kd_raw = np.clip(kd_raw, self.kd_min, self.kd_max)

        self.kp_diag = (1.0 - self.gain_filter_alpha) * self.kp_diag + self.gain_filter_alpha * kp_raw
        self.kd_diag = (1.0 - self.gain_filter_alpha) * self.kd_diag + self.gain_filter_alpha * kd_raw
        return dq, tau_ext

    def compute_control(self):
        q, qdot = self.get_joint_state()
        wrench_ext = self.read_external_wrench()
        dq, tau_ext = self.solve_variable_gains(q, wrench_ext)

        tau_imp = -self.kp_diag * dq - self.kd_diag * qdot
        h = self.dynamics_calc.compute_coriolis_gravity(q, qdot)
        tau = tau_imp + h
        tau = np.clip(tau, -self.tau_limit, self.tau_limit)
        return tau, dq, wrench_ext, tau_ext

    def control_step(self):
        tau, _, _, _ = self.compute_control()
        self.data.ctrl[:7] = tau

    def run(self):
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            viewer.cam.trackbodyid = self.ee_body_id
            while viewer.is_running():
                self.control_step()
                mujoco.mj_step(self.model, self.data)
                viewer.sync()


if __name__ == "__main__":
    sim = VariableJointImpedanceMouse(
        model_path="scene.xml",
        urdf_path="fr3v2.urdf",
        ee_body_name="fr3v2_link7",
        ee_frame_name="fr3v2_link8",
        wrench_mask=[1, 1, 1, 1, 1, 1],
    )
    sim.set_reference_from_current()
    sim.run()
