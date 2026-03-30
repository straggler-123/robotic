#!/usr/bin/env python3
import mujoco
import mujoco.viewer
import numpy as np
from scipy.spatial.transform import Rotation
from dynamics_calculator_wv import DynamicsCalculator


class BimanualAdmittance:

    def __init__(self, model_path="xml/surface_force_control.xml"):

        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)

        # ================= 左右臂 =================
        self.joint_names_left = [f"panda_left_joint{i+1}" for i in range(7)]
        self.joint_names_right = [f"panda_right_joint{i+1}" for i in range(7)]

        self.dof_left = [self.model.joint(n).dofadr[0] for n in self.joint_names_left]
        self.dof_right = [self.model.joint(n).dofadr[0] for n in self.joint_names_right]

        # Dynamics（左右分开）
        self.dyn_left = DynamicsCalculator(model_path, "panda_left_hand", self.joint_names_left)
        self.dyn_right = DynamicsCalculator(model_path, "panda_right_hand", self.joint_names_right)

        # 初始位姿
        q_init = np.array([0, -0.785, 0, -2.356, 0, 1.571, 0.785])

        for i, adr in enumerate(self.dof_left):
            self.data.qpos[adr] = q_init[i]

        for i, adr in enumerate(self.dof_right):
            self.data.qpos[adr] = q_init[i]

        mujoco.mj_forward(self.model, self.data)

        # ================= 共享目标 =================
        pL = self.data.body("panda_left_hand").xpos
        pR = self.data.body("panda_right_hand").xpos

        self.init_pos = (pL + pR) / 2.0  # 中点作为目标

        # ================= 控制参数 =================
        self.K_p = np.diag([20, 20, 20, 100, 100, 100]) * 5
        self.K_d = np.diag([2, 2, 2, 10, 10, 10]) * 5

        # ================= 导纳 =================
        self.adm_pos = np.zeros(6)
        self.adm_vel = np.zeros(6)

        self.M_adm = np.diag([2, 2, 2, 0.5, 0.5, 0.5])
        self.D_adm = np.diag([50, 50, 50, 10, 10, 10])
        self.K_adm = np.zeros((6, 6))

    # ================= 工具函数 =================
    def spatial_error(self, p, q, p_d, q_d):
        qc = np.roll(q, -1)
        qd = np.roll(q_d, -1)

        Rc = Rotation.from_quat(qc).as_matrix()
        Rd = Rotation.from_quat(qd).as_matrix()

        R_err = Rd @ Rc.T
        omega = Rotation.from_matrix(R_err).as_rotvec()

        pos_e = p_d - p
        return np.concatenate([omega, pos_e])

    # ================= 控制 =================
    def control_step(self, dt):

        # ===== 读取双臂状态 =====
        qL = np.array([self.data.qpos[i] for i in self.dof_left])
        qR = np.array([self.data.qpos[i] for i in self.dof_right])

        qdL = np.array([self.data.qvel[i] for i in self.dof_left])
        qdR = np.array([self.data.qvel[i] for i in self.dof_right])

        # ===== 外力（取左手为交互）=====
        body_id = self.model.body("panda_left_hand").id
        W_ext = self.data.xfrc_applied[body_id].copy()

        # ===== 导纳 =====
        acc = np.linalg.inv(self.M_adm) @ (
            W_ext - self.D_adm @ self.adm_vel - self.K_adm @ self.adm_pos
        )

        self.adm_vel += acc * dt
        self.adm_pos += self.adm_vel * dt

        # ===== 目标 =====
        pos_d = self.init_pos + self.adm_pos[:3]

        quat_d = self.data.body("panda_left_hand").xquat.copy()

        # ================= 左臂 =================
        pL = self.data.body("panda_left_hand").xpos
        qL_quat = self.data.body("panda_left_hand").xquat

        X_e_L = self.spatial_error(pL, qL_quat, pos_d, quat_d)

        JL = self.dyn_left.compute_spatial_jacobian(qL, 6)
        VL = JL @ qdL

        LambdaL = self.dyn_left.compute_task_space_mass_matrix(qL, 6)

        F_L = LambdaL @ (self.K_p @ X_e_L + self.K_d @ (-VL))

        tau_L = JL.T @ F_L + self.dyn_left.compute_coriolis_gravity(qL, qdL)

        # ================= 右臂 =================
        pR = self.data.body("panda_right_hand").xpos
        qR_quat = self.data.body("panda_right_hand").xquat

        X_e_R = self.spatial_error(pR, qR_quat, pos_d, quat_d)

        JR = self.dyn_right.compute_spatial_jacobian(qR, 6)
        VR = JR @ qdR

        LambdaR = self.dyn_right.compute_task_space_mass_matrix(qR, 6)

        F_R = LambdaR @ (self.K_p @ X_e_R + self.K_d @ (-VR))

        tau_R = JR.T @ F_R + self.dyn_right.compute_coriolis_gravity(qR, qdR)

        # ===== 合并控制 =====
        tau_full = np.zeros(self.model.nu)

        tau_full[:7] = tau_L
        tau_full[7:14] = tau_R

        self.data.ctrl[:] = tau_full

    # ================= 运行 =================
    def run(self):
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            while viewer.is_running():
                dt = self.model.opt.timestep
                self.control_step(dt)
                mujoco.mj_step(self.model, self.data)
                viewer.sync()


if __name__ == "__main__":
    sim = BimanualAdmittance()
    sim.run()