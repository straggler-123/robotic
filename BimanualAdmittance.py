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

        self.joint_names_left = [f"panda_left_joint{i+1}" for i in range(7)]
        self.joint_names_right = [f"panda_right_joint{i+1}" for i in range(7)]

        self.dof_left = [self.model.joint(n).dofadr[0] for n in self.joint_names_left]
        self.dof_right = [self.model.joint(n).dofadr[0] for n in self.joint_names_right]

        self.dyn_left = DynamicsCalculator(model_path, "panda_left_hand", self.joint_names_left)
        self.dyn_right = DynamicsCalculator(model_path, "panda_right_hand", self.joint_names_right)

        # 初始位姿
        q_init = np.array([0, -0.785, 0, -2.356, 0, 3.14, 0.785])

        for i, adr in enumerate(self.dof_left):
            self.data.qpos[adr] = q_init[i]

        for i, adr in enumerate(self.dof_right):
            self.data.qpos[adr] = q_init[i]

        mujoco.mj_forward(self.model, self.data)

        pL = self.data.body("panda_left_hand").xpos
        pR = self.data.body("panda_right_hand").xpos


        # ================= 控制参数 =================
        self.K_p = np.diag([20, 20, 20, 100, 100, 100]) * 5
        self.K_d = np.diag([2, 2, 2, 10, 10, 10]) * 5

        # ================= 导纳 =================
        self.adm_pos_L = np.zeros(6)
        self.adm_vel_L = np.zeros(6)
        self.adm_pos_R = np.zeros(6)
        self.adm_vel_R = np.zeros(6)

        self.init_pos_L = pL.copy()
        self.init_pos_R = pR.copy()

        self.M_adm = np.diag([2, 2, 2, 0.5, 0.5, 0.5])
        self.D_adm = np.diag([50, 50, 50, 10, 10, 10])
        self.K_adm = np.zeros((6, 6))

    def spatial_error(self, p, q, p_d, q_d):
        qc = np.roll(q, -1)
        qd = np.roll(q_d, -1)

        Rc = Rotation.from_quat(qc).as_matrix()
        Rd = Rotation.from_quat(qd).as_matrix()

        R_err = Rd @ Rc.T
        omega = Rotation.from_matrix(R_err).as_rotvec()

        pos_e = p_d - p
        return np.concatenate([omega, pos_e])

    def control_step(self, dt):

        qL = np.array([self.data.qpos[i] for i in self.dof_left])
        qR = np.array([self.data.qpos[i] for i in self.dof_right])

        qdL = np.array([self.data.qvel[i] for i in self.dof_left])
        qdR = np.array([self.data.qvel[i] for i in self.dof_right])

    # ===== 外力检测 =====
        id_L = self.model.body("panda_left_hand").id
        id_R = self.model.body("panda_right_hand").id

        W_L = self.data.xfrc_applied[id_L].copy()
        W_R = self.data.xfrc_applied[id_R].copy()

        norm_L = np.linalg.norm(W_L)
        norm_R = np.linalg.norm(W_R)

    # ===== 当前位姿 =====
        pL = self.data.body("panda_left_hand").xpos
        qL_quat = self.data.body("panda_left_hand").xquat

        pR = self.data.body("panda_right_hand").xpos
        qR_quat = self.data.body("panda_right_hand").xquat

        tau_full = np.zeros(self.model.nu)

    # ================= LEFT HAND =================
        if norm_L > 1e-3:

        # 导纳
            acc = np.linalg.inv(self.M_adm) @ (
                W_L - self.D_adm @ self.adm_vel_L - self.K_adm @ self.adm_pos_L
            )
            self.adm_vel_L += acc * dt
            self.adm_pos_L += self.adm_vel_L * dt

        else:
        # 外力消失 → 锁定当前位置
            self.adm_vel_L[:] = 0.0
            self.init_pos_L = pL - self.adm_pos_L[:3]
            self.adm_pos_L[:] = 0.0

    # 目标位姿
        pos_d_L = self.init_pos_L + self.adm_pos_L[:3]

        rot_offset_L = Rotation.from_rotvec(self.adm_pos_L[3:])
        rot_current_L = Rotation.from_quat(np.roll(qL_quat, -1))
        rot_target_L = rot_offset_L * rot_current_L
        quat_d_L = np.roll(rot_target_L.as_quat(), 1)

    # OSC
        X_e_L = self.spatial_error(pL, qL_quat, pos_d_L, quat_d_L)
        JL = self.dyn_left.compute_spatial_jacobian(qL, 6)
        VL = JL @ qdL
        LambdaL = self.dyn_left.compute_task_space_mass_matrix(qL, 6)

        F_L = LambdaL @ (self.K_p @ X_e_L + self.K_d @ (-VL))
        tau_L = JL.T @ F_L + self.dyn_left.compute_coriolis_gravity(qL, qdL)

        tau_full[:7] = tau_L

    # ================= RIGHT HAND =================
        if norm_R > 1e-3:

            acc = np.linalg.inv(self.M_adm) @ (
                W_R - self.D_adm @ self.adm_vel_R - self.K_adm @ self.adm_pos_R
            )
            self.adm_vel_R += acc * dt
            self.adm_pos_R += self.adm_vel_R * dt

        else:
            self.adm_vel_R[:] = 0.0
            self.init_pos_R = pR - self.adm_pos_R[:3]
            self.adm_pos_R[:] = 0.0

        pos_d_R = self.init_pos_R + self.adm_pos_R[:3]

        rot_offset_R = Rotation.from_rotvec(self.adm_pos_R[3:])
        rot_current_R = Rotation.from_quat(np.roll(qR_quat, -1))
        rot_target_R = rot_offset_R * rot_current_R
        quat_d_R = np.roll(rot_target_R.as_quat(), 1)

        X_e_R = self.spatial_error(pR, qR_quat, pos_d_R, quat_d_R)
        JR = self.dyn_right.compute_spatial_jacobian(qR, 6)
        VR = JR @ qdR
        LambdaR = self.dyn_right.compute_task_space_mass_matrix(qR, 6)

        F_R = LambdaR @ (self.K_p @ X_e_R + self.K_d @ (-VR))
        tau_R = JR.T @ F_R + self.dyn_right.compute_coriolis_gravity(qR, qdR)

        tau_full[7:14] = tau_R
        
        self.data.ctrl[:] = tau_full

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