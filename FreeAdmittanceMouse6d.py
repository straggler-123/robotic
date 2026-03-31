#!/usr/bin/env python3
import mujoco
import mujoco.viewer
import numpy as np
from scipy.spatial.transform import Rotation
from dynamics_calculator_wv import DynamicsCalculator

class FreeAdmittanceMouse:
    def __init__(self, model_path="surface_force_control.xml"):
        # 初始化模型和数据
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.joint_names = [f"panda_joint{i+1}" for i in range(7)]
        
        self.dof_adr = [self.model.joint(name).dofadr[0] for name in self.joint_names]

        self.dynamics_calc = DynamicsCalculator(model_path, "panda_hand", self.joint_names)

        # 初始关节位姿
        self.q_init = np.array([0, -0.785, 0, -2.356, 0, 3.14, 0.785])
        for i, adr in enumerate(self.dof_adr):
            self.data.qpos[adr] = self.q_init[i]
        mujoco.mj_forward(self.model, self.data)
        self.init_pos = np.array(self.data.body("panda_hand").xpos)

        # 控制参数 末端 PD 控制的刚度（K_p）和阻尼（K_d）
        self.K_p = np.diag([20, 20, 20, 100, 100, 100])*5
        self.K_d = np.diag([2, 2, 2, 10, 10, 10])*5
        # ===== 导纳参数（6D）=====
        self.adm_vel = np.zeros(6)
        self.adm_pos = np.zeros(6)
        # 虚拟质量-阻尼-刚度
        self.M_adm = np.diag([2, 2, 2, 0.5, 0.5, 0.5])
        self.D_adm = np.diag([50, 50, 50, 10, 10, 10])
        self.K_adm = np.diag([0, 0, 0, 0, 0, 0])  # 设为0 = 纯导纳
        self.max_pos_offset = 0.1   # 10cm
        self.max_rot_offset = 0.5   # rad

    def compute_spatial_error(self, p, q, p_d, q_d):  #空间误差计算 任务空间 PD 控制误差计算，用于生成末端力
        qc = np.roll(q, -1)
        qd = np.roll(q_d, -1)
        Rc = Rotation.from_quat(qc).as_matrix()
        Rd = Rotation.from_quat(qd).as_matrix()
        R_err = Rd @ Rc.T #计算期望姿态与当前姿态的误差矩阵
        omega = Rotation.from_matrix(R_err).as_rotvec() #将旋转误差转成旋转向量
        pos_e = p_d - p #位置误差
        return np.concatenate([omega, pos_e])

    def control_step(self, dt):
        q = np.array([self.data.qpos[adr] for adr in self.dof_adr])
        qdot = np.array([self.data.qvel[adr] for adr in self.dof_adr])
        body_id = self.model.body("panda_hand").id
        # ===== 6D 外力（wrench）=====
        body_id = self.model.body("panda_hand").id
        W_ext = self.data.xfrc_applied[body_id].copy()  # [Fx Fy Fz Mx My Mz]
        # ===== 6D 导纳动力学 =====
        if np.linalg.norm(W_ext) > 1e-3:
            acc = np.linalg.inv(self.M_adm) @ (
            W_ext - self.D_adm @ self.adm_vel - self.K_adm @ self.adm_pos
            )
            self.adm_vel += acc * dt
            self.adm_pos += self.adm_vel * dt
            self.adm_pos[:3] = np.clip(self.adm_pos[:3], -self.max_pos_offset, self.max_pos_offset)
            self.adm_pos[3:] = np.clip(self.adm_pos[3:], -self.max_rot_offset, self.max_rot_offset)
            # 1.导纳限制x方向 只有位置导纳，姿态不受外力影响
            # self.adm_pos[1:] = 0
            # self.adm_vel[1:] = 0

        else:
            # 外力消失 → 锁定当前位置
            self.adm_vel[:] = 0.0
            current_pos = self.data.body("panda_hand").xpos.copy()
            self.init_pos = current_pos - self.adm_pos[:3]
            self.adm_pos[:] = 0.0
        # ===== 目标位置 =====
        pos_d = self.init_pos + self.adm_pos[:3]
        quat_current = self.data.body("panda_hand").xquat.copy()

        rot_offset = Rotation.from_rotvec(self.adm_pos[3:])
        rot_current = Rotation.from_quat(np.roll(quat_current, -1))

        rot_target = rot_offset * rot_current
        quat_d = np.roll(rot_target.as_quat(), 1)
        # 2.锁姿态
        # quat_d = self.data.body("panda_hand").xquat.copy()

        # ===== 控制计算 ===== OSC（操作空间控制）
        X_e = self.compute_spatial_error(np.array(self.data.body("panda_hand").xpos),
                                         np.array(self.data.body("panda_hand").xquat),
                                         pos_d, quat_d)

        J = self.dynamics_calc.compute_spatial_jacobian(q, 6) #将关节速度映射到末端速度
        V = J @ qdot
        Lambda = self.dynamics_calc.compute_task_space_mass_matrix(q, 6) #将关节质量分布映射到末端
        V_e = -V
        # 3.柔顺方向（X）——导纳
        # S = np.diag([0, 0, 0, 1, 0, 0])
        # S_c = np.eye(6) - S
        # F_motion = S @ (self.K_p @ X_e + self.K_d @ V_e)
        # K_stiff = 2000
        # D_stiff = 50
        # F_constraint = S_c @ (K_stiff * X_e + D_stiff * V_e)
        # F_task = Lambda @ (F_motion + F_constraint)


        F_task = Lambda @ (self.K_p @ X_e + self.K_d @ V_e)
        tau = J.T @ F_task + self.dynamics_calc.compute_coriolis_gravity(q, qdot)
        self.data.ctrl[:] = tau

    def run(self):
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            viewer.cam.trackbodyid = self.model.body("panda_hand").id
            while viewer.is_running():
                dt = self.model.opt.timestep
                self.control_step(dt)
                mujoco.mj_step(self.model, self.data)
                viewer.sync()


if __name__ == "__main__":
    sim = FreeAdmittanceMouse()
    sim.run()