#!/usr/bin/env python3
import mujoco
import mujoco.viewer
import numpy as np
from scipy.spatial.transform import Rotation
from dynamics_calculator_wv import DynamicsCalculator

class FreeAdmittanceMouse:
    def __init__(self, model_path="surface_force_control_disk.xml"):
        # 初始化模型和数据
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.joint_names = [f"panda_joint{i+1}" for i in range(7)]
        self.dof_adr = [self.model.joint(name).dofadr[0] for name in self.joint_names]

        self.dynamics_calc = DynamicsCalculator(model_path, "panda_hand", self.joint_names)

        # 初始关节位姿
        self.q_init = np.array([0, -0.785, 0, -2.356, 0, 1.571, 0.785])
        for i, adr in enumerate(self.dof_adr):
            self.data.qpos[adr] = self.q_init[i]
        mujoco.mj_forward(self.model, self.data)

        # 末端初始位置（Z固定）
        self.init_pos = np.array(self.data.body("panda_hand").xpos)

        # 控制参数 末端 PD 控制的刚度（K_p）和阻尼（K_d）
        self.K_p = np.diag([20, 20, 20, 100, 100, 100])*5
        self.K_d = np.diag([2, 2, 2, 10, 10, 10])*5

        # 导纳参数（XY方向）
        self.adm_velocity = np.zeros(2)
        self.adm_offset = np.zeros(2)
        self.K_adm = 0.02
        self.D_adm = 1.0
        self.max_offset = 0.1  # 最大位移 10cm

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

        # ===== 获取鼠标施加的外力 =====
        fx = self.data.xfrc_applied[body_id, 0]  # MuJoCo x方向外力
        fy = self.data.xfrc_applied[body_id, 1]  # MuJoCo y方向外力
        F_xy = np.array([fx, fy])

        # ===== 导纳更新 =====
        if np.linalg.norm(F_xy) > 1e-3:
            # 有力时更新速度和偏移
            acc = self.K_adm * F_xy - self.D_adm * self.adm_velocity
            self.adm_velocity += acc * dt
            self.adm_offset += self.adm_velocity * dt
            self.adm_offset = np.clip(self.adm_offset, -self.max_offset, self.max_offset)
        else:
            # 外力消失，锁定当前位置
            self.adm_velocity[:] = 0.0
            current_pos = np.array(self.data.body("panda_hand").xpos)
            self.init_pos[:2] = current_pos[:2] - self.adm_offset

        # 目标位置 =初始位置 + 导纳偏移
        pos_d = self.init_pos.copy()
        pos_d[0] += self.adm_offset[0]
        pos_d[1] += self.adm_offset[1]

        # 保持姿态
        quat_d = np.array(self.data.body("panda_hand").xquat)

        # ===== 控制计算 =====
        X_e = self.compute_spatial_error(np.array(self.data.body("panda_hand").xpos),
                                         np.array(self.data.body("panda_hand").xquat),
                                         pos_d, quat_d)

        J = self.dynamics_calc.compute_spatial_jacobian(q, 6) #将关节速度映射到末端速度
        V = J @ qdot
        Lambda = self.dynamics_calc.compute_task_space_mass_matrix(q, 6) #将关节质量分布映射到末端
        V_e = -V

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