#!/usr/bin/env python3
import mujoco
import mujoco.viewer
import numpy as np
from scipy.spatial.transform import Rotation as R
from dynamics_calculator_pinocchio import DynamicsCalculatorPinocchio


def mj_quat_to_scipy(q_mj: np.ndarray) -> np.ndarray:
    return np.array([q_mj[1], q_mj[2], q_mj[3], q_mj[0]], dtype=float)


class TaskSpaceImpedance6DNoSensor:
    """
    固定目标位姿的阻抗控制器
    约定：
    - 任务空间误差 / 速度 / wrench 排布统一为 [rx, ry, rz, x, y, z]
    - wrench 内部排布为 [mx, my, mz, fx, fy, fz]
    - 从 MuJoCo 读取 xfrc_applied 时，原始顺序是 [Fx, Fy, Fz, Mx, My, Mz]
      这里会自动重排成 [mx, my, mz, fx, fy, fz]
    - ctrl_mask 顺序同上：1 启用，0 关闭
    """

    def __init__(
        self,
        model_path: str = "scene.xml",
        urdf_path: str = "fr3v2.urdf",
        ee_body_name: str = "fr3v2_link7",
        ee_frame_name: str = "fr3v2_link8",
        joint_names=None,
        ctrl_mask=None,
    ):
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)

        self.joint_names = joint_names or [f"fr3v2_joint{i}" for i in range(1, 8)]
        self.qpos_adr = [self.model.joint(name).qposadr[0] for name in self.joint_names]
        self.qvel_adr = [self.model.joint(name).dofadr[0] for name in self.joint_names]

        self.ee_body_name = ee_body_name
        self.ee_body_id = self.model.body(ee_body_name).id

        self.dyn = DynamicsCalculatorPinocchio(
            urdf_path=urdf_path,
            end_effector_frame=ee_frame_name,
            joint_names=self.joint_names,
        )

        # [rx, ry, rz, x, y, z]
        self.K = np.diag([80.0, 80.0, 80.0, 1500.0, 1500.0, 1500.0])
        self.D = np.diag([8.0, 8.0, 8.0, 60.0, 60.0, 60.0])
        self.set_mask(ctrl_mask if ctrl_mask is not None else [1, 1, 1, 1, 1, 1])

        self.wrench_bias = np.zeros(6)
        self.tau_limit = self._get_tau_limit()

        self.reset_to_home()
        self.set_target_from_current()

    def _get_tau_limit(self) -> np.ndarray:
        limit = []
        for name in self.joint_names:
            act_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            if act_id < 0 or not self.model.actuator_ctrllimited[act_id]:
                limit.append(np.inf)
            else:
                limit.append(abs(self.model.actuator_ctrlrange[act_id, 1]))
        return np.asarray(limit, dtype=float)

    def set_mask(self, mask) -> None:
        mask = np.asarray(mask, dtype=float).reshape(6)
        self.mask = np.clip(mask, 0.0, 1.0)
        self.S = np.diag(self.mask)

    def set_impedance(self, K=None, D=None) -> None:
        if K is not None:
            self.K = np.diag(K) if np.asarray(K).ndim == 1 else np.asarray(K, dtype=float)
        if D is not None:
            self.D = np.diag(D) if np.asarray(D).ndim == 1 else np.asarray(D, dtype=float)

    def reset_to_home(self, key_name: str = "home") -> None:
        q_home = self.model.keyframe(key_name).qpos[:7].copy()
        for i in range(7):
            self.data.qpos[self.qpos_adr[i]] = q_home[i]
            self.data.qvel[self.qvel_adr[i]] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def set_target(self, pos_d: np.ndarray, quat_d_mj: np.ndarray) -> None:
        self.pos_d = np.asarray(pos_d, dtype=float).copy()
        self.quat_d = np.asarray(quat_d_mj, dtype=float).copy()

    def set_target_from_current(self) -> None:
        pos, quat = self.get_ee_pose()
        self.set_target(pos, quat)

    def get_joint_state(self):
        q = np.array([self.data.qpos[i] for i in self.qpos_adr], dtype=float)
        qdot = np.array([self.data.qvel[i] for i in self.qvel_adr], dtype=float)
        return q, qdot

    def get_ee_pose(self):
        body = self.data.body(self.ee_body_name)
        return body.xpos.copy(), body.xquat.copy()

    @staticmethod
    def pose_error(pos, quat_mj, pos_d, quat_d_mj) -> np.ndarray:
        r = R.from_quat(mj_quat_to_scipy(quat_mj))
        r_d = R.from_quat(mj_quat_to_scipy(quat_d_mj))
        rot_err = (r_d * r.inv()).as_rotvec()
        pos_err = np.asarray(pos_d) - np.asarray(pos)
        return np.hstack([rot_err, pos_err])

    def read_wrench(self) -> np.ndarray:
        """
        从 xfrc_applied 读取世界系外力。

        MuJoCo 原始顺序: [Fx, Fy, Fz, Mx, My, Mz]
        返回控制器顺序: [Mx, My, Mz, Fx, Fy, Fz]
        """
        fm = self.data.xfrc_applied[self.ee_body_id].copy()
        wrench = np.hstack([fm[3:], fm[:3]])
        return wrench - self.wrench_bias

    def zero_wrench_bias(self) -> None:
        self.wrench_bias += self.read_wrench()

    def control_step(self) -> None:
        q, qdot = self.get_joint_state()
        pos, quat = self.get_ee_pose()

        J = self.dyn.compute_spatial_jacobian(q, task_space_dim=6)   # [ω; v]
        xdot = J @ qdot                                              # [ω; v]
        e = self.pose_error(pos, quat, self.pos_d, self.quat_d)      # [rot; pos]
        wrench_ext = self.read_wrench()                              # [m; f]

        wrench_cmd = self.S @ (self.K @ e - self.D @ xdot - wrench_ext)
        tau = J.T @ wrench_cmd + self.dyn.compute_coriolis_gravity(q, qdot)
        tau = np.clip(tau, -self.tau_limit, self.tau_limit)
        self.data.ctrl[:7] = tau

    def run(self) -> None:
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            viewer.cam.trackbodyid = self.ee_body_id
            while viewer.is_running():
                self.control_step()
                mujoco.mj_step(self.model, self.data)
                viewer.sync()


if __name__ == "__main__":
    ctrl = TaskSpaceImpedance6DNoSensor(
        model_path="scene.xml",
        urdf_path="fr3v2.urdf",
        ee_body_name="fr3v2_link7",     # xfrc_applied 施加在哪个 body，就读哪个 body
        ee_frame_name="fr3v2_link7",    # Pinocchio 任务空间 frame
        ctrl_mask=[1, 1, 1, 1, 1, 1],    # [rx, ry, rz, x, y, z]
    )

    # 例子：只控制位置
    # ctrl.set_mask([0, 0, 0, 1, 1, 1])

    # 例子：z 方向不控
    # ctrl.set_mask([1, 1, 1, 1, 1, 0])

    ctrl.zero_wrench_bias()
    ctrl.run()
