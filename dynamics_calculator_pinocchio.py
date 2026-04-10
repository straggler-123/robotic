#!/usr/bin/env python3
"""
机器人动力学量计算器 - 使用 Pinocchio

修复点：
1. 不再使用“最后 7 个驱动关节”这种危险逻辑。
2. 显式按关节名绑定 fr3_joint1 ~ fr3_joint7，避免把 finger joints 混进来。
3. 末端 frame 优先精确匹配；若缺失，再按候选顺序回退。

约定：
- 所有任务空间量（雅可比、速度、力）均在世界坐标系 / Spatial Frame 下表示。
- Twist V_s = [ω_x, ω_y, ω_z, v_x, v_y, v_z]^T
- Wrench F_s = [m_x, m_y, m_z, f_x, f_y, f_z]^T
"""

import numpy as np
from typing import Optional, Sequence

try:
    import pinocchio as pin
    PINOCCHIO_AVAILABLE = True
except ImportError:
    PINOCCHIO_AVAILABLE = False
    print("⚠ Pinocchio 未安装，请安装: pip install pin")


class DynamicsCalculatorPinocchio:
    """机器人动力学量计算器 (使用 Pinocchio)"""

    def __init__(
        self,
        urdf_path: str = "fr3v2.urdf",
        end_effector_frame: str = "fr3v2_link7",
        root_joint_type=None,
        joint_names: Optional[Sequence[str]] = None,
    ):
        if not PINOCCHIO_AVAILABLE:
            raise ImportError("Pinocchio 未安装，请先安装: pip install pin")

        try:
            if root_joint_type is None:
                self.model = pin.buildModelFromUrdf(urdf_path)
            else:
                self.model = pin.buildModelFromUrdf(urdf_path, pin.JointModelFreeFlyer())
        except Exception as e:
            raise FileNotFoundError(
                f"无法加载 URDF 文件 {urdf_path}: {e}\n请确保文件存在且路径正确"
            )

        self.data = self.model.createData()
        self.has_fixed_base = (self.model.nq > self.model.nv)

        # -------- 1) 显式绑定机械臂关节 --------
        if joint_names is None:
            joint_names = [f"fr3v2_joint{i}" for i in range(1, 8)]
        self.actuated_joint_names = list(joint_names)
        self.actuated_joint_indices = self._resolve_actuated_joint_indices(self.actuated_joint_names)
        self.n_actuated_joints = len(self.actuated_joint_indices)
        self.n_joints = self.n_actuated_joints

        # 为 _expand_configuration 预先构建 v_idx -> q_idx 映射，避免每次扫描所有 joints
        self.v_to_q_index = {}
        for joint_id in range(1, self.model.njoints):
            joint = self.model.joints[joint_id]
            if joint.nv <= 0:
                continue
            for local_k in range(joint.nv):
                self.v_to_q_index[joint.idx_v + local_k] = joint.idx_q + local_k

        # -------- 2) 精确解析末端 frame --------
        self.end_effector_name, self.end_effector_id = self._resolve_end_effector_frame(end_effector_frame)


    def _resolve_actuated_joint_indices(self, joint_names: Sequence[str]) -> list:
        """按 joint 名称显式解析 idx_v，避免 finger joints 混入。"""
        resolved = []
        for name in joint_names:
            if not self.model.existJointName(name):
                raise ValueError(f"Pinocchio 模型中找不到关节 '{name}'")
            joint_id = self.model.getJointId(name)
            if joint_id <= 0:
                raise ValueError(f"Pinocchio 模型中找不到有效关节 '{name}'")
            joint = self.model.joints[joint_id]
            if joint.nv != 1:
                raise ValueError(
                    f"关节 '{name}' 的 nv={joint.nv}，当前代码默认每个驱动关节 1 自由度。"
                )
            resolved.append(joint.idx_v)
        return resolved

    def _resolve_end_effector_frame(self, requested_name: str):
        frame_names = [self.model.frames[i].name for i in range(self.model.nframes)]

        # 精确匹配优先
        if requested_name in frame_names:
            frame_id = self.model.getFrameId(requested_name)
            return requested_name, frame_id

        # 常见 fr3 末端候选，优先 hand
        fallback_candidates = [
            "fr3v2_hand",
            "fr3v2_link7",
            "fr3v2_link8",

        ]
        for name in fallback_candidates:
            if name in frame_names:
                print(
                    f"⚠ 未找到指定框架 {requested_name}，回退使用 {name}"
                )
                return name, self.model.getFrameId(name)

        # 再做模糊匹配
        fuzzy = [
            n for n in frame_names
            if ("hand" in n.lower()) or ("link8" in n.lower()) or ("tcp" in n.lower())
        ]
        if fuzzy:
            print(f"⚠ 未找到指定框架 {requested_name}，模糊匹配使用 {fuzzy[0]}")
            return fuzzy[0], self.model.getFrameId(fuzzy[0])

        raise ValueError(
            f"找不到末端执行器框架 '{requested_name}'。可用 frame: {frame_names}"
        )

    def _expand_configuration(self, q: np.ndarray) -> np.ndarray:
        """将驱动关节配置扩展为完整配置向量（包括固定基座）"""
        q = np.asarray(q).reshape(-1)
        if len(q) != self.n_joints:
            raise ValueError(f"q 长度 {len(q)} 与驱动关节数量 {self.n_joints} 不匹配")

        if self.model.nq == len(q):
            return q.copy()

        q_full = np.zeros(self.model.nq)
        for i, v_idx in enumerate(self.actuated_joint_indices):
            q_idx = self.v_to_q_index.get(v_idx, None)
            if q_idx is None or q_idx >= self.model.nq:
                raise RuntimeError(f"无法将 idx_v={v_idx} 映射到配置索引 idx_q")
            q_full[q_idx] = q[i]
        return q_full

    def _expand_velocity(self, qdot: np.ndarray) -> np.ndarray:
        qdot = np.asarray(qdot).reshape(-1)
        if len(qdot) != self.n_joints:
            raise ValueError(f"qdot 长度 {len(qdot)} 与驱动关节数量 {self.n_joints} 不匹配")
        qdot_full = np.zeros(self.model.nv)
        for i, v_idx in enumerate(self.actuated_joint_indices):
            qdot_full[v_idx] = qdot[i]
        return qdot_full

    def _extract_actuated_matrix(self, M_full: np.ndarray) -> np.ndarray:
        M = np.zeros((self.n_joints, self.n_joints))
        for i, v_idx_i in enumerate(self.actuated_joint_indices):
            for j, v_idx_j in enumerate(self.actuated_joint_indices):
                M[i, j] = M_full[v_idx_i, v_idx_j]
        return M

    def compute_mass_matrix(self, q: np.ndarray) -> np.ndarray:
        q_full = self._expand_configuration(q)
        pin.computeJointJacobians(self.model, self.data, q_full)
        M_full = pin.crba(self.model, self.data, q_full)
        return self._extract_actuated_matrix(M_full)

    def compute_coriolis_gravity(self, q: np.ndarray, qdot: np.ndarray) -> np.ndarray:
        q_full = self._expand_configuration(q)
        qdot_full = self._expand_velocity(qdot)

        pin.computeCoriolisMatrix(self.model, self.data, q_full, qdot_full)
        pin.computeGeneralizedGravity(self.model, self.data, q_full)
        h_full = self.data.C @ qdot_full + self.data.g
        return np.array([h_full[v_idx] for v_idx in self.actuated_joint_indices])

    def compute_gravity_term(self, q: np.ndarray) -> np.ndarray:
        q_full = self._expand_configuration(q)
        pin.computeGeneralizedGravity(self.model, self.data, q_full)
        return np.array([self.data.g[v_idx] for v_idx in self.actuated_joint_indices])

    def compute_spatial_jacobian(self, q: np.ndarray, task_space_dim: int = 6) -> np.ndarray:
        """
        Pinocchio 的 LOCAL_WORLD_ALIGNED 返回 [v; ω] 排布，
        这里转换为 [ω; v]。
        """
        q_full = self._expand_configuration(q)
        pin.computeJointJacobians(self.model, self.data, q_full)
        pin.updateFramePlacements(self.model, self.data)

        J_local_full = pin.getFrameJacobian(
            self.model,
            self.data,
            self.end_effector_id,
            pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
        )
        J_local = J_local_full[:, self.actuated_joint_indices]

        if task_space_dim == 6:
            J_s = np.zeros((6, self.n_joints))
            J_s[:3, :] = J_local[3:, :]
            J_s[3:, :] = J_local[:3, :]
        elif task_space_dim == 3:
            J_s = J_local[:3, :]
        else:
            raise ValueError("不支持的任务空间维度")
        return J_s

    def compute_jacobian_derivative(
        self,
        q: np.ndarray,
        qdot: np.ndarray,
        task_space_dim: int = 6,
        epsilon: float = 1e-6,
    ) -> np.ndarray:
        J_dot = np.zeros((task_space_dim, self.n_joints))
        for i in range(self.n_joints):
            q_plus = q.copy()
            q_plus[i] += epsilon
            J_plus = self.compute_spatial_jacobian(q_plus, task_space_dim)

            q_minus = q.copy()
            q_minus[i] -= epsilon
            J_minus = self.compute_spatial_jacobian(q_minus, task_space_dim)

            dJ_dqi = (J_plus - J_minus) / (2 * epsilon)
            J_dot += dJ_dqi * qdot[i]
        return J_dot

    def compute_task_space_mass_matrix(
        self,
        q: np.ndarray,
        task_space_dim: int = 6,
        use_pseudoinverse: bool = True,
        damping: float = 1e-6,
    ) -> np.ndarray:
        M = self.compute_mass_matrix(q)
        J_s = self.compute_spatial_jacobian(q, task_space_dim)

        if use_pseudoinverse or self.n_joints > task_space_dim:
            try:
                M_inv = np.linalg.inv(M)
                Js_Minv_JsT = J_s @ M_inv @ J_s.T
                Js_Minv_JsT += damping * np.eye(task_space_dim)
                Lambda_s = np.linalg.inv(Js_Minv_JsT)
            except np.linalg.LinAlgError:
                J_pinv = J_s.T @ np.linalg.inv(J_s @ J_s.T + damping * np.eye(task_space_dim))
                Lambda_s = J_pinv.T @ M @ J_pinv
        else:
            M_inv = np.linalg.inv(M)
            Lambda_s = np.linalg.inv(J_s @ M_inv @ J_s.T)
        return Lambda_s

    def compute_task_space_coriolis(
        self,
        q: np.ndarray,
        V_s: np.ndarray,
        task_space_dim: int = 6,
        use_pseudoinverse: bool = True,
        damping: float = 1e-6,
    ) -> np.ndarray:
        J_s = self.compute_spatial_jacobian(q, task_space_dim)

        if use_pseudoinverse:
            J_inv = J_s.T @ np.linalg.inv(J_s @ J_s.T + damping * np.eye(task_space_dim))
        else:
            J_inv = np.linalg.pinv(J_s)

        qdot = J_inv @ V_s
        h = self.compute_coriolis_gravity(q, qdot)
        J_dot = self.compute_jacobian_derivative(q, qdot, task_space_dim)
        Lambda_s = self.compute_task_space_mass_matrix(q, task_space_dim, use_pseudoinverse, damping)

        term1 = J_inv.T @ h
        term2 = Lambda_s @ J_dot @ J_inv @ V_s
        eta_s = term1 - term2
        return eta_s

    def get_end_effector_pose(self, q: np.ndarray) -> tuple:
        q_full = self._expand_configuration(q)
        pin.forwardKinematics(self.model, self.data, q_full)
        pin.updateFramePlacements(self.model, self.data)

        oMf = self.data.oMf[self.end_effector_id]
        pos = oMf.translation.copy()
        quat = pin.Quaternion(oMf.rotation).coeffs()  # [x, y, z, w]
        quat_mujoco = np.array([quat[3], quat[0], quat[1], quat[2]])
        return pos, quat_mujoco
