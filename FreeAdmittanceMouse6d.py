#!/usr/bin/env python3
import mujoco
import mujoco.viewer
import numpy as np
from scipy.constants import nano
from scipy.spatial.transform import Rotation

from dynamics_calculator_pinocchio import DynamicsCalculatorPinocchio

# adm_pos / adm_vel 先由外力更新
# 然后进入任务空间控制
# 通过 K_p / K_d 和 S / S_c 生成末端任务力
# 再映射成关节力矩 tau
# 所以它本质上还是“有主动跟踪/约束”的控制器

# M_adm 决定“外力推一下有多容易加速”。越大越沉，越不容易被拖动。
# D_adm 决定“运动时的阻尼/粘滞感”。越大越稳，但拖动会更重、更慢。
# K_adm 决定“回弹/回中”效果。为 0 时，没有弹簧回复力；外力消失后不会因为导纳弹簧自己拉回某个目标。

class FreeAdmittanceMouse:
    WRENCH_AXES = ("fx", "fy", "fz", "mx", "my", "mz")
    POSITION_AXES = ("x", "y", "z")
    ROTATION_AXES = ("rx", "ry", "rz")
    ADMITTANCE_AXES = POSITION_AXES + ROTATION_AXES
    TASK_ERROR_AXES = ROTATION_AXES + POSITION_AXES

    def __init__(
        self,
        model_path="scene.xml",
        urdf_path="fr3v2.urdf",
        ee_body_name="fr3v2_link7",
        ee_frame_name="fr3v2_link8",
        motion_mask=None,  #修改掩码映射
        # motion_mask=[1, 0, 0, 0, 0, 0],
    ):
        self.task_motion_mask = None
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

        self.K_p = np.diag([60.0, 60.0, 70.0, 100.0, 100.0, 100.0])
        self.K_d = np.diag([12.0, 12.0, 14.0, 20.0, 20.0, 20.0])
        self.K_adm = np.diag([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.M_adm = np.diag([5.0, 5.0, 5.0, 1.0, 1.0, 1.0])
        self.D_adm = np.diag([80.0, 80.0, 80.0, 20.0, 20.0, 20.0])
        self.K_constraint = np.diag([2000.0, 2000.0, 2000.0, 2000.0, 2000.0, 2000.0])
        self.D_constraint = np.diag([50.0, 50.0, 50.0, 50.0, 50.0, 50.0])

        self.max_pos_offset = 0.1
        self.max_rot_offset = 0.5
        self.force_deadband = 1e-3

        self.adm_vel = np.zeros(6)
        self.adm_pos = np.zeros(6)
        self.set_motion_mask(motion_mask)

        self.q_init = self.model.keyframe("home").qpos.copy()[:7]
        for i, adr in enumerate(self.qpos_adr):
            self.data.qpos[adr] = self.q_init[i]
        mujoco.mj_forward(self.model, self.data)
        self.init_pos = self.get_ee_pose()[0]

    # -------------------- public interface --------------------
    def set_motion_mask(self, mask=None):
        if mask is None:
            mask = np.ones(6)
        mask_array = np.asarray(mask, dtype=float).reshape(6)
        self.motion_mask = np.clip(mask_array, 0.0, 1.0)
        self.task_motion_mask = self._admittance_mask_to_task_mask(self.motion_mask)

    def set_direction_mask(
        self,
        x=1.0,
        y=0.0,
        z=0.0,
        rx=1.0,
        ry=1.0,
        rz=1.0,
    ):
        self.set_motion_mask([x, y, z, rx, ry, rz])

    def set_translation_mask(self, x=1.0, y=1.0, z=1.0):
        self.motion_mask[:3] = np.clip(np.asarray([x, y, z], dtype=float), 0.0, 1.0)
        self.task_motion_mask = self._admittance_mask_to_task_mask(self.motion_mask)

    def set_rotation_mask(self, rx=1.0, ry=1.0, rz=1.0):
        self.motion_mask[3:] = np.clip(np.asarray([rx, ry, rz], dtype=float), 0.0, 1.0)
        self.task_motion_mask = self._admittance_mask_to_task_mask(self.motion_mask)

    def set_task_gains(self, kp=None, kd=None):
        if kp is not None:
            self.K_p = np.diag(np.asarray(kp, dtype=float).reshape(6))
        if kd is not None:
            self.K_d = np.diag(np.asarray(kd, dtype=float).reshape(6))

    def set_admittance_gains(self, mass=None, damping=None, stiffness=None):
        if mass is not None:
            self.M_adm = np.diag(np.asarray(mass, dtype=float).reshape(6))
        if damping is not None:
            self.D_adm = np.diag(np.asarray(damping, dtype=float).reshape(6))
        if stiffness is not None:
            self.K_adm = np.diag(np.asarray(stiffness, dtype=float).reshape(6))

    def set_constraint_gains(self, stiffness=None, damping=None):
        if stiffness is not None:
            self.K_constraint = np.diag(np.asarray(stiffness, dtype=float).reshape(6))
        if damping is not None:
            self.D_constraint = np.diag(np.asarray(damping, dtype=float).reshape(6))

    def get_motion_mask_info(self):
        return {
            "wrench_axes": self.WRENCH_AXES,
            "admittance_axes": self.ADMITTANCE_AXES,
            "task_error_axes": self.TASK_ERROR_AXES,
            "mask": self.motion_mask.copy(),
            "task_mask": self.task_motion_mask.copy(),
            "direction_mask_axes": ("x", "y", "z", "rx", "ry", "rz"),
        }

    def compute_control(self, dt):
        q, qdot = self.get_joint_state()
        pos, quat = self.get_ee_pose()
        wrench_ext = self.read_external_wrench()

        self.update_admittance(dt, wrench_ext)
        pos_d, quat_d = self.build_target_pose()
        tau = self.solve_task_torque(q, qdot, pos, quat, pos_d, quat_d)
        return tau

    def apply_control(self, tau):
        self.data.ctrl[:7] = tau

    # -------------------- helpers --------------------
    def get_joint_state(self):
        q = np.array([self.data.qpos[adr] for adr in self.qpos_adr])
        qdot = np.array([self.data.qvel[adr] for adr in self.qvel_adr])
        return q, qdot

    def get_ee_pose(self):
        body = self.data.body(self.ee_body_name)
        return np.array(body.xpos), np.array(body.xquat)

    def read_external_wrench(self):
        return self.data.xfrc_applied[self.ee_body_id].copy()

    @staticmethod
    def _quat_wxyz_to_xyzw(quat_wxyz):
        return np.roll(quat_wxyz, -1)

    @staticmethod
    def _admittance_mask_to_task_mask(mask):
        mask = np.asarray(mask, dtype=float).reshape(6)
        return np.array([mask[3], mask[4], mask[5], mask[0], mask[1], mask[2]], dtype=float)

    def compute_spatial_error(self, p, q, p_d, q_d):
        rc = Rotation.from_quat(self._quat_wxyz_to_xyzw(q)).as_matrix()
        rd = Rotation.from_quat(self._quat_wxyz_to_xyzw(q_d)).as_matrix()
        r_err = rd @ rc.T
        omega = Rotation.from_matrix(r_err).as_rotvec()
        pos_e = p_d - p
        return np.concatenate([omega, pos_e])

    def update_admittance(self, dt, wrench_ext):
        wrench_eff = wrench_ext * self.motion_mask

        if np.linalg.norm(wrench_eff) > self.force_deadband:
            acc = np.linalg.solve(
                self.M_adm,
                wrench_eff - self.D_adm @ self.adm_vel - self.K_adm @ self.adm_pos,
            )
            self.adm_vel += acc * dt
            self.adm_pos += self.adm_vel * dt
            self.adm_pos[:3] = np.clip(
                self.adm_pos[:3],
                -self.max_pos_offset,
                self.max_pos_offset,
            )
            self.adm_pos[3:] = np.clip(
                self.adm_pos[3:],
                -self.max_rot_offset,
                self.max_rot_offset,
            )
            self.adm_vel *= self.motion_mask
            self.adm_pos *= self.motion_mask
            return

        self.adm_vel[:] = 0.0
        current_pos, _ = self.get_ee_pose()
        self.init_pos = current_pos - self.adm_pos[:3]
        self.adm_pos[:] = 0.0

    def build_target_pose(self):
        pos_d = self.init_pos + self.adm_pos[:3]
        quat_current = self.get_ee_pose()[1]

        rot_offset = Rotation.from_rotvec(self.adm_pos[3:])
        rot_current = Rotation.from_quat(self._quat_wxyz_to_xyzw(quat_current))
        rot_target = rot_offset * rot_current
        quat_d = np.roll(rot_target.as_quat(), 1)

        # Preserve the original behavior: keep current orientation locked.
        quat_d = quat_current.copy()
        return pos_d, quat_d

    def solve_task_torque(self, q, qdot, pos, quat, pos_d, quat_d):
        x_e = self.compute_spatial_error(pos, quat, pos_d, quat_d)
        v_jacobian = self.dynamics_calc.compute_spatial_jacobian(q, 6)
        v = v_jacobian @ qdot
        lambda_task = self.dynamics_calc.compute_task_space_mass_matrix(q, 6)
        v_e = -v

        s = np.diag(self.task_motion_mask)
        s_c = np.eye(6) - s
        f_motion = s @ (self.K_p @ x_e + self.K_d @ v_e)
        f_constraint = s_c @ (self.K_constraint @ x_e + self.D_constraint @ v_e)
        f_task = lambda_task @ (f_motion + f_constraint)
        tau = v_jacobian.T @ f_task + self.dynamics_calc.compute_coriolis_gravity(q, qdot)
        return tau

    # -------------------- main loop --------------------
    def control_step(self, dt):
        tau = self.compute_control(dt)
        self.apply_control(tau)

    def run(self):
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            viewer.cam.trackbodyid = self.ee_body_id
            while viewer.is_running():
                dt = self.model.opt.timestep
                self.control_step(dt)
                mujoco.mj_step(self.model, self.data)
                viewer.sync()


if __name__ == "__main__":
    sim = FreeAdmittanceMouse()
    sim.run()
