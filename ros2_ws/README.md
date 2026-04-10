# ROS 2 Workspace

This workspace wraps the FR3 MuJoCo controller into a ROS 2 Python node.

Package:
- `fr3_force_control_ros2`

Node:
- `variable_joint_impedance_admittance_node`

Topics:
- Sub `/fr3_force_control/command/motion_mask` `std_msgs/Float64MultiArray`
- Sub `/fr3_force_control/command/admittance_gains` `std_msgs/Float64MultiArray`
- Sub `/fr3_force_control/command/debug_interval` `std_msgs/Float64`
- Pub `/fr3_force_control/state/q_ref` `std_msgs/Float64MultiArray`
- Pub `/fr3_force_control/state/kp` `std_msgs/Float64MultiArray`
- Pub `/fr3_force_control/state/kd` `std_msgs/Float64MultiArray`
- Pub `/fr3_force_control/state/wrench` `std_msgs/Float64MultiArray`
- Pub `/fr3_force_control/state/adm_pos` `std_msgs/Float64MultiArray`

Services:
- `/fr3_force_control/reset_reference` `std_srvs/Trigger`
- `/fr3_force_control/set_debug_print` `std_srvs/SetBool`

Expected array layout:
- `motion_mask`: `[x, y, z, rx, ry, rz]`
- `admittance_gains`: `[M(6), D(6), K(6)]`

Build:
```bash
cd ros2_ws
colcon build
source install/setup.bash
```

Run:
```bash
ros2 run fr3_force_control_ros2 variable_joint_impedance_admittance_node
```
