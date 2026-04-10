from setuptools import find_packages, setup


package_name = "fr3_force_control_ros2"


setup(
    name=package_name,
    version="0.0.1",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Codex",
    maintainer_email="dev@example.com",
    description="ROS 2 wrapper for FR3 variable joint impedance admittance control in MuJoCo.",
    license="MIT",
    entry_points={
        "console_scripts": [
            "variable_joint_impedance_admittance_node = fr3_force_control_ros2.variable_joint_impedance_admittance_node:main",
        ],
    },
)
