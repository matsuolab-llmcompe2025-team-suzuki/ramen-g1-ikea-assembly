from setuptools import find_packages, setup

package_name = "g1_dataset_replay"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="mimi",
    maintainer_email="mimi@example.com",
    description="Replay LeRobot-format G1 IKEA assembly dataset states to ROS topics.",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "dataset_replay_node = g1_dataset_replay.dataset_replay_node:main",
        ],
    },
)
