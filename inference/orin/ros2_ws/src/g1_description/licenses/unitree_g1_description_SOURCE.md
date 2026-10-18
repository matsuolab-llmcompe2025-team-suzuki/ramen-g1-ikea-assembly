# Unitree G1 Description Source

The official G1 URDF and meshes in this package were copied from:

https://github.com/unitreerobotics/unitree_ros/tree/master/robots/g1_description

Selected variant:

- g1_29dof_mode_15_with_dex1_1.urdf

Local changes:

- Mesh filenames were rewritten from relative `meshes/...` paths to `package://g1_description/meshes/unitree_g1/...` so RViz can resolve them from the installed ROS package.
- Only meshes referenced by the selected URDF were copied.

Upstream license: BSD 3-Clause. See `unitree_ros_BSD-3-Clause_LICENSE`.
