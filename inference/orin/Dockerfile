FROM ros:humble-ros-base

ENV DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-c"]

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-colcon-common-extensions \
    python3-numpy \
    python3-rosdep \
    python3-pip \
    ros-humble-rmw-cyclonedds-cpp \
    ros-humble-robot-state-publisher \
    ros-humble-rviz2 \
    ros-humble-usb-cam \
    ros-humble-image-transport-plugins \
    ros-humble-realsense2-camera \
    v4l-utils \
    git \
    && rm -rf /var/lib/apt/lists/*

# unitree_sdk2py が cyclonedds==0.10.2 (cp310 wheel only) を hard pin する。
# ros:humble-ros-base = Ubuntu 22.04 = Python 3.10 default で cp310 wheel と互換。
# ARM64 の ROS apt package は libddsc.so を multiarch directory に配置するが、
# cyclonedds-python 0.10.2 の source build は <prefix>/{include,bin,lib} を要求する。
# ROS の実体を複製せず、build/runtime 共通の prefix を symlink で用意する。
ENV CYCLONEDDS_HOME=/opt/cyclonedds
RUN DDS_LIB_DIR="$(dirname "$(find /opt/ros/humble/lib -name libddsc.so -print -quit)")" \
    && test -n "${DDS_LIB_DIR}" \
    && mkdir -p "${CYCLONEDDS_HOME}" \
    && ln -s /opt/ros/humble/include "${CYCLONEDDS_HOME}/include" \
    && ln -s /opt/ros/humble/bin "${CYCLONEDDS_HOME}/bin" \
    && ln -s "${DDS_LIB_DIR}" "${CYCLONEDDS_HOME}/lib"
RUN pip3 install --no-cache-dir \
    pyarrow==19.0.1 \
    cyclonedds==0.10.2

# unitree_sdk2py: pixi runtime env (path-editable) と同じ commit を container 内で
# git clone → editable install。host 側の third_party/ 有無に依存しない
# (third_party/ は .gitignore 済で fresh clone には無いため COPY は不可)。
# commit SHA は pixi runtime env の checkout と一致させる (2026-07-15 時点、
# unitree_sdk2_python setup.py version 1.0.1)。
ARG UNITREE_SDK2PY_SHA=37116c521f1588482e238d8450e471ba78ab9863
RUN git clone https://github.com/unitreerobotics/unitree_sdk2_python.git /opt/unitree_sdk2_python \
    && cd /opt/unitree_sdk2_python \
    && git checkout ${UNITREE_SDK2PY_SHA} \
    && pip3 install --no-cache-dir -e .

WORKDIR /workspace
COPY . /workspace

RUN source /opt/ros/humble/setup.bash \
    && rosdep update || true \
    && rosdep install --from-paths ros2_ws/src --ignore-src --skip-keys "ament_python" -r -y \
    && cd ros2_ws && colcon build --symlink-install

COPY entrypoint.sh /docker_entrypoint.sh
RUN chmod +x /docker_entrypoint.sh

ENTRYPOINT ["/docker_entrypoint.sh"]
CMD ["bash"]
