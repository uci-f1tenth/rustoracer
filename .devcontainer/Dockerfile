FROM docker.io/ros:jazzy-ros-core

RUN curl https://sh.rustup.rs -sSf | sh -s -- -y
RUN apt-get update && apt-get install -y gcc libclang-dev ros-$ROS_DISTRO-ackermann-msgs ros-$ROS_DISTRO-tf2-msgs ros-$ROS_DISTRO-foxglove-bridge ros-$ROS_DISTRO-slam-toolbox ros-$ROS_DISTRO-tf-transformations ros-$ROS_DISTRO-cv-bridge python3-colcon-common-extensions ros-$ROS_DISTRO-nav2-map-server python3-pip git
RUN pip3 install "greenlet>=3.0.0" "gevent>=23.9.0" "eventlet>=0.35.0" Flask==1.1.1 Flask-SocketIO==4.1.0 python-socketio==4.2.0 python-engineio==3.13.0 gevent-websocket==0.10.1 itsdangerous==2.0.1 werkzeug==2.0.3 transforms3d pillow --break-system-packages
RUN echo "source /opt/ros/$ROS_DISTRO/setup.bash" >> ~/.bashrc
