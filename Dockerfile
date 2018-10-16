FROM ros:kinetic
# install ros tutorials packages
RUN apt-get update && apt-get install -y
# RUN rosdep update && rm -rf /var/lib/apt/lists/
RUN apt-get install -y ros-kinetic-cv-bridge ros-kinetic-image-transport ros-kinetic-geographic-msgs python-bloom ros-kinetic-jsk-recognition-msgs
    # ros-kinetic-common-tutorials \
WORKDIR /catkin_ws
# RUN echo "source /catkin_ws/devel/setup.bash" >> /root/.bashrc
# RUN echo "hogehoge" >> /root/.bashrc
