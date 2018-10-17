build:
	docker run -it --rm \
		-v ~/src/OUXT/docker-test/node-making/catkin_ws:/catkin_ws \
		ros:ros-node-making \
		/bin/bash -c ". devel/setup.bash; rosdep install -i -r -y --from-paths src --rosdistro kinetic; rospack find cv_bridge; catkin_make"

run_master:
	# docker rm master
	docker run -it --rm \
		--net rosnet --name master \
		ros:ros-node-making roscore

run_slave:
	# docker rm talker
	docker run -it --rm \
		--net rosnet --name talker \
		--env ROS_HOSTNAME=talker \
		--env ROS_MASTER_URI=http://master:11311 \
		-v ~/src/OUXT/docker-test/node-making/catkin_ws:/catkin_ws \
		ros:ros-node-making \
		/bin/bash -c ". devel/setup.bash; rosrun image_test publisher"
		# /bin/bash -c ". devel/setup.bash; rosmsg show hoge/Num"

run_listener:
	docker run -it --rm \
		--net rosnet --name listener \
		--env ROS_HOSTNAME=listener \
		--env ROS_MASTER_URI=http://master:11311 \
		-v ~/src/OUXT/docker-test/node-making/catkin_ws:/catkin_ws \
		ros:ros-node-making \
		/bin/bash -c ". devel/setup.bash; rosrun image_test detector"
		# /bin/bash -c ". devel/setup.bash; rosmsg show hoge/Num"

inspect:
	docker run -it --rm \
		--net rosnet --name debug \
		--env ROS_HOSTNAME=debug \
		--env ROS_MASTER_URI=http://master:11311 \
		-v ~/src/OUXT/docker-test/node-making/catkin_ws:/catkin_ws \
		ros:ros-node-making \
		/bin/bash -c ". devel/setup.bash; rosnode info /publisher; rosnode info /hoge; rosnode info /detector"

launch:
	docker run -it --rm \
		--net rosnet --name debug \
		--env ROS_HOSTNAME=debug \
		--env ROS_MASTER_URI=http://master:11311 \
		-v ~/src/OUXT/docker-test/node-making/catkin_ws:/catkin_ws \
		ros:ros-node-making \
		/bin/bash -c ". devel/setup.bash; export HOGE=hoge; roslaunch image_test run.launch CUDA_ENABLED:=true"
