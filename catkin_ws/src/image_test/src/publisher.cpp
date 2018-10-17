#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <custom_messages/Num.h>
#include <robotx_msgs/ObjectRegionOfInterestArray.h>
#include <robotx_msgs/ObjectRegionOfInterest.h>
#include <robotx_msgs/ObjectType.h>

void cb(custom_messages::Num res){
  /* ROS_INFO("%d", res.num); */
}
// http://blog-sk.com/ubuntu/ros_cvbridge/
int main(int argc, char** argv) {
	ros::init (argc, argv, "publisher");
	ros::NodeHandle nh("~");
  ROS_INFO("%s", ros::this_node::getName().c_str());

  ros::Publisher roi_pub = nh.advertise<robotx_msgs::ObjectRegionOfInterestArray>("hogehoge", 1);
	image_transport::ImageTransport it(nh);
	image_transport::Publisher image_pub = it.advertise("image", 1);

  // 画像生成
	cv::Mat image;
  image = cv::imread("/home/ubuntu/tensorrt-ros-node/catkin_ws/src/hoge/data/00.png", 1);
  if(image.empty()) {
    ROS_INFO("failed to load image");
    return -1;
  }else{
    ROS_INFO("loaded");
  }
  sensor_msgs::ImagePtr msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", image).toImageMsg();

  // roi生成
  robotx_msgs::ObjectRegionOfInterestArray array;
  robotx_msgs::ObjectRegionOfInterest roi;
  roi.roi_2d.width = 42;
  roi.roi_2d.height = 95;
  roi.roi_2d.x_offset = 0;
  roi.roi_2d.y_offset = 0;
  /* roi.objectness = 0.9; */
  /* roi.object_type.ID = roi.object_type.GREEN_BUOY; */
  array.object_rois.push_back(roi);

	ros::Rate looprate(0.1);   // capture image at 10Hz
	while(ros::ok()) {
    ROS_INFO("published!");
    roi_pub.publish(array);
		image_pub.publish(msg);
    ROS_INFO("%f", array.object_rois[0].objectness);
		ros::spinOnce();
		looprate.sleep();
	}
	return 0;
}
