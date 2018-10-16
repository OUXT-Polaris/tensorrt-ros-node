#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <custom_messages/Num.h>
#include <robotx_msgs/ObjectRegionOfInterestArray.h>

void cb(custom_messages::Num res){
  ROS_INFO("%d", res.num);
}
// http://blog-sk.com/ubuntu/ros_cvbridge/
int main(int argc, char** argv) {
	ros::init (argc, argv, "img_publisher");
	ros::NodeHandle nh("~");
  custom_messages::Num x;
  robotx_msgs::ObjectRegionOfInterestArray arr;
  ros::Subscriber roi_sub = nh.subscribe("object_bbox_extractor_node/object_roi", 1, cb);
	image_transport::ImageTransport it(nh);
	image_transport::Publisher image_pub = it.advertise("wam_v/front_camera/front_image_raw", 1);
	cv::Mat image;
  image = cv::imread("/catkin_ws/src/hoge/data/00.png", 1);
  if(image.empty()) {
    ROS_INFO("failed to load image");
    return -1;
  }else{
    ROS_INFO("loaded");
  }
	ros::Rate looprate(1);   // capture image at 10Hz
	while(ros::ok()) {
    ROS_INFO("published!");
		sensor_msgs::ImagePtr msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", image).toImageMsg();
		image_pub.publish(msg);
		ros::spinOnce();
		looprate.sleep();
	}
	return 0;
}
