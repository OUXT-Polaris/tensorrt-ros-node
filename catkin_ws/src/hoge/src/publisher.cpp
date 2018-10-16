#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
// http://blog-sk.com/ubuntu/ros_cvbridge/
int main(int argc, char** argv) {
	ros::init (argc, argv, "img_publisher");
	ros::NodeHandle nh("~");
	image_transport::ImageTransport it(nh);
	image_transport::Publisher image_pub = it.advertise("image", 10);
	cv::Mat image;
  image = cv::imread("../data/00.png", 1);
  if(image.empty()) {
    ROS_INFO("failed to load image");
    return -1;
  }
	ros::Rate looprate (10);   // capture image at 10Hz
	while(ros::ok()) {
		sensor_msgs::ImagePtr msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", image).toImageMsg();
		image_pub.publish(msg);
		ros::spinOnce();
		looprate.sleep();
	}
	return 0;
}
