#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <custom_messages/Num.h>
#include <robotx_msgs/ObjectRegionOfInterestArray.h>
#include <robotx_msgs/ObjectRegionOfInterest.h>
#include <robotx_msgs/ObjectType.h>

void publish_image_and_roi(std::string filename, ros::Publisher roi_pub, image_transport::Publisher image_pub) {
  // 画像生成
	cv::Mat image;
  image = cv::imread(filename, 1);
  if(image.empty()) {
    ROS_INFO("failed to load image");
  }else{
    ROS_INFO("loaded");
  }
  sensor_msgs::ImagePtr msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", image).toImageMsg();

  ROS_INFO("publishing %d %d", image.rows, image.cols);

  // roi生成 画像全体をroiにする
  robotx_msgs::ObjectRegionOfInterestArray array;
  robotx_msgs::ObjectRegionOfInterest roi;
  roi.roi_2d.width = image.cols;
  roi.roi_2d.height = image.rows;
  roi.roi_2d.x_offset = 0;
  roi.roi_2d.y_offset = 0;
  array.object_rois.push_back(roi);

  roi_pub.publish(array);
  image_pub.publish(msg);
}

// http://blog-sk.com/ubuntu/ros_cvbridge/
int main(int argc, char** argv) {
	ros::init (argc, argv, "publisher");
	ros::NodeHandle nh("~");
  ROS_INFO("%s", ros::this_node::getName().c_str());

  // parameter
  std::string filename = "";
  nh.getParam("filename", filename);
  ROS_INFO("filename: %s", filename.c_str());

  ros::Publisher roi_pub = nh.advertise<robotx_msgs::ObjectRegionOfInterestArray>("hogehoge", 1);
	image_transport::ImageTransport it(nh);
	image_transport::Publisher image_pub = it.advertise("image", 1);

	ros::Rate looprate(0.5);   // capture image at 10Hz
	while(ros::ok()) {
		ros::spinOnce();
    publish_image_and_roi(filename, roi_pub, image_pub);
		looprate.sleep();
	}
	return 0;
}
