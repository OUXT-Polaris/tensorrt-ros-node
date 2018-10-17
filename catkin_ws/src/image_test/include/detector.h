#include <robotx_msgs/ObjectRegionOfInterestArray.h>
#include <robotx_msgs/ObjectRegionOfInterest.h>
#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <vector>

class cnn_predictor {
  public:
    cnn_predictor();
    ~cnn_predictor();
  private:
    ros::NodeHandle _nh;
    image_transport::ImageTransport _it;
    // publisher
    ros::Publisher  _roi_pub;
    // subscriber
    ros::Subscriber _roi_sub;
    image_transport::Subscriber _image_sub;

    // callbacks
    void _image_callback(const sensor_msgs::ImageConstPtr& msg);
    void _roi_callback(const robotx_msgs::ObjectRegionOfInterestArray msg);

    // stores
    ros::Time _image_timestamp;
    cv::Mat _image;
    robotx_msgs::ObjectRegionOfInterestArray _rois;

    // functions
    int _infer(const cv::Mat image);
    robotx_msgs::ObjectRegionOfInterestArray _image_recognition(const robotx_msgs::ObjectRegionOfInterestArray rois, const cv::Mat image);
};
