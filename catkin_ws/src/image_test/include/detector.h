#ifndef CNN_DETECTOR_H
#define CNN_DETECTOR_H

#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <vector>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <image_transport/subscriber_filter.h>

// messages
#include <robotx_msgs/ObjectRegionOfInterestArray.h>
#include <robotx_msgs/ObjectRegionOfInterest.h>
#include <sensor_msgs/Image.h>

class cnn_predictor {
  public:
    cnn_predictor();
    ~cnn_predictor();
    void callback(const sensor_msgs::ImageConstPtr& image_msg,
                  const robotx_msgs::ObjectRegionOfInterestArrayConstPtr& rois_msg);

  private:
    ros::NodeHandle _nh;
    image_transport::ImageTransport _it;

    // publisher
    ros::Publisher  _roi_pub;

    // subscriber and synchronizer
    message_filters::Subscriber<robotx_msgs::ObjectRegionOfInterestArray> _roi_sub;
    image_transport::SubscriberFilter _image_sub;

    typedef message_filters::sync_policies::ApproximateTime<
      sensor_msgs::Image,
      robotx_msgs::ObjectRegionOfInterestArray> SyncPolicy;
    message_filters::Synchronizer<SyncPolicy> _sync;

    // functions
    robotx_msgs::ObjectRegionOfInterestArray _image_recognition(
        const robotx_msgs::ObjectRegionOfInterestArray rois,
        const cv::Mat image);
};
#endif
