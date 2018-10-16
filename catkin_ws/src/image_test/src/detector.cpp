/* detector
 *
 * @node_name: cnn_prediction
 * @in: wam_v/front_camera/front_image_raw (Image)
 * @in: object_bbox_extractor_node/object_roi (OROIA)
 * @out: cnn_prediction_node/object_roi (ObjectRegionOfInterestArr)
 */
// message
/* #include <robotx_msgs/ObjectRegionOfInterestArray.h> */
#include <ros/ros.h>
#include <detector.h>
/*
void image_callback(const sensor_msgs::ImageConstPtr& msg) {
  // 画像を登録
  cv::Mat image = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8)->image;
  // timestamp
}
*/

/*
void roi_callback(const robotx_msgs::ObjectRegionOfInterestArray msg) {
  ROS_INFO("got roi");
  robotx_msgs::ObjectRegionOfInterestArray res;

  for (int i = 0; i < msg.object_rois.size(); i++) {
    robotx_msgs::ObjectRegionOfInterest r = msg.object_rois[i];

    // 返信
    robotx_msgs::RegionOfInterest2D roi_msg;
    roi_msg.roi_2d.x_offset = x;
    roi_msg.roi_2d.y_offset = y;
    roi_msg.roi_2d.width    = w;
    roi_msg.roi_2d.height   = h;
    roi_msg.roi_2d.objectness = 0.9;
    roi_msg.roi_2d.object_type = 1;

    // 一つだけ追加
    res.object_rois.push_back(roi_msg);
  }
}
*/

// constructor
cnn_predictor::cnn_predictor() : _it(_nh) {
  _roi_pub   = _nh.advertise<robotx_msgs::ObjectRegionOfInterestArray>("cnn_prediction_node/object_roi", 1);
  _image_sub = _it.subscribe("wam_v/front_camera/front_image_raw",    1, &cnn_predictor::_image_callback, this);
  _roi_sub   = _nh.subscribe("object_bbox_extractor_node/object_roi", 1, &cnn_predictor::_roi_callback, this);
  // TODO paramを読み込むようにする
}
// destructor
cnn_predictor::~cnn_predictor() {}

// callbacks
void cnn_predictor::_image_callback(const sensor_msgs::ImageConstPtr& msg) {
  // 画像のcopy
  cv::Mat image;
  try {
    image = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8)->image;
  } catch (cv_bridge::Exception& e) {
    ROS_ERROR("cv_bridge exception: %s", e.what());
    return;
  }
  ROS_INFO("got image");
  // 実行時間の一致確認
  // publish
  /* robotx_msgs::ObjectRegionOfInterestArray res; */
  /* _roi_pub.publish(res); */
}
void cnn_predictor::_roi_callback(const robotx_msgs::ObjectRegionOfInterestArray msg) {
  ROS_INFO("got roi");
}



/* main */

int main(int argc, char *argv[]) {
  ros::init(argc, argv, "detector");
  cnn_predictor cnn_predictor;
  ros::spin();
  return 0;
}

/*
// http://blog-sk.com/ubuntu/ros_cvbridge/
int main(int argc, char** argv) {
  // 初期化
	ros::init (argc, argv, "detector");
	ros::NodeHandle nh;

  //// 受信部分
  // 1. 画像
	image_transport::ImageTransport it(nh);
	image_transport::Subscriber image_sub = it.subscribe("wam_v/front_camera/front_image_raw", 1, image_callback);
  // 2. ROI
  ros::Subscriber roi_sub = nh.subscribe("object_bbox_extractor_node/object_roi", 1, image_callback);

  //// 送信部分
  ros::Publisher roi_pub = nh.advertise<robotx_msgs::ObjectRegionOfInterestArray>("cnn_prediction_node/object_roi", 1);

  // 実行
  ros::spin();
	return 0;
}
*/
