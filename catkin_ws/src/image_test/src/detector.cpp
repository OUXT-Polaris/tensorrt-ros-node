/* detector
 *
 * @node_name: cnn_prediction
 * @in: wam_v/front_camera/front_image_raw (Image)
 * @in: object_bbox_extractor_node/object_roi (OROIA)
 * @out: cnn_prediction_node/object_roi (ObjectRegionOfInterestArr)
 */
// message
#include <ros/ros.h>
#include <detector.h>

// constructor
cnn_predictor::cnn_predictor() : _it(_nh) {
  _roi_pub   = _nh.advertise<robotx_msgs::ObjectRegionOfInterestArray>("cnn_prediction_node/object_roi", 1);
  _image_sub = _it.subscribe("publisher/image",    1, &cnn_predictor::_image_callback, this);
  _roi_sub   = _nh.subscribe("publisher/hogehoge", 1, &cnn_predictor::_roi_callback, this);
  // TODO paramを読み込むようにする
  ROS_INFO("inited");

  // TODO ブイの情報
  // TODO tensorrtの初期化
}
// destructor
cnn_predictor::~cnn_predictor() {}

// callbacks
void cnn_predictor::_image_callback(const sensor_msgs::ImageConstPtr& msg) {
  // 画像が入ってきたときのコールバック こちらは頻度が低いことが予想できるので、とりあえず保存しておく
  // 画像のcopy
  cv::Mat image;
  try {
    image = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8)->image;
  } catch (cv_bridge::Exception& e) {
    ROS_ERROR("cv_bridge exception: %s", e.what());
    return;
  }
  ROS_INFO("got image");

  // store
  _image_timestamp = msg->header.stamp;
  _image = image;
}

// 今ある画像に対応するroiを選んで、CNNで判定した結果をくっつけて再送信する
void cnn_predictor::_roi_callback(const robotx_msgs::ObjectRegionOfInterestArray msg) {
  ROS_INFO("got roi");
  ROS_INFO("objectness: %f", msg.object_rois[0].objectness);
  // 実行時間の一致確認
  if(msg.object_rois[0].header.stamp == _image_timestamp) {
    ROS_INFO("same stamp");
    robotx_msgs::ObjectRegionOfInterestArray res = _image_recognition(msg, _image);
    // 判定結果を送信する
    _roi_pub.publish(res);
  } else {
    ROS_INFO("different, rejected");
  }
}

robotx_msgs::ObjectRegionOfInterestArray cnn_predictor::_image_recognition(const robotx_msgs::ObjectRegionOfInterestArray rois, const cv::Mat image) {
  ROS_INFO("tensorrt");
  robotx_msgs::ObjectRegionOfInterestArray res;
  for (int i = 0; i < rois.object_rois.size(); i++) {
    robotx_msgs::ObjectRegionOfInterest roi = rois.object_rois[i];
    robotx_msgs::ObjectRegionOfInterest roi_alt;
    roi_alt.roi_2d = roi.roi_2d;  // TODO これでいける？
    // 矩形領域の切り出し
    cv::Rect rect(cv::Point(roi.roi_2d.x_offset, roi.roi_2d.y_offset),
                  cv::Size(roi.roi_2d.width, roi.roi_2d.height));
    cv::Mat subimage = image(rect);
    // 切り出したところについてtensorrtで確率を計算
    int r = _infer(subimage);
    roi_alt.object_type.ID = roi_alt.object_type.OTHER;
    roi_alt.objectness = 1.0;

    res.object_rois.push_back(roi_alt);
  }
  return res;
}

int cnn_predictor::_infer(const cv::Mat image) {
  // 画像を元に推論する。object typeを返す？
  // tensorrtのやつ
  return 0;
}


/* main */
// http://blog-sk.com/ubuntu/ros_cvbridge/

int main(int argc, char *argv[]) {
  ros::init(argc, argv, "detector");
  cnn_predictor cnn_predictor;
  ros::spin();
  return 0;
}

