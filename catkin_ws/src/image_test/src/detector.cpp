/* detector
 *
 * @node_name: cnn_prediction
 * @in: wam_v/front_camera/front_image_raw (Image)
 * @in: object_bbox_extractor_node/object_roi (OROIA)
 * @out: cnn_prediction_node/object_roi (ObjectRegionOfInterestArr)
 * ref: https://ipx.hatenablog.com/entry/2018/05/21/102659
 */
// message
#include <ros/ros.h>
#include <detector.h>
#include <string>

/* extern void setup(std::string planFilename, std::string inputName, std::string outputName); */
extern void setup();
extern void destroy(void);
extern void infer(cv::Mat image, float* out);
extern void test(void);

typedef message_filters::sync_policies::ApproximateTime<> SyncPolicy;

// constructor
cnn_predictor::cnn_predictor() : _it(_nh) {
  _roi_pub   = _nh.advertise<robotx_msgs::ObjectRegionOfInterestArray>("cnn_prediction_node/object_roi", 1);

  // filterを使う場合
  /* message_filters::Subscriber<Image> _image_sub(nh, "publisher/image", 1); */
  /* image_transport::SubscriberFilter _image_sub(_it, "publisher/image", 1); */
  /* message_filters::Subscriber<robotx_msgs::ObjectRegionOfInterestArray> _roi_sub(_nh, "publisher/hogehoge", 1); */
  /* TimeSynchronizer<Image, CameraInfo> _sync(_image_sub, _roi_sub, 10); */
  /* sync.registerCallback(boost::bind(&callback, _1, _2)); */

  _image_sub = _it.subscribe("publisher/image",    1, &cnn_predictor::_image_callback, this);
  _roi_sub   = _nh.subscribe("publisher/hogehoge", 1, &cnn_predictor::_roi_callback, this);
  // TODO paramを読み込むようにする
  ROS_INFO("inited");
  _image_stored = false;
  _rois_stored = false;
  // tensorrt 初期化
  /* setup("/home/ubuntu/tensorrt/resnet_test/resnet_v1_50_finetuned_4class_altered_model.plan", */
        /* "images", "resnet_v1_50/SpatialSqueeze"); */
  setup();
  // TODO ブイの情報 mapper
}
// destructor
cnn_predictor::~cnn_predictor() {
  destroy();
}

// callbacks
void cnn_predictor::_image_callback(const sensor_msgs::ImageConstPtr& msg) {
  // 画像が入ってきたときのコールバック こちらは頻度が低いことが予想できるので、とりあえず保存しておく
  // 画像のcopy
  cv::Mat image;
  try {
    image = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8)->image;
    cv::cvtColor(image, image, cv::COLOR_BGR2RGB, 3);
  } catch (cv_bridge::Exception& e) {
    ROS_ERROR("cv_bridge exception: %s", e.what());
    return;
  }
  ROS_INFO("got image %d, %d", image.rows, image.cols);

  // store
  if(!_image_stored) {
    _image_timestamp = msg->header.stamp;
    _image = image;
  } else if () {
    // タイムスタンプが古い時 更新する

  }
}

// 今ある画像に対応するroiを選んで、CNNで判定した結果をくっつけて再送信する
void cnn_predictor::_roi_callback(const robotx_msgs::ObjectRegionOfInterestArray msg) {
  ROS_INFO("got roi");
  ROS_INFO("objectness: %f", msg.object_rois[0].objectness);
  // 実行時間の一致確認
  if(msg.object_rois[0].header.stamp == _image_timestamp) {
    ROS_INFO("same stamp");
    _process(msg, _image);
  } else {
    ROS_INFO("different, rejected");
  }
}

void cnn_predictor::_process(const robotx_msgs::ObjectRegionOfInterestArray msg, const cv::Mat image) {
  robotx_msgs::ObjectRegionOfInterestArray res = _image_recognition(msg, _image);
  // 判定結果を送信する
  _roi_pub.publish(res);

}

// 画像認識: 基本的にはroisのアップデートをする感じ (rois, image) -> (rois)
robotx_msgs::ObjectRegionOfInterestArray cnn_predictor::_image_recognition(const robotx_msgs::ObjectRegionOfInterestArray rois, const cv::Mat image) {
  ROS_INFO("tensorrt recognition for roi #%d", rois.object_rois.size());
  robotx_msgs::ObjectRegionOfInterestArray res;
  for (int i = 0; i < rois.object_rois.size(); i++) {
    robotx_msgs::ObjectRegionOfInterest roi = rois.object_rois[i];
    robotx_msgs::ObjectRegionOfInterest roi_alt;
    roi_alt.roi_2d = roi.roi_2d;  // TODO これでいける？
    // 矩形領域の切り出し
    ROS_INFO("h:%d, w:%d, x:%d, y:%d, H:%d, W:%d", roi.roi_2d.height, roi.roi_2d.width, roi.roi_2d.x_offset, roi.roi_2d.y_offset, image.rows, image.cols);
    cv::Rect rect(cv::Point(roi.roi_2d.x_offset, roi.roi_2d.y_offset),
                  cv::Size(roi.roi_2d.width, roi.roi_2d.height));
    /* ROS_INFO("size: %d,%d,%d,%d  full:%d,%d", roi.roi_2d.x_offset, roi.roi_2d.y_offset, roi.roi_2d.width, roi.roi_2d.height, image.rows, image.cols); */
    cv::Mat subimage = image(rect);
    // 切り出した部分をCNNのinputサイズに合わせる
    // 切り出したところについてtensorrtで確率を計算
    float result[4];
    infer(subimage, result);
    ROS_INFO("probability %f,%f,%f,%f", result[0], result[1], result[2], result[3]);
    float max_prob = -1000;
    int index = -1;
    for (int i = 0; i < 4; i++) {
      if (max_prob < result[i]) {
        max_prob = result[i];
        index = i;
      }
    }
    switch(index) {
      case 0: 
        roi_alt.object_type.ID = roi_alt.object_type.GREEN_BUOY;
        break;
      case 1:
        roi_alt.object_type.ID = roi_alt.object_type.RED_BUOY;
        break;
      case 2:
        roi_alt.object_type.ID = roi_alt.object_type.WHITE_BUOY;
        break;
      case 3:
        roi_alt.object_type.ID = roi_alt.object_type.OTHER;
        break;
    }
    roi_alt.objectness = result[index];
    res.object_rois.push_back(roi_alt);
  }
  return res;
}

/* main */
// http://blog-sk.com/ubuntu/ros_cvbridge/
int main(int argc, char *argv[]) {
  ros::init(argc, argv, "detector");
  cnn_predictor cnn_predictor;
  ros::spin();
  return 0;
}

