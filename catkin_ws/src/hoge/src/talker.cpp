#include "ros/ros.h"
#include "std_msgs/String.h"
#include <sstream>

int main(int argc, char **argv) {
  // talker: node name
  ros::init(argc, argv, "talker");

  // ノードハンドラ
  ros::NodeHandle n;
  ros::Publisher chatter_pub = n.advertise<std_msgs::String>("chatter", 1000);
  ros::Rate loop_rate(10); // Rate::sleep()の実行間隔を管理する 10Hz

  int count = 0;
  while(ros::ok()) { // 実行中 C-cが入るとfalseになる
    std_msgs::String msg;
    std::stringstream ss;
    ss<<"hello world "<<count;
    msg.data = ss.str();
    ROS_INFO("%s", msg.data.c_str());

    // 発信
    chatter_pub.publish(msg);

    ros::spinOnce();
    loop_rate.sleep();
    ++count;
  }

  return 0;
}
