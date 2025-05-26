# 文件：panda_image_viewer.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from common_msgs.msg import ArmDataCollect
import numpy as np


class PandaImageViewer(Node):
    def __init__(self):
        super().__init__("panda_image_viewer")
        self.bridge = CvBridge()

        # 创建图像订阅器
        # self.subscription = self.create_subscription(
        #     Image,
        #     '/pnp_sim/hand_image_raw',  # 确保与发布话题一致
        #     self.image_callback,
        #     10  # QoS队列深度
        # )
        self.subscription = self.create_subscription(
            ArmDataCollect,
            "/le_robot_data",  # 确保与发布话题一致
            self.image_callback,
            10,  # QoS队列深度
        )
        self.subscription  # 防止未使用警告

        # OpenCV窗口初始化
        cv2.namedWindow("Viewer", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Viewer", 640, 480)

        # 定时检查窗口状态
        self.timer = self.create_timer(0.1, self.check_window)

    def image_callback(self, msg):

        try:
            if hasattr(msg, "top_image"):
                cv_image1 = self.bridge.imgmsg_to_cv2(
                    msg.top_image,
                    desired_encoding="bgr8",
                )
                cv_image2 = self.bridge.imgmsg_to_cv2(
                    msg.right_image,
                    desired_encoding="bgr8",
                )
                # 两个图像合并
                cv_image = np.concatenate((cv_image1, cv_image2), axis=1)
            else:
                cv_image = self.bridge.imgmsg_to_cv2(
                    msg,
                    desired_encoding="bgr8",
                )
            # 3. 显示或处理图像
            cv2.imshow("Viewer", cv_image)
            cv2.waitKey(1)

        except Exception as e:
            self.get_logger().error(f"图像处理失败: {str(e)}")

    def check_window(self):
        # 检查窗口是否被关闭
        if cv2.getWindowProperty("Viewer", cv2.WND_PROP_VISIBLE) < 1:
            self.get_logger().info("检测到窗口关闭，退出节点...")
            self.destroy_node()

    def destroy_node(self):
        cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = PandaImageViewer()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
