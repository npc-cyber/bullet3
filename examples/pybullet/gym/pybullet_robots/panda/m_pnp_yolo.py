import rclpy
from rclpy.node import Node
import cv2
import os
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose
from ultralytics import YOLO


class YOLOv8Detector(Node):  # 继承自 Node 类
    def __init__(self):
        super().__init__("yolov8_detector")  # 初始化 ROS2 节点

        # 加载 YOLO 模型
        model_path = os.path.expanduser("/home/yangrui/trilib/pybullet/yolov8n.pt")
        self.model = YOLO(model_path)
        self.class_names = self.model.names
        print(len(self.class_names), self.class_names)

        # 图像转换工具
        self.bridge = CvBridge()

        # 创建发布/订阅
        self.image_sub = self.create_subscription(
            Image, "/pnp_sim/image_raw", self.image_callback, 10
        )
        self.det_pub = self.create_publisher(Detection2DArray, "/detections", 10)
        self.vis_pub = self.create_publisher(Image, "/detection_result", 10)

        self.get_logger().info("YOLOv8检测节点已启动")  # 修改日志记录方式
        self.img_count = 0
        self.have_find = False

    def get_detect(self, cv_image):

        results = self.model(source=cv_image, conf=0.6, imgsz=640, verbose=False)

        # 检查是否有有效检测结果
        if len(results[0].boxes) == 0:  # 如果没有检测到任何目标
            self.get_logger().info("未检测到目标")
            return  # 直接返回

        # 获取第一个检测结果的边界框
        boxes = results[0].boxes.xyxy.cpu().numpy()
        classes = results[0].boxes.cls.cpu().numpy()
        confidences = results[0].boxes.conf.cpu().numpy()

        # 绘制检测结果
        for box, cls, conf in zip(boxes, classes, confidences):
            # if self.class_names[int(cls)] != "apple":
            #     self.get_logger().debug("未检测到")
            #     return  # 直接返回
            # else:
            #     self.get_logger().info("检测到苹果")
            #     self.have_find = True
            x1, y1, x2, y2 = map(int, box)
            label = f"{self.class_names[int(cls)]} {conf:.2f}"
            cv2.rectangle(cv_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                cv_image,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

        cv2.imwrite(
            "/home/yangrui/trilib/pybullet/log/test" + str(self.img_count) + ".jpg",
            cv_image,
        )
        self.img_count += 1

    def image_callback(self, msg):
        try:
            # if self.have_find:
            #     return
            # 转换 ROS 图像消息为 OpenCV 格式
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.get_detect(cv_image)

        except Exception as e:
            self.get_logger().error(f"图像处理失败: {str(e)}")  # 修改错误日志记录方式


def main(args=None):
    rclpy.init(args=args)
    detector = YOLOv8Detector()
    try:
        rclpy.spin(detector)  # ROS2 的 spin 方式
    except KeyboardInterrupt:
        pass
    finally:
        detector.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
