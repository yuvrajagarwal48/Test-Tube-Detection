#!/usr/bin/env python3
import cv2 as cv
import math
import cvzone
import rospy
from std_msgs.msg import Bool
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from ultralytics import YOLO

# Initialize ROS node
rospy.init_node('test_tube_detector')

# Initialize YOLOv5 model
model = YOLO("best.pt")

# Create ROS publisher for test tube detection flag
detection_publisher = rospy.Publisher('test_tube_detected', Bool, queue_size=10)

# Create a CvBridge to convert ROS images to OpenCV images
bridge = CvBridge()

className = "test-tube"

# Callback function for processing images from the webcam
def image_callback(msg):
    try:
        # Convert ROS image to OpenCV image
        frame = bridge.imgmsg_to_cv2(msg, "bgr8")

        # Perform inference on the frame
        results = model(frame)

        # test_tube_detected = False  # Flag to check if test tube is detected


        for r in results:
            test_tube_detected = False 
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                w, h = x2 - x1, y2 - y1
                # cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                confidence = math.ceil((box.conf[0] * 100)) / 100
                
                # Display distance in the text
                text = f'{className} {confidence}'
                # cvzone.putTextRect(frame, text, (max(0, x1), max(35, y1)), scale=1, thickness=2)
                if confidence > 0.5:
                    cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cvzone.putTextRect(frame, text, (max(0, x1), max(35, y1)), scale=1, thickness=2)
                    test_tube_detected = True    
                detection_publisher.publish(test_tube_detected)

        # Display the frame with bounding box predictions
        cv.imshow("Video", frame)
        cv.waitKey(1)

    except Exception as e:
        rospy.logerr(e)

if __name__ == "__main__":
    # Create OpenCV video capture object
    cap = cv.VideoCapture(0)

    # Ensure camera is opened successfully
    if not cap.isOpened():
        rospy.logerr("Failed to open webcam.")
        exit()

    # Loop to process webcam frames
    while not rospy.is_shutdown():
        ret, frame = cap.read()  # Read frame from webcam

        if ret:
            try:
                image_message = bridge.cv2_to_imgmsg(frame, "bgr8")  # Convert OpenCV image to ROS image message
                image_callback(image_message)  # Call image callback function
            except Exception as e:
                rospy.logerr(e)
        else:
            rospy.logerr("Failed to read frame from webcam.")
            break

    # Release resources
    cap.release()
    cv.destroyAllWindows()
