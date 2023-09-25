# Human_Dog_Counter
Human-Dog Categorization using YOLO v3 with OpenCV
Overview:
This script uses the YOLOv3 (You Only Look Once) object detection model to identify and categorize humans ("person") and dogs in real-time video feed captured from the computer's webcam.

Key Components:
Loading YOLO:

The YOLO model is loaded using OpenCV with provided weights (yolov3.weights) and configuration (yolov3.cfg).
The coco.names file, containing class names, is loaded. This file provides the names of objects that YOLO can detect.
Webcam Capture:

OpenCV is used to capture live video feed from the computer's default camera.
Object Detection:

Each frame from the webcam feed is preprocessed and forwarded through the YOLO model.
Detections with a confidence level above 0.5 are considered, and bounding boxes are drawn around the detected objects.
Category Counting:

The script keeps a count of detected humans (persons) and dogs in each frame.
The count is displayed on the video feed.
Visualization:

Bounding boxes are drawn around detected humans and dogs. Each box is labeled with the class name ("person" or "dog").
The current count of detected humans and dogs is displayed at the top of the video feed.
Exit:

The video feed is displayed in a window that can be closed by pressing the 'q' key.
How It Works:
Once the script is executed, it initializes the YOLO model and starts capturing video feed from the webcam.
Each frame is analyzed for the presence of humans and dogs.
Identified humans and dogs are surrounded with bounding boxes, and their counts are displayed on the frame.
The processed frame is displayed in a window, which is 2 times larger than the original frame size.
Pressing the 'q' key exits the video feed and ends the program.
