This project runs real-time object detection using a TensorRT-optimized model and an ASUS Xtion RGB camera.

The script loads a TensorRT `.engine` file, initializes the Xtion camera through OpenNI2, captures live RGB frames, preprocesses them, and performs GPU-accelerated inference using CUDA and PyCUDA. The model output is decoded into bounding boxes, confidence scores, and class IDs. It supports both single-class and multi-class detection models.

Detected objects are filtered by confidence threshold, processed with Non-Maximum Suppression to remove overlapping detections, and drawn on the live video feed with class names, confidence values, FPS, and detection count. Class labels are loaded from a `classes.txt` file, with one class name per line.

The application is designed for real-time deployment on NVIDIA platforms, such as Jetson devices, where TensorRT can be used to maximize inference performance.
