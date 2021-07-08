# Object Detection and Tracking using YOLOv3
A python application to detect, track and count vehicles in video footage using YOLOv3 architecture and OpenCV.
YOLOv3 network uses an open-source neural network framework called Darknet, which is integrated with OpenCV. 
Pre-trained weights from the COCO dataset are used to identify 80 classes. A count of each vehicle class, the class label, 
and a confidence score are overlayed on the final prediction video. Additionally, it is made sure that the same vehicle is not detected 
again by checking the history of the mid-point of the bounding boxes. 
## Directory Structure

```
VehicleCounter/
.
├── coco.names
├── configs.yaml
├── mean.py
├── README.md
├── yolov3.cfg
├── yolov3.weights
└── videos
```
## Steps
1. Add an input video to the ```videos``` directory and make appropriate changes in the ```configs.yaml``` file.
2. Download the YOLOv3-416 weights,      
    ```wget https://pjreddie.com/media/files/yolov3.weights ```
3. Make sure all the modules and libraries imported in the ```main.py``` python file are installed. If not, a conda environment can be created using the ```requirements.txt``` file.
4. Finally, run the ```main.py``` file and wait for it to process the input video and save the predcition video. 
   
## Refrences
1. [YOLOv3: An Incremental Improvement.](https://arxiv.org/abs/1804.02767)
2. [YOLO: Real-Time Object Detection.](https://pjreddie.com/darknet/yolo/)
3. [PyImageSearch: YOLO object detection with OpenCV.](https://www.pyimagesearch.com/2018/11/12/yolo-object-detection-with-opencv/)
