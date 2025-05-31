# Drone Waste Detector
_Solo University Project_

![](/mv6.png)

## Idea

The idea behind the project consists in a way to detect rubbish and litter located on the ground through UAV flights using deep learning-powered object detection and segmentation (with YOLONet) and stereo triangulation to locate the height of the objects. Using two cameras attached to the drone and facing downwards, it is possible to identify small and common elements of rubbish (such as abandoned cardboard pieces, plastic bottles, crumpled paper...) and identify their distance from the drone using stereo triangulation based on the YOLONet's bounding boxes instead of other physical sensors which may be heavy for the drone or might demonstrate a low synergy with YOLONet. Moreover, to simplify model training and testing without the usage of a real drone, an idea is presented to simulate the task in a virtual environment built with the Panda3D framework.

## Requirements

- Windows 10 or Linux
- Panda3D 1.10.14
- The Ultralytics Python Library (it should handle the pytorch installation on its own, but CUDA is necessary)

## Development

The development of the project was not an attempt to build a wide and comprehensive model for rubbish detection in all environments, conditions and soil types, but a test to verify whether this kind of task may be possible and if the usage of a virtual environment could aid its training to prevent the need to build complex datasets that require the physical presence of a drone in a real littered environment. Thus, it assumed the usage of a drone to detect littering in urban or partially-natural environments (streets, fields, gardens) and used a dataset and a 3D model that reflect these features.

The dataset used for training was [UAVVaste Dataset](https://github.com/PUTvision/UAVVaste), a dataset containing aerial views of countryside and urban environments with small elements of litter marked with bounding boxes and annotations. The dataset was then used to train a YOLOv8 model using the notebook [Yolo_Machine_Vision_c.ipynb](Yolo_Machine_Vision_c.ipynb).

As for the 3D environment and the development of the stereo triangulation, the [Panda3D](https://github.com/panda3d/panda3d) framework was used to simulate a virtual environment and two virtual cameras with coordinated movements were used to simulate the drone snapping photos during flight. Panda3D was chosen due to being based on python and for the good synergy with the **OpenCV** library, which is necessary for the rectification of the images and the math behind the triangulation. To avoid the complexity of creating or assembling a photorealistic environment, the 3D model used to simulate the littered environment was based on the photogrammetry of a small village near the river Garigliano, in Lazio, Italy, which was subsequently modified in the software Blender to manually place .png files of small elements of litter on the ground. The original photogrammetry has a Creative Commons license and is available at [this link](https://sketchfab.com/3d-models/rio-garigliano-e916af64dcab423ea4167c21a064da9e).

The process behind the detection and triangulation is the following:
- two Panda3D renders are exported from the slightly different views of the two cameras
- the images are eventually **rectified** to remove lens distortion using OpenCV
- the model trained with YOLOv8 is used to identify and draw the bounding boxes of possible elements of littering in both photos separately
- **template matching** is applied using the bounding boxes detected by YOLOv8 as sliding windows to find areas of overlapping texture in the images thanks to **normalized cross-correlation**
- after outliers removal, the median of distances between pixels inside the bounding box representing the same point in the virtual space is calculated as the distance of said object from the drone
- the real distance of the object from the camera, easily obtainable in a virtual simulation, is compared to the distance calculated through stereo triangulation

After installing the requirements and placing the rio_garigliano.glb file in the folder, the simulation can be tested with [drone_simulation_displayed_bboxes_v2.py](drone_simulation_displayed_bboxes_v2.py). More details on the algorithm used can be found in the comments of the file.

**Some important points about the stereo triangulation:**

The virtualized cameras are pinhole cameras, which means that they don't use lenses to focus light and are idealized cameras that cannot exist in the real world. A real life application of this technology would require **image rectification** to remove lens distortion and more complex calculations to achieve the intrinsic matrix of the two cameras such as a **camera calibration** process, as well as a much higher resolution if the baseline (the distance in meters between the two cameras) is not high enough. Some commented code is included in the file that includes the rectification process, but it's just a guideline as a real implementation of the project with physical cameras would require more steps anyway to obtain the internal parameters of the cameras.

## Results

The training with YOLOv8 did not perform as well as expected, maybe due to the limitations of training on images of size 1920x1080, as the elements of litter are quite small and YOLOv8 might struggle with identifying elements that are just as big as a handful of pixels. The results can definitively be improved with some ablation and with higher resolutions or with zooming/dividing the images in smaller quadrants where to perform the object recognition.

As for the stereo triangulation, the results are positive and quite precise, although the precision decreases with altitude as the distance in pixels between the points becomes smaller (with a resolution of 1920x1080, it starts to struggle at about 70m of height). A chart depicting the comparison between ground truth and stereo triangulation can be seen here:

![The orange line represents ground truth while the blue line shows the height calculated through stereo triangulation](/mv4.png)

Eventual developments and improvements on this project are not planned for now, as this was a quick university project to explore the possibility of using simulated environments in object recognition/segmentation to cut costs. The justification behind the implementation of stereo triangulation for distance calculation of the detected objects is that it's a sleeker way to identify different distances of different items in the same view without using sensors and it can be a decently precise way to place the object in a virtual map for another external agent to pick up or, in a more ambitious project, to lead the UAV to get closer to the object so that it may get picked up (this is unrealistic and would probably need a lot of sensors anyway). As for considerations on the application of such a technology, it could be used to aid janitors locate litter and leftovers in large closed areas where the drone can operate under supervision, such as an amusement park after closure, a field after an open-air concert or a fair. There is a lot that can be discussed about this project, such as the problem of variable/poor lighting in the environment, the foggy definition of what constitutes rubbish or littering (are papers lying on the ground always rubbish?), the difficulty in avoiding false positives, the computational power required for real-time GPU processing on a compact UAV, the problem of drone navigation in complex environments... This discussion is beyond the scope of this simple project and is left to future research, as there is a lot of interest in AI technology for waste management and this is a very small contribute to the field.
