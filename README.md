# Cricket-Object-Detection

* [Introduction](#introduction)
* [Tensorflow](#tensorflow)
* [Dependencies](#dependencies)
* [Cricket Ball Tracking](#cricket-ball-tracking)
* [YOLOv8](#yolov8-object-detection)


https://github.com/user-attachments/assets/ebd49b40-f3a7-4660-a152-fc7595d98ff2


## Introduction

With a growing interest in Data Science I chose to explore Object Detection using Python packages.

My initial learning was aided by the brilliant tutorial found at:

https://www.youtube.com/watch?v=yqkISICHH-U&t=14254s&pp=ygUbdGVuc29yZmxvdyBvYmplY3QgZGV0ZWN0aW9u

## Tensorflow

Tensorflow Model Creation Steps:

- Image Collection
- Image Annotation (Label Img package)
- Partitioning (Test and Train data sets)
- Tensorflow Model Zoo selection
- TF records generation
- Object Detection
- Exporting Model

![aa](https://blogger.googleusercontent.com/img/b/R29vZ2xl/AVvXsEhKis9ECId8eIwn_p0SVMBt3a1vfvKOcOZXy6zK0fWoyzXnzQTguKc2CV__6oI1Pwg22NjWsErpDKqjwQdzjilvmqwWkXPj2ncglphh6mAhpoZ_QXQiDwxnwo-GjKEP0fEOb3uBlNlh9sc/s1600/tensorflow2objectdetection.png)

## Dependencies

When going to initiate developing my own custom models I had some blocking dependency issues

I developed inside of a Conda (Miniconda) environment with key package installations for running Tensorflow including:

- Protobuf
- Tensorflow
- Tensorflow MacOS
- Tensorflow Metal
- Tensorflow Deps

I was sadly unable to sort a never ending string of dependency issues, despite employing many work arounds and made an executive decision to try a new framework

![aa](https://i.sstatic.net/lTWqp.png)

## Cricket Ball Tracking


Hawk-Eye in cricket is a cutting-edge computer vision system used to track the ball's trajectory and assist with umpiring decisions. It employs 6-10 high-speed cameras strategically positioned around the ground to capture the ball's movement in real-time. These cameras are synchronized to provide multiple angles, allowing the system to accurately detect and track the ball using color filtering, shape detection, and motion detection algorithms.

The system reconstructs the ball's 3D trajectory through triangulation, applying physics-based modeling to predict its future path, such as where it would have hit the stumps in LBW decisions. Hawk-Eye processes this data in real-time, providing immediate feedback for decisions like ball tracking, LBW calls, and no-ball detection.

Visualizations of the ball’s path are displayed for umpires, players, and spectators, and Hawk-Eye is integral to the Decision Review System (DRS), where players can challenge on-field calls. The system is highly accurate, typically within a few millimeters, and is calibrated before each match to ensure precision. This makes Hawk-Eye a vital tool in ensuring fair and accurate decision-making in cricket.

![aa](https://miro.medium.com/v2/resize:fit:1400/1*t5I7QjBlKTgXBFFjRghRRw.jpeg)


## YOLOv8 Object Detection

In the final part of this project I used the YOLOv8 package by Ultralytics to simplify setup for Object Detection

My goal was to emulate the ball tracking aspect of Hawkeye, creating a model to track a ball when given a MP4

Below is my file structure, with partitioned datasets; video and static image testing; as well as the main training code

![Alt text](Screenshot%202024-08-06%20at%2016.49.55.png)

The train.py file progressively builds upon the previous models best output whenever called, making decisions based on a specific .yaml file

![Alt text](Screenshot%202024-08-06%20at%2016.50.57.png)

The final model weights is shown in final.pt and the models output, running on a test video, is shown in the MP4 attached
