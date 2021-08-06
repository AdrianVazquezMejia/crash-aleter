# Description

We are international team (Adrian Vazquez Mejia & Ireneusz Cierpisz). Our project named Artificial Assistant Dog was made as part of the OpenCV AI Competition 2021, sponsored by Microsoft Azure and Intel. #OAK2021 competition is focused on solutions solving real world problems using spatial AI. We use the OpenCV AI Kit D (OAK-D) to solve our challenge area. OAK is a tiny low-power hardware edge AI computing module based on Intel Movidius Myriad-X embedded AI chip. Our project is a program to alert people in danger to be hit by a car. We use computer vision capabilities provided by the OAK-D camera, OpenCV and Luxonis DepthAI libraries. 
People are thoughtful, listening to music through headphones, looking at the smartphone screen, etc. We think AI could recognize the likelihood of a collision and warn a pedestrian to stop, even if they are sometimes not aware that they are on the verge of an accident or death.

In the project we tried to solve a few problems:
   - detecting people and vehicles,
   - collecting adjusted and selected tracking data, 
   - compute a direction of movement of objects, 
   - compute an intersection points for trajectories of many objects in three dimensional space,
   - calculating the speed of objects and determining whether there is a possibility of a collision for each pedestrian-vehicle pair,
   - alarm activation for one to two seconds before the collision,
   - building a system independent of the Internet or computer network, which could be mounted anywhere,
      - setup the IoT application on a Raspberry Pi with OAK-D, 
      - build a system connecting a microcomputer with transmitters and with a device emitting a warning signal. 

Models:
   For pedestrian and vehicle detection, we tested the use of models such as the Intel OpenVINO IR network person-vehicle-bike-intersection-1016 based on MobileNetV2 + SSD, pedestrian and vehicle detection network-adas-0001 based on MobileNet v1.0 + SSD, as well as the network mobilenet-ssd the same one that Luxonis uses in many tutorials. The latter network turned out to be the most useful for our purposes and we obtained the presented results with its use.
   

## Setup
After cloning the this repository, go to the project folder and the follow

```
cd crash-aleter
```
```
python3 -m venv venv
```
```
source venv/bin/activate
```
```
python3 pip install -r requierements.txt
```
```
python3 python3 install_requierements.py
```

Make sure you place your model into the __models__ folder.

Then you can run the app, by

```
python3 src/main.py
```
