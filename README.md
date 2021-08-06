# Description

Our project was made as part of the OpenCV AI Competition 2021, sponsored by Microsoft Azure and Intel. #OAK2021 competition is focused on solutions solving real world problems using spatial AI. We use the OpenCV AI Kit D (OAK-D) to solve our challenge area. OAK is a tiny low-power hardware edge AI computing module based on Intel Movidius Myriad-X embedded AI chip. Our project is a program to alert people in danger to be hit by a car. Using computer vision capabilities provided by the OAK-D camera, OpenCV and Luxonis libraries. 

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
