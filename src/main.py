from pathlib import Path
import sys
import cv2
import depthai as dai
import numpy as np
import time
import matplotlib.pyplot as plt
import os

import logging

logging.getLogger().setLevel(logging.INFO)

#Global variables
labelMap = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
                "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
syncNN = True

nnBlobPath = str((Path(__file__).parent / Path('../models/mobilenet.blob')).resolve().absolute())

not_street = False   # 'False' for real life distances, I mean street. 'True' for experiments with toys.


# check the list of objects to see if there's an object that has come out of a frame for more than 2sec and delete it
def clean_redundant_data(objects, current_time):
    l = [] # list of indices of cars which has come out of a frame and should be deleted from cars tracking list
    for i in range(len(objects)):
        if current_time - objects[i][1] > 2:
            l.append(i)
    objects = [obj for idx,obj in enumerate(objects) if idx not in l]

    return objects


def write_id_in_frame(objects):
    for obj in objects:
        id, point = obj[0], (obj[-1][0], obj[-1][1] - 10)
        cv2.putText(frame, f"{id}", point, cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0,0,255))
    return 

  
# check if the deviations in series of coordinates, needed to calculate the direction of object movement, are below the assumed threshold
# reject a random error that occurs in depth coords X,Y,Z
def check_deviation_of_depth_coords(obj, xc, yc, X, Y, Z):   # this function should be called if len(obj)>6
    #compute an average deviation of object coords for the last three positions and multiply by 2, then calculate delta
    dx = (abs(obj[4][0] - obj[5][0]) + abs(obj[5][0] - obj[6][0]))//2 * 2
    if dx <= 5: 
        dx = 5
    dy = (abs(obj[4][1] - obj[5][1]) + abs(obj[5][1] - obj[6][1]))//2 * 2
    if dy <= 5: 
        dy = 5
    dX = (abs(obj[4][2] - obj[5][2]) + abs(obj[5][2] - obj[6][2]))/2 * 2
    if dX < abs(0.1 * obj[6][2]): 
        dX = obj[6][2] * 0.2
    dY = (abs(obj[4][3] - obj[5][3]) + abs(obj[5][3] - obj[6][3]))/2 * 2
    if dY < abs(0.1 * obj[6][3]): 
        dY = obj[6][3] * 0.2
    dZ = (abs(obj[4][4] - obj[5][4]) + abs(obj[5][4] - obj[6][4]))/2 * 2
    if dZ < abs(0.1 * obj[6][4]): 
        dZ = obj[6][4] * 0.1

    # if xc, yc is within the mean deviation and value of X or Y or Z is very different then the last one, ignore this depth value 
    # and assigne a calculated value which differ from the mean no more than 0.01
    if abs(obj[-1][0] - xc) <= dx and abs(obj[-1][1] - yc) <= dy and abs(obj[-1][2] - X) > (dX * 2):
        X = ((obj[4][2] + obj[5][2] + obj[6][2]) / 3) * 1.01
    if abs(obj[-1][0] - xc) <= dx and abs(obj[-1][1] - yc) <= dy and abs(obj[-1][3] - Y) > (dY * 2):
        Y = ((obj[4][3] + obj[5][3] + obj[6][3]) / 3) * 1.01
    if abs(obj[-1][0] - xc) <= dx and abs(obj[-1][1] - yc) <= dy and abs(obj[-1][4] - Z) > dZ and (obj[-1][4] - Z) < 0:
        Z = ((obj[4][4] + obj[5][4] + obj[6][4]) / 3) * 1.01
    if abs(obj[-1][0] - xc) <= dx and abs(obj[-1][1] - yc) <= dy and abs(obj[-1][4] - Z) > dZ and (obj[-1][4] - Z) > 0:
        Z = ((obj[4][4] + obj[5][4] + obj[6][4]) / 3) * 0.99
        
        
    return X,Y,Z

  

def print_data_of_detected(obj_list, obj_name):
    oppos_id = "person"
    if obj_name == "person":
        oppos_id = "car"
    if obj_list:
        for p in obj_list:
            print(f'  <{obj_name}_{p[0]}>  3Dpos1:{p[2][0]}, 3Dlast_pos:{p[2][1]}')                                # t:{p[1]}, dir_vect:{p[2][2]}')
            if p[3]:
                for e in p[3]:
                    print(f'       intersect_coords: {e[0]},  intersect with {oppos_id}_{e[1]}')                        #, detect_time:{e[3]:.3f}')
                    print(f'       distance2collision_point: {e[2]:.3f},  ({obj_name}_speed, time2crash): {e[4]}')      #, last_pos:{e[5]}')
            if len(p) == 7: print(f'       all_coords>  pos1:{p[4]},  pos2:{p[5]},  last_pos:{p[6]}')
            if len(p) == 6: print(f'       all_coords>  pos1:{p[4]},  pos2:{p[5]}')
            if len(p) == 5: print(f'       all_coords>  pos1:{p[4]}')
    else:
        print(f'No {obj_name} in the designated field.')

        
        
def replace_insert_crashdata(collide_list, new_intersect):
    alarm_flag = False
    if collide_list:
        i = 0
        not_found = True
        while not_found:
            crash_event = collide_list[i]          # [np.array([xi, yi, zi]), c[0], p_distance, p_time, (speed, time_to_collision), p_last_position]
            if new_intersect[1] == crash_event[1]:  # obj id
                crash_event[0] = new_intersect[0]   # intersection
                dd = np.sqrt(sum(e**2 for e in (new_intersect[5] - crash_event[5])))
                dt = new_intersect[3] - crash_event[3]   # sec
                if dt > 0:
                    speed = dd/dt   # m/s
                else: speed = 0.0
                crash_event[2] = new_intersect[2]   # distance to collision
                if speed > 0:
                    time_to_collision = crash_event[2] / speed
                else: time_to_collision = 0.0
                crash_event[3] = new_intersect[3]   # time of detection the last position
                crash_event[4] = (speed, time_to_collision)
                if time_to_collision < 1 and time_to_collision > 0:   # if time to collision < 1sec
                    alarm_flag = True
                not_found = False
                continue
            elif i < (len(collide_list) - 1):
                i += 1
            else:
                collide_list.append(new_intersect)
    else:
        collide_list.append(new_intersect)

        
    return alarm_flag, collide_list

        
    
        
'''
Spatial detection network demo.
    Performs inference on RGB camera and retrieves spatial location coordinates: x,y,z relative to the center of depth map.
'''
if __name__=="__main__":

    if len(sys.argv) > 1:
        nnBlobPath = sys.argv[1]
    
    # Start defining a pipeline
    pipeline = dai.Pipeline()
    
    # Define a source - color camera
    colorCam = pipeline.createColorCamera()
    spatialDetectionNetwork = pipeline.createMobileNetSpatialDetectionNetwork()
    monoLeft = pipeline.createMonoCamera()
    monoRight = pipeline.createMonoCamera()
    stereo = pipeline.createStereoDepth()
    
    xoutRgb = pipeline.createXLinkOut()
    xoutNN = pipeline.createXLinkOut()
    
    xoutBoundingBoxDepthMapping = pipeline.createXLinkOut()
    xoutDepth = pipeline.createXLinkOut()
    
    xoutRgb.setStreamName("rgb")
    xoutNN.setStreamName("detections")
    xoutBoundingBoxDepthMapping.setStreamName("boundingBoxDepthMapping")
    xoutDepth.setStreamName("depth")
    
    #colorCam.setPreviewSize(672, 384) # preview output resized to fit the pedestrian-and-vehicle-detector-adas-0001.blob
    #colorCam.setPreviewSize(512, 512) # preview output resized to fit the person-vehicle-bike-detection-crossroad-1016.blob
    colorCam.setPreviewSize(300, 300)
    colorCam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    colorCam.setInterleaved(False)
    colorCam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
    
    monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
    monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)
    
    # setting node configs
    stereo.setOutputDepth(True)
    stereo.setConfidenceThreshold(240) #255#600  set the confidence for disparity; set it to higher will cause less "holes"(values 0) but disparities/depths won't be as accurate
    
    spatialDetectionNetwork.setBlobPath(nnBlobPath)
    spatialDetectionNetwork.setConfidenceThreshold(0.5)
    spatialDetectionNetwork.input.setBlocking(False)
    spatialDetectionNetwork.setBoundingBoxScaleFactor(0.5)
    spatialDetectionNetwork.setDepthLowerThreshold(100)
    spatialDetectionNetwork.setDepthUpperThreshold(45000) #(5000)
    
    # Create outputs
    
    monoLeft.out.link(stereo.left)
    monoRight.out.link(stereo.right)
    
    colorCam.preview.link(spatialDetectionNetwork.input)
    if(syncNN):
        spatialDetectionNetwork.passthrough.link(xoutRgb.input)
    else:
        colorCam.preview.link(xoutRgb.input)
    
    spatialDetectionNetwork.out.link(xoutNN.input)
    spatialDetectionNetwork.boundingBoxMapping.link(xoutBoundingBoxDepthMapping.input)
    
    stereo.depth.link(spatialDetectionNetwork.inputDepth)
    spatialDetectionNetwork.passthroughDepth.link(xoutDepth.input)
    
    
    # Pipeline defined, now the device is connected to
    with dai.Device(pipeline) as device:
        # Start pipeline
        device.startPipeline()
    
        # Output queues will be used to get the rgb frames and nn data from the outputs defined above
        previewQueue = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
        detectionNNQueue = device.getOutputQueue(name="detections", maxSize=4, blocking=False)
        xoutBoundingBoxDepthMapping = device.getOutputQueue(name="boundingBoxDepthMapping", maxSize=4, blocking=False)
        depthQueue = device.getOutputQueue(name="depth", maxSize=4, blocking=False)
        
        frame = None
        detections = []
    
        startTime = time.monotonic()
        counter = 0
        fps = 0
        color = (255, 255, 255)
    
        # create lists to collect persons and vehicles last position (X,Y coordinates)
        cars, persons = [], [] 
        car_id = 0
        person_id = 0
        count = 0   # frame number
    
        #create lists for scatterplot data
        plotf, plotx, ploty, plotX, plotY = [], [], [], [], [] 
    
        while True:
            
            inPreview = previewQueue.get()
            inNN = detectionNNQueue.get()
            depth = depthQueue.get()

            counter+=1
            current_time = time.monotonic()
            if (current_time - startTime) > 1 :
                fps = counter / (current_time - startTime)
                counter = 0
                startTime = current_time
    
            frame = inPreview.getCvFrame()
            #depthFrame = depth.getFrame()
    
            #depthFrameColor = cv2.normalize(depthFrame, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
            #depthFrameColor = cv2.equalizeHist(depthFrameColor)
            #depthFrameColor = cv2.applyColorMap(depthFrameColor, cv2.COLORMAP_HOT)
            detections = inNN.detections
            count += 1
            i = 0   # bb number in a frame
            detections_list = []
    
            # if the frame is available, draw bounding boxes on it and show the frame
            height = frame.shape[0]
            width  = frame.shape[1]
            for detection in detections:
                # denormalize bounding box
                x1 = int(detection.xmin * width)
                x2 = int(detection.xmax * width)
                y1 = int(detection.ymin * height)
                y2 = int(detection.ymax * height)
                #gets the center point of a bounding box
                xc, yc = (x2+x1)//2, (y2+y1)//2
    
                # get value of spatial coords in meters
                X = int(detection.spatialCoordinates.x) / 1000
                Y = int(detection.spatialCoordinates.y) / 1000
                Z = int(detection.spatialCoordinates.z) / 1000
    
                try:
                    label = labelMap[detection.label]
                except:
                    label = detection.label

                
                #fill out the checking list
                detections_list.append((i, label, xc, yc, X, Y, Z))
                i += 1
                print(f'\nF>{count}, ct>{current_time}, Detected : {detections_list}')
                
                # Draw data in the frame
                if label == "person" or label == "car":
                    cv2.putText(frame, str(label), (x1 + 10, y1 + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
                    cv2.putText(frame, "{:.2f}".format(detection.confidence*100), (x1 + 10, y1 + 35), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
                    cv2.putText(frame, f"X: {X} m", (x1 + 10, y1 + 50), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
                    cv2.putText(frame, f"Y: {Y} m", (x1 + 10, y1 + 65), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
                    cv2.putText(frame, f"Z: {Z} m", (x1 + 10, y1 + 80), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, cv2.FONT_HERSHEY_SIMPLEX)
                    cv2.circle(frame, (xc, yc), 5, (0,0,255), -1)

    
    
    #---start tracking---------------- should be here def of tracking objects; collecting data for further computations of movement direction and collision point
                # updates person_id and localization in the frame
                if persons and (label == "person"):  # if list of persons is not empty and it's a person
                    j = 0  #index of person in persons
                    not_found = True
                    # find if an object exist in the list, try until is not find
                    while not_found:   
                        p = persons[j]  #predecessor data
                        if len(p) > 6:
                            X, Y, Z = check_deviation_of_depth_coords(p, xc, yc, X, Y, Z)
                        if (abs(p[-1][0]-xc) < 50) and (abs(p[-1][1]-yc) < 50) and (abs(p[-1][2]-X) < 0.500) and (abs(p[-1][3]-Y) < 0.500) and (abs(p[-1][4]-Z) < 1.000):
                            p_time = time.monotonic()
                            # if it is not a "hole" value (depth measurement error), add new coordinates of an object
                            if X != 0 or Y != 0:
                                p[1] = p_time
                                p.append((xc, yc, X, Y, Z))    # 
                            if len(p) > 7:  # leave only the last three positions of the person needed to calculate the direction of movement
                                del p[4]
                            not_found = False
                            continue
                        elif j < len(persons)-1:   # try to get next object from the list
                            j += 1
                        else:            # append a new object
                            # if it is not a "hole" value (depth measurement error), add a new object
                            if not_street and (X != 0 or Y != 0) and Z != 0:    # for very close distances
                                p_time = time.monotonic()
                                person_id += 1
                                persons.append([person_id, p_time, ([0,0,0],[0,0,0],[0,0,0]), [], (xc, yc, X, Y, Z)])   # append coordinates to the list as a new object position 
                                not_found = False
                            elif Z > 5:        # for real life
                                p_time = time.monotonic()
                                person_id += 1
                                persons.append([person_id, p_time, ([0,0,0],[0,0,0],[0,0,0]), [], (xc, yc, X, Y, Z)])   # append coordinates to the list as a new object position 
                                not_found = False
                elif label == "person":     # append the first object
                    p_time = time.monotonic()
                    # append obj id, last possition detection time, extrapolation line parameters(p0,pn,v), intersection point coords and obj id-s, spatial position
                    if not_street and (X != 0 or Y != 0) and Z != 0:
                        persons.append([person_id, p_time, ([0,0,0],[0,0,0],[0,0,0]), [], (xc, yc, X, Y, Z)])
                    elif Z > 5: # if distance from the cam is bigger then 5m
                        persons.append([person_id, p_time, ([0,0,0],[0,0,0],[0,0,0]), [], (xc, yc, X, Y, Z)])
                
                persons = clean_redundant_data(persons, current_time)
                
                write_id_in_frame(persons)
                
                
                # updates car_id and time and its last position in the frame
                if cars and (label == "car"):  # if list of cars is not empty and it's a car
                    j = 0  #index of car in cars
                    not_found = True
                    # find if an object exist in the list, try until is not find
                    while not_found:   
                        p = cars[j]  #predecessor data
                        if len(p) > 6:
                            X, Y, Z = check_deviation_of_depth_coords(p, xc, yc, X, Y, Z)
                        if (abs(p[-1][0]-xc) < 50) and (abs(p[-1][1]-yc) < 50) and (abs(p[-1][2]-X) < 0.500) and (abs(p[-1][3]-Y) < 0.500) and (abs(p[-1][4]-Z) < 1.000):
                            p_time = time.monotonic()
                            # if it is not a "hole" value (depth measurement error), add new coordinates of an object
                            if X != 0 or Y != 0:
                                p[1] = p_time
                                p.append((xc, yc, X, Y, Z))    # 
                            if len(p) > 7:  # leave only the last three positions of the car needed to calculate the direction of movement
                                del p[4]
                            not_found = False
                            continue
                        elif j < len(cars)-1:   # try to take next object from the list
                            j += 1
                        else:            # append a new object
                            # if it is not a "hole" value (depth measurement error), add a new object
                            if not_street and (X != 0 or Y != 0) and Z != 0:
                                p_time = time.monotonic()
                                car_id += 1
                                cars.append([car_id, p_time, ([0,0,0],[0,0,0],[0,0,0]), [], (xc, yc, X, Y, Z)])   # append coordinates to the list as a new object position 
                                not_found = False
                            elif Z > 5:
                                p_time = time.monotonic()
                                car_id += 1
                                cars.append([car_id, p_time, ([0,0,0],[0,0,0],[0,0,0]), [], (xc, yc, X, Y, Z)])   # append coordinates to the list as a new object position 
                                not_found = False
                elif label == "car":     # append the first object
                    p_time = time.monotonic()
                    if not_street and (X != 0 or Y != 0) and Z != 0:
                        cars.append([car_id, p_time, ([0,0,0],[0,0,0],[0,0,0]), [], (xc, yc, X, Y, Z)])
                    elif Z > 5: # distance from the cam is bigger then 5m
                        cars.append([car_id, p_time, ([0,0,0],[0,0,0],[0,0,0]), [], (xc, yc, X, Y, Z)])
    
                cars = clean_redundant_data(cars, current_time)  
      
                write_id_in_frame(cars)
      
                
                # print selected data from trackers
                print('IN FRAME:')
                print_data_of_detected(persons, 'person')
                print_data_of_detected(cars, 'car')
   
                 
                # COMPUTE AN OBJECT MOVEMENT DIRECTION LINE
                # add to the data array first and last point of object position and a line direction vector
                for car in cars:
                    if len(car) > 5:   # if there are in car at least two tuples with coords
                        # get the first and the last spacial position of a car
                        (X0,Y0,Z0), (X2,Y2,Z2) = car[4][2:], car[-1][2:]  
                        cp0 = np.array([X0,Y0,Z0])
                        v = np.array([X2 - X0, Y2 - Y0, Z2 - Z0])  #direction vector
                        cp = cp0 + v  
                        car[2] = (cp0, cp, v)   # append two points on the line of a car movement and the direction vector
                for person in persons:
                    if len(person) > 5:  
                        # get the first and the last spacial position of a person
                        (X0,Y0,Z0), (X2,Y2,Z2) = person[4][2:], person[-1][2:]  
                        pp0 = np.array([X0,Y0,Z0])  # point zero == first from the last three person's positions
                        v = np.array([X2 - X0, Y2 - Y0, Z2 - Z0])
                        pp = pp0 + v 
                        person[2] = (pp0, pp, v)  
                
                # COMPUTE AN INTERSECTION/CRASH POINT IN GIVEN FRAME. 
                # calculations for each pair a car-person

                if len(cars) != 0 and len(persons) != 0:   # if there is a car and a person detected
                    print('Searching for Pedestrian-Car Crossing Point...')
                    for c in cars:
                        if len(c) > 6:

                            #print('car id: ', c[0])

                            for p in persons:
                                if len(p) > 6:
                                    p[3] = delete_unnecessary_crash_points(p[3], cars)  # update crash points list

                                    #print('person id: ', p[0])

                                    ## if the set of coefficients of the direction vectors of two lines, car and person routs, are proportional these lines are parallel to each other
                                    #and their direction vectors Cross Product equals 0
                                    if any(np.cross(c[2][2], p[2][2])):         
                                        # get a point on the car line and its direction vector coefficients 
                                        x_1, a_1, y_1, b_1, z_1, c_1 = c[2][0][0], c[2][2][0], c[2][0][1], c[2][2][1], c[2][0][2], c[2][2][2]

                #                        print('car position, y_1: {},  z_1: {}'.format(c[2][0][1], c[2][0][2]))
                #                        print('car vector, a_1: {:.3f}, b_1: {:.3f},  c_1: {:.3f}'.format(c[2][2][0], c[2][2][1], c[2][2][2]))

                                        # car line equation: x=a_1*t+x_1, y=b_1*t+y_1, z=c_1*t+z_1
                                        # a normal vector n to a person's plane it is a np.cross product of two vectors (pp0->pp1, pp0->pp2) 
                                        # where pp0, pp1, pp2 are three points in the plane, and pp2 differs from pp0 only by the value of the y coordinate increased by 10
                                        # vector pp0->pp1 == direction vector == p[2][2]
                                        # vector pp0->pp2 == [p[2][0][0] - p[2][0][0], p[2][0][1] + 10 - p[2][0][1], p[2][0][2] - p[2][0][2]] == [0, 10, 0]
                                        # Get a normal vector of a person's plane
                                        n = np.cross(p[2][2], np.array([0, 10, 0]))
                                        nx, ny, nz = n[0], n[1], n[2]

                #                        print('normal vector, nx: {:.3f}, ny: {:.3f}, nz: {:.3f}'.format(nx, ny, nz))

                                        # equation of the plane nx*(x-x0) + ny*(y-y0) + nz*(z-z0) == 0  => 
                                        # get a point on the plane
                                        x0, y0, z0 = p[2][0][0], p[2][0][1], p[2][0][2] #coords of pp0
                                        d = -(nx*x0 + ny*y0 + nz*z0)
                                        # as nx*x + ny*y + nz*z + d = 0
                                        # so nx*(a_1*t + x_1) + n_y*(b_1*t + y_1) + nz*(c_1*t + z_1) + d == 0
                                        if (nx*a_1 + ny*b_1 + nz*c_1) != 0:
                                            t = -(d + nx*x_1 + ny*y_1 + nz*z_1) / (nx*a_1 + ny*b_1 + nz*c_1)

                #                            print(f'coefficient t: {t}')

                                            # intersection point
                                            xi = (a_1*t + x_1)
                                            yi = (b_1*t + y_1)
                                            zi = (c_1*t + z_1)

                                            #print('  Intersection x,y,z: ', xi,yi,zi)

                                            # distance to the crash point:
                                            p_distance = np.sqrt(sum(e**2 for e in (np.array([xi, yi, zi]) - p[2][1])))  # pedestrian
                                            c_distance = np.sqrt(sum(e**2 for e in (np.array([xi, yi, zi]) - c[2][1])))  # car

                                            #print(f'  Person distance to the collision point: {p_distance:.3f}m')

                                            p_last_pos = p[2][1]  # last position of a pedestrian                                        
                                            p_time = p[1]         # the time of detecting the last position of a pedestrian
                                            # insert intersection coords, an id of a car the person can collide with, distance to the hypothetical crash point etc.:
                                            alarm_flag_p, p[3] = replace_insert_crashdata(p[3], [np.array([xi, yi, zi]), c[0], p_distance, p_time, (0, 0), p_last_pos])

                                            print(f'    Person{p[0]} Alarm Flag: {alarm_flag_p}')

                                            #print(f'  Car distance to the collision point: {c_distance:.3f}m')

                                            # insert intersection coords, an id of a person the car can collide with, distance to the hypothetical crash point etc.:
                                            alarm_flag_c, c[3] = replace_insert_crashdata(c[3], [np.array([xi, yi, zi]), p[0], c_distance, c[1], (0, 0), c[2][1]])

                                            print(f'    Car{c[0]} Alarm Flag: {alarm_flag_c}')

                                            ## ALARM
                                            if alarm_flag_p and alarm_flag_c:
                                                print(f'ALARM!!! ALARM!!! Person{p[0]} is going to collide with Car{c[0]}')    # send_alarm_message_to_device()

                            #print('END OF THE LOOP OF SEARCHING FOR THE CROSSING POINT OF A CAR WITH A PEDESTRIAN PLANE ')
                            # draw in the frame a line connecting each pair of a person and a car for which time_to_collision is computed
                            if p[3]: 
                                for crash in p[3]:
                                    if crash[4][1] != 0:  # time to collision
                                        carid = crash[1]  # a car involved in
                                        for v in cars:    # v -- vehicle
                                            if v[0] == carid:
                                                car_2d_pos = (v[6][0],v[6][1])     # location in frame
                                                person_2d_pos = (p[6][0],p[6][1])
                                                #TODO: pobrac wspol. przestrzenne do wykresu
                                                cv2.line(frame, person_2d_pos, car_2d_pos, (255,0,0), 1)
    
    #---end tracking-------------------
    
            print('\nCars: F', count, cars)
            print('\nPersons: F', count, persons)
    
    
            cv2.putText(frame, "NN fps: {:.2f}".format(fps), (2, frame.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.4, color)
            #cv2.imshow("depth", depthFrameColor)
            cv2.imshow("rgb", frame)
    
            ##create scatterplot of y and Y data with gridlines
            #plt.scatter(plotf, ploty, s=1)
            #plt.scatter(plotf, plotY, s=1)
            #plt.minorticks_on()
            #plt.grid(which='minor')
            #plt.grid(which='major')
            #plt.xlabel("Frames")
            #plt.ylabel("The value of the yc_bb(blue) and Y_depth(orange)")
            #plt.title("Discontinuities in the occurrence of the Y, depth coordinates while bb is detected in the frame")
            #if 500 < count < 502: plt.show()
            ##plt.close()
    
    
    
            if cv2.waitKey(1) == ord('q'):
                break
