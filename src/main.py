from pathlib import Path
import sys
import cv2
import depthai as dai
import numpy as np
import time
import matplotlib.pyplot as plt
import os

from alarm_device import alarm_device
from logger import build_logger

import logging

logging.getLogger().setLevel(logging.INFO)

#Global variables
labelMap = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
                "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
syncNN = True

nnBlobPath = str((Path(__file__).parent / Path('../models/mobilenet.blob')).resolve().absolute())

not_street = False   # 'True' for experiments with miniatures.  'False' for real life distances.

valid_objects = ["car", "person"]

# check the list of objects to see if there's an object that has come out of a frame for more than 2sec and delete it
def clean_redundant_data(objects, current_time):
    l = [] # list of indices of objects (cars or persons) which has come out of a frame and should be deleted from cars or persons tracking list
    # check for each object in list of objects:
    for i in range(len(objects)):
        if current_time - objects[i][1] > 2:
            l.append(i)
    # create new list
    objects = [obj for idx,obj in enumerate(objects) if idx not in l]

    return objects


def write_id_in_frame(objects, frame):
    for obj in objects:
        id, point = obj[0], (obj[-1][0], obj[-1][1] - 10)
        cv2.putText(frame, f"{id}", point, cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0,0,255))
        
    return 

  
# check if the deviations in series of coordinates, needed to calculate the direction of object movement, are below the assumed threshold
# reject a random error that occurs in depth coords x_depth,y_depth,z_depth
def check_deviation_of_depth_coords(obj, x_center, y_center, x_depth, y_depth, z_depth):   # this function should be called if len(obj)>6
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

    # if x_center, y_center is within the mean deviation and value of x_depth or y_depth or z_depth is very different then the last one, ignore this depth value 
    # and assigne a calculated value which differ from the mean no more than 0.01
    if abs(obj[-1][0] - x_center) <= dx and abs(obj[-1][1] - y_center) <= dy and abs(obj[-1][2] - x_depth) > (dX * 2):
        x_depth = ((obj[4][2] + obj[5][2] + obj[6][2]) / 3) * 1.01
    if abs(obj[-1][0] - x_center) <= dx and abs(obj[-1][1] - y_center) <= dy and abs(obj[-1][3] - y_depth) > (dY * 2):
        y_depth = ((obj[4][3] + obj[5][3] + obj[6][3]) / 3) * 1.01
    if abs(obj[-1][0] - x_center) <= dx and abs(obj[-1][1] - y_center) <= dy and abs(obj[-1][4] - z_depth) > dZ and (obj[-1][4] - z_depth) < 0:
        z_depth = ((obj[4][4] + obj[5][4] + obj[6][4]) / 3) * 1.01
    if abs(obj[-1][0] - x_center) <= dx and abs(obj[-1][1] - y_center) <= dy and abs(obj[-1][4] - z_depth) > dZ and (obj[-1][4] - z_depth) > 0:
        z_depth = ((obj[4][4] + obj[5][4] + obj[6][4]) / 3) * 0.99
        
        
    return x_depth,y_depth,z_depth

  

def print_data_of_detected(obj_list, obj_name):
    oppos_id = "person"
    if obj_name == "person":
        oppos_id = "car"
    if obj_list:
        for predecessor_person in obj_list:
            print(f'  <{obj_name}_{predecessor_person[0]}>:')    
            if predecessor_person[3]:
                for e in predecessor_person[3]:
                    print(f'       trajectory crossing with path of {oppos_id}_{e[1]}:')  
                    print(f'          distance2collision: {e[2]:.2f}m,  {obj_name}_speed: {e[4][0]:.2f}m/s, {obj_name}_time2crash: {e[4][1]:.3f}s') 
            if len(predecessor_person) == 7: print(f'       {obj_name}_adjusted_coords>  pos1:{predecessor_person[4]},  pos2:{predecessor_person[5]},  last_pos:{predecessor_person[6]}')
            if len(predecessor_person) == 6: print(f'       {obj_name}_adjusted_coords>  pos1:{predecessor_person[4]},  pos2:{predecessor_person[5]}')
            if len(predecessor_person) == 5: print(f'       {obj_name}_adjusted_coords>  pos1:{predecessor_person[4]}')
    else:
        print(f'No {obj_name} in the designated field.')


        
        
# Update data related to the potencial collision for each person or car
def replace_insert_crashdata(collide_list, new_intersect):
    alarm_flag = False  # Suppose that specific object(e.g. person) is not in danger of a collision 
    if collide_list:
        i = 0    # an index of the event in the collide_list of that specific object(e.g. a person)
        # search for an object(e.g. a car), if it exist on the list of potential participants of a collision with that specific object(in this case a person)
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
                if time_to_collision < 2 and time_to_collision > 0:   # if time to collision < 2sec
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

  
def delete_unnecessary_crash_points(crashpoints, objects_heading_2_collision):
    if len(crashpoints) != 0:
        for obj_number in [crashpoint[1] for crashpoint in crashpoints]:
            if obj_number not in [obj[0] for obj in objects_heading_2_collision]:
                crashpoints = [crashpoint for crashpoint in crashpoints if crashpoint[1] != obj_number]

    return crashpoints

def draw_data_on_frame(frame, detection):


    cv2.putText(frame, str(object_label), (x_min + 10, y_min + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
    cv2.putText(frame, "{:.2f}".format(detection.confidence*100), (x_min + 10, y_min + 35), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
    cv2.putText(frame, f"x_depth: {x_depth} m", (x_min + 10, y_min + 50), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
    cv2.putText(frame, f"y_depth: {y_depth} m", (x_min + 10, y_min + 65), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
    cv2.putText(frame, f"z_depth: {z_depth} m", (x_min + 10, y_min + 80), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, cv2.FONT_HERSHEY_SIMPLEX)
    cv2.circle(frame, (x_center, y_center), 5, (0,0,255), -1)  
    cv2.putText(frame, "NN fps: {:.2f}".format(fps), (2, frame.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.4, color)
    
    return frame 
 

# Tracking persons or cars ; e.g.: track_object(cars, 'car', car_id, object_label, x_center, y_center, x_depth, y_depth, z_depth, not_street)
def track_object(object_list, object_name, object_id, object_label, x_center, y_center, x_depth, y_depth, z_depth, not_street):
    # updates object id and localization in the frame
    if len(object_list) > 0 and (object_label == object_name):  # if object_list is not empty and it's that object name ('person' or 'car')
        object_index = 0  #index of an object in object_list
        object_not_found = True
        # Check if the object is already on the list. Keep trying until the object is found or the list is over.
        while object_not_found:   
            predecessor_object = object_list[object_index]  #predecessor data
            if len(predecessor_object) > 6:
                x_depth, y_depth, z_depth = check_deviation_of_depth_coords(predecessor_object, x_center, y_center, x_depth, y_depth, z_depth)
            if (abs(predecessor_object[-1][0]-x_center) < 50) and (abs(predecessor_object[-1][1]-y_center) < 50) and (abs(predecessor_object[-1][2]-x_depth) < 0.500) and (abs(predecessor_object[-1][3]-y_depth) < 0.500) and (abs(predecessor_object[-1][4]-z_depth) < 1.000):
                p_time = time.monotonic()
                # if it is not a "hole" value (depth measurement error), add new coordinates of an object
                if x_depth != 0 or y_depth != 0:
                    predecessor_object[1] = p_time
                    predecessor_object.append((x_center, y_center, x_depth, y_depth, z_depth))    # 
                if len(predecessor_object) > 7:  # leave only the last three positions of the object, needed to calculate the direction of movement
                    del predecessor_object[4]
                object_not_found = False
                continue
            elif object_index < len(object_list)-1:   # try to get next object from the list
                object_index += 1
            # if a predecessor of the object is not found in the object_list, append a new object 
            else:            
                # check if it is not a "hole" value (depth measurement error)
                if not_street and (x_depth != 0 or y_depth != 0) and z_depth != 0:    # for very close distances
                    p_time = time.monotonic()
                    object_id += 1
                    object_list.append([object_id, p_time, ([0,0,0],[0,0,0],[0,0,0]), [], (x_center, y_center, x_depth, y_depth, z_depth)])   # append coordinates to the list as a new object position 
                    object_not_found = False
                elif z_depth > 5:        # for real life
                    p_time = time.monotonic()
                    object_id += 1
                    object_list.append([object_id, p_time, ([0,0,0],[0,0,0],[0,0,0]), [], (x_center, y_center, x_depth, y_depth, z_depth)])   # append coordinates to the list as a new object position 
                    object_not_found = False
    elif object_label == object_name:     # append the first object
        p_time = time.monotonic()
        # append obj id, last possition detection time, extrapolation line parameters(p0,pn,v), intersection point coords and obj id-s, spatial position
        if not_street and (x_depth != 0 or y_depth != 0) and z_depth != 0:
            object_list.append([object_id, p_time, ([0,0,0],[0,0,0],[0,0,0]), [], (x_center, y_center, x_depth, y_depth, z_depth)])
        elif z_depth > 5: # if distance from the cam is bigger then 5m
            object_list.append([object_id, p_time, ([0,0,0],[0,0,0],[0,0,0]), [], (x_center, y_center, x_depth, y_depth, z_depth)])

    return object_list, object_id



'''
    Performs inference on RGB camera and retrieves spatial location coordinates: x,y,z relative to the center of depth map.
'''

if __name__=="__main__":
    log = build_logger()
    log.debug("Test of log to file")
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
        frames_counter = 0
        fps = 0
        color = (255, 255, 255)
    
        # create lists to collect persons and vehicles last position (x_depth,y_depth coordinates)
        cars, persons = [], []
        car_id = 0
        person_id = 0
        operation_count = 0   # frame number
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        
        out = cv2.VideoWriter("out.mp4", fourcc, 30, (300, 300), True)

        while True:
            try:
                inPreview = previewQueue.get()
                inNN = detectionNNQueue.get()
                depth = depthQueue.get()
    
                frames_counter+=1
                current_time = time.monotonic()
                if (current_time - startTime) > 1 :
                    fps = frames_counter / (current_time - startTime)
                    frames_counter = 0
                    startTime = current_time
        
                frame = inPreview.getCvFrame()
                detections = inNN.detections
            
                operation_count += 1
                bb_number = 0
                detections_list = []
        
                # if the frame is available, draw bounding boxes on it and show the frame
                
                height = frame.shape[0]
                width  = frame.shape[1]
                for detection in detections:
                    # denormalize bounding box
                    x_min = int(detection.xmin * width)
                    x_max = int(detection.xmax * width)
                    y_min = int(detection.ymin * height)
                    y_max = int(detection.ymax * height)
                    #gets the center point of a bounding box
                    x_center = (x_max+x_min)//2
                    y_center = (y_max+y_min)//2
        
                    # get value of spatial coords in meters
                    x_depth = int(detection.spatialCoordinates.x) / 1000
                    y_depth = int(detection.spatialCoordinates.y) / 1000
                    z_depth = int(detection.spatialCoordinates.z) / 1000
        
                    try:
                        object_label = labelMap[detection.label]
                    except:
                        object_label = detection.label
                    
                    #fill out the checking list
                    detections_list.append((bb_number, object_label, x_center, y_center, x_depth, y_depth, z_depth))
                    bb_number += 1
                    
                    # Draw data in the frame
                    if object_label in valid_objects:
                        frame = draw_data_on_frame(frame, detection)
        
        #---tracking---------------- 
        
                    persons, person_id = track_object(persons, 'person', person_id, object_label, x_center, y_center, x_depth, y_depth, z_depth, not_street)
                    persons = clean_redundant_data(persons, current_time)
                    write_id_in_frame(persons, frame)
                    
                    
                    cars, car_id = track_object(cars, 'car', car_id, object_label, x_center, y_center, x_depth, y_depth, z_depth, not_street)
                    cars = clean_redundant_data(cars, current_time)  
                    write_id_in_frame(cars, frame)
        #---------------------------
                    
                print(f'\nF>{operation_count}, ct>{current_time}, Detected : {detections_list}')
                print('IN FRAME:')
       
                     
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
                    print('Searching for Pedestrian-Car Crash Point...')
                    for c in cars:
                        if len(c) > 6:
                            c[3] = delete_unnecessary_crash_points(c[3], persons)
                            for predecessor_person in persons:
                                if len(predecessor_person) > 6:
                                    predecessor_person[3] = delete_unnecessary_crash_points(predecessor_person[3], cars)  # update crash points list
                                    ## if the set of coefficients of the direction vectors of two lines, car and person routs, are proportional these lines are parallel to each other
                                    #and their direction vectors Cross Product equals 0
                                    if any(np.cross(c[2][2], predecessor_person[2][2])):         
                                        # get a point on the car's line and its direction vector coefficients 
                                        x_1, a_1, y_1, b_1, z_1, c_1 = c[2][0][0], c[2][2][0], c[2][0][1], c[2][2][1], c[2][0][2], c[2][2][2]
                                        # Get a normal vector of a person's plane
                                        n = np.cross(predecessor_person[2][2], np.array([0, 10, 0]))
                                        nx, ny, nz = n[0], n[1], n[2]
                                        # get a point zero on the person's plane
                                        x0, y0, z0 = predecessor_person[2][0][0], predecessor_person[2][0][1], predecessor_person[2][0][2] #coords of pp0
                                        d = -(nx*x0 + ny*y0 + nz*z0)
                                        # find car's line t coefficient
                                        if (nx*a_1 + ny*b_1 + nz*c_1) != 0:
                                            t = -(d + nx*x_1 + ny*y_1 + nz*z_1) / (nx*a_1 + ny*b_1 + nz*c_1)
                                            # intersection point
                                            xi = (a_1*t + x_1)
                                            yi = (b_1*t + y_1)
                                            zi = (c_1*t + z_1)
                                            # distance to the crash point:
                                            p_distance = np.sqrt(sum(e**2 for e in (np.array([xi, yi, zi]) - predecessor_person[2][1])))  # pedestrian
                                            c_distance = np.sqrt(sum(e**2 for e in (np.array([xi, yi, zi]) - c[2][1])))  # car
                                            p_last_pos = predecessor_person[2][1]  # last position of a pedestrian                                        
                                            p_time = predecessor_person[1]         # the time of detecting the last position of a pedestrian
                                            # insert intersection coords, an id of a car the person can collide with, distance to the hypothetical crash point etc.:
                                            alarm_flag_p, predecessor_person[3] = replace_insert_crashdata(predecessor_person[3], [np.array([xi, yi, zi]), c[0], p_distance, p_time, (0, 0), p_last_pos])
    
                                            print(f'    Person{predecessor_person[0]} -> Possible collision: {alarm_flag_p}')
    
                                            # insert intersection coords, an id of a person the car can collide with, distance to the hypothetical crash point etc.:
                                            alarm_flag_c, c[3] = replace_insert_crashdata(c[3], [np.array([xi, yi, zi]), predecessor_person[0], c_distance, c[1], (0, 0), c[2][1]])
    
                                            print(f'    Car{c[0]}    -> Possible collision: {alarm_flag_c}')
                                            print("    ---")
    
                                            ## ALARM
                                            if alarm_flag_p and alarm_flag_c:
                                                print(f'ALARM!!! ALARM!!! Person{predecessor_person[0]} is going to collide with Car{c[0]}')   
                                            # raise an alarm or print a reassuring message    
                                            alarm_device(alarm_flag_p, alarm_flag_c, predecessor_person[0], c[0])
    
                            # draw in the frame a line connecting each pair of a person and a car, when person's time_to_collision is computed
                            if predecessor_person[3]: 
                                for crash in predecessor_person[3]:
                                    if crash[4][1] > 0:  # person's time to the collision
                                        carid = crash[1]  # id of the car involved in
                                        for v in cars:    # v-vehicle
                                            if (v[0] == carid) and (len(v) > 6):
                                                car_2d_pos = (v[6][0],v[6][1])     # location in frame
                                                person_2d_pos = (predecessor_person[6][0],predecessor_person[6][1])
                                                cv2.line(frame, person_2d_pos, car_2d_pos, (255,0,0), 1)
                                                if (crash[4][1] < 2) and (crash[2] < 10):  # if person's time to the collision is less than 2sec and the distance is less than 10m
                                                    cv2.putText(frame, "Collision in 1sec!", (20, 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0,255,255))
        
                # print selected data from trackers
                print_data_of_detected(persons, 'person')
                print_data_of_detected(cars, 'car')        
        
                cv2.imshow("rgb", frame)
                out.write(frame)
        
                if cv2.waitKey(100) == ord('q'):
                    out.release()
                    break
            except KeyboardInterrupt:
                out.release()
                quit()
