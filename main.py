from src.static_frame_detection import *
from src.roi_detection import *
from src.object_detection import *

import numpy as np
import math
import os
import sys
import argparse
import cv2
import csv

# test
# python main.py -i .\dataset\videos\CV20_video_1.mp4 -r .\results
# python main.py -i .\dataset\images\CV20_image_1.png -r .\results

if __name__ == "__main__":

    # Create the parser
    my_parser = argparse.ArgumentParser(description='Detect Them All!, implementation of the "Detect the Bottle Cap" project.',
                                        epilog='by Stefano Esposito')

    # Add the arguments
    my_parser.add_argument('-i', '--input_path',
                            action='store',
                            help='the path to the input .mp4 video or to a .png video frame',
                            required=True)
    my_parser.add_argument('-r', '--results_folder_path',
                            action='store',
                            help='the path to the folder where to store the results',
                            required=True)
    my_parser.add_argument('-p', '--plot',
                            action='store',
                            default='False',
                            help='True to plot the result, False otherwise')

    # Execute the parse_args() method
    args = vars(my_parser.parse_args())

    if not os.path.isfile(args['input_path']):
        print('The path to the input file specified does not exist')
        sys.exit()

    if not os.path.isdir(args['results_folder_path']):
        print('The path to the folder where to store the results does not exists')
        sys.exit()

    if args['plot'] != 'True' and args['plot'] != 'False':
        print('The --plot argument is not valid, it has to be either True or False')
        sys.exit()

    input_name = args['input_path'].split("\\")[-1].split(".")[0]
    input_type = args['input_path'].split("\\")[-1].split(".")[1]

    ## ================== PIPELINE STEP 1 ==================

    print("\n##### PHASE 1 Detection of a still frame #####\n")

    if input_type == "mp4": # input is a video
    
        # detect still frame
        frame, frame_index = get_static_frame(args['input_path'])
        #cv2.imshow('Still frame', cv2.resize(frame, (0, 0), None, .25, .25))
    
    else: # input is an image

        print("The input file is a ." + input_type + ", skipping frame detection")
        frame_index = -1
        frame = cv2.imread(args['input_path'], cv2.IMREAD_COLOR) # TODO remove

    ## ================== PIPELINE STEP 2 ==================

    print("\n##### PHASE 2 Detection of the Region of Interest #####\n")

    # find roi corner points
    print("ROI detection ...",  end = '')
    internal_roi_vertices, external_roi_vertices = find_roi_corner_points(frame)
    print(" OK ")

    # draw the biggest contour
    img_result = frame.copy()
    img_result = draw_rectangle(img_result, internal_roi_vertices, color=(255, 0, 255), thickness=3) 
    img_result = draw_rectangle(img_result, external_roi_vertices, color=(255, 255, 0), thickness=3)
    # cv2.imshow('ROI detected', cv2.resize(img_result, (0, 0), None, .25, .25))

    internal_roi_vertices = internal_roi_vertices.reshape((4, 2)) 
    external_roi_vertices = external_roi_vertices.reshape((4, 2)) 

    # polygon used to check if both bb's top-left and bottom-right vertices
    # are inside the ROI
    roi_polygon = [external_roi_vertices[0], external_roi_vertices[1], external_roi_vertices[3], external_roi_vertices[2]]

    ## ================== PIPELINE STEP 3 ==================

    print("\n##### PHASE 3 YOLO object detection on the frame #####\n")

    # define model
    print("Instantiating yolo model ...", end = '')
    model = yolo_net()
    print(" OK ")

    if False:
        print(model.summary())

    # load pretrained weights
    print("Loading pre-trained model weights ...", end = '')
    model.load_weights('weights/training_1_0.04312899.h5')
    print(" OK ")

    # pass image through the network
    print("Model prediction on selected frame ...",  end = '')
    boxes, scores, classes = predict(model, frame)
    print(" OK ")

    ## ================== PIPELINE STEP 4 ==================

    print("\n##### PHASE 4 Postprocessing object detection results #####\n")

    # postprocess YOLO results
    count_detected = boxes.shape[0] 
    w_img = frame.shape[1]
    h_img = frame.shape[0]

    caps_detected = [0 for i in range(len(LABELS))]

    print("Postprocessing detections ...",  end = '')
    results = []
    for i in range(count_detected):

        box = boxes[i,...].numpy()
        x_resized = box[0]
        y_resized = box[1]
        x_prime_resized = box[2]
        y_prime_resized = box[3]
        predicted_class = classes[i].numpy()

        x = (w_img / IMAGE_W) * x_resized
        y = (h_img / IMAGE_H) * y_resized
        x_prime = (w_img / IMAGE_W) * x_prime_resized
        y_prime = (h_img / IMAGE_H) * y_prime_resized

        cond_1 = is_within_polygon(roi_polygon, (x, y))
        cond_2 = is_within_polygon(roi_polygon, (x_prime, y_prime))
        if cond_1 and cond_2:
            
            predicted = ''
            if predicted_class == 0:
                predicted = "'BottleCap_FaceDown'"
                color = (0, 255, 0)
                caps_detected[0] += 1
            elif predicted_class == 1:
                predicted = "'BottleCap_FaceUp'"
                color = (0, 0, 255)
                caps_detected[1] += 1
            else:
                predicted = "'BottleCap_Deformed'"
                color = (255, 0, 0)
                caps_detected[2] += 1
            
            x = int(round(x))
            y = int(round(y))
            x_prime = int(round(x_prime))
            y_prime = int(round(y_prime))
            
            # zero-based index of the video frame [frame_index]
            # x-coordinate of the detected object position
            # y-coordinate of the detected object position
            # the assigned label
            x_com = math.floor((x_prime + x)/2)
            y_com = math.floor((y_prime + y)/2)
            results.append([frame_index, x_com, y_com, predicted]) # calculating center of mass of the bb
            
            # result image
            img_result = cv2.rectangle(img_result, (x, y), (x_prime, y_prime), color, 3)    
    
    print(" OK ")

    # show object detection results
    print('\n**** BOTTLE CAPS DETECTED **** \n\nFace down:', caps_detected[0], '\nFace up:', caps_detected[1], '\nDeformed:', caps_detected[2])
    
    ## =================== SHOW AND SAVE RESULTS ====================

    if args['plot'] == 'True':
        cv2.imshow('Object detection results', cv2.resize(img_result, (0, 0), None, .5, .5))

    print("Saving results ...",  end = '')

    # export results as .csv-file where entries are always comma separated

    with open(args['results_folder_path'] + '/' + input_name + '.csv', 'w', newline='') as file:
        writer = csv.writer(file, delimiter=',')
        writer.writerows(results)
    
    print(" OK ")

    print("\nExecution completed, please close all windows.")

    # Closes all the frames
    if cv2.waitKey(0) & 0xFF == ord('q'):  
        cv2.destroyAllWindows() 