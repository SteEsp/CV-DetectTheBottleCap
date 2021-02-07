import sys
import glob
sys.path.append('../../src')
from roi_detection import *
from object_detection import *

import numpy as np
import math
import os
import sys
import argparse
import cv2
import csv

# Usage
# python main.py -f ..\..\dataset\images\CV20_image_1.png -a ..\..\dataset\annotations\CV20_label_renamed_1.json

def get_one_image(frame_, gt_frame_):

    if frame_.shape[0] > frame_.shape[1]:
        frame = cv2.rotate(frame_, cv2.ROTATE_90_CLOCKWISE)
    else:
        frame = frame_
    gt_frame = gt_frame_

    max_width = max(frame.shape[1], gt_frame.shape[1])
    total_height = frame.shape[0] + gt_frame.shape[0]

    # create a new array with a size large enough to contain all the images
    final_image = np.zeros((total_height, max_width, 3), dtype=np.uint8)

    current_y = 0  # keep track of where your current image was last placed in the y coordinate

    frame = np.hstack((frame, np.zeros((frame.shape[0], max_width - frame.shape[1], 3))))
    final_image[current_y:current_y + frame.shape[0], :, :] = frame
    
    current_y += frame.shape[0]

    gt_frame = np.hstack((gt_frame, np.zeros((gt_frame.shape[0], max_width - gt_frame.shape[1], 3))))
    final_image[current_y:current_y + gt_frame.shape[0], :, :] = gt_frame

    return final_image

if __name__ == "__main__":

    # Create the parser
    my_parser = argparse.ArgumentParser(description='Detect Them All! [App Testing], implementation of the "Detect the Bottle Cap" project.',
                                        epilog='by Stefano Esposito')

    # Add the arguments
    # Path to images folder
    my_parser.add_argument('-f', '--frame_path',
                            action='store',
                            help='the path to a .png file',
                            required=True)
    # Path to annotations folder
    my_parser.add_argument('-a', '--annotation_path',
                            action='store',
                            help='the path to the corresponding annotation .json file',
                            required=True)
    # Path where to store results
    my_parser.add_argument('-r', '--results_folder_path',
                            action='store',
                            help='the path to a folder where to store the result')

    # Execute the parse_args() method
    args = vars(my_parser.parse_args())

    if not os.path.isfile(args['frame_path']): # if not file, report error
        print('Please insert a valid path to a .png file')
        sys.exit()

    if not os.path.isfile(args['annotation_path']): # check if annotation path is a file
        print('Please insert a valid path to the a .json file')
        sys.exit()

    save = True
    if not args['results_folder_path']:
        save = False
    else:
        if not os.path.isdir(args['results_folder_path']): # check for path validity
            print('The path to the folder where to store the results does not exists')
            sys.exit()

    file_name = args['frame_path'].split("\\")[-1].split(".")[0]

    print("\n*** PROCESSING  " + file_name + " ***\n")

    frame = cv2.imread(args['frame_path'], cv2.IMREAD_COLOR)

    ########### Make ground truth image ###########
    gt_frame = frame.copy()

    # read annotation json
    with open(args['annotation_path']) as json_file:
        annotation = json.load(json_file)
    
    # get original image buinding boxes
    bounding_boxes = get_bounding_boxes(annotation)

    # plot them on the original image
    for bb in bounding_boxes:

        if bb[4] == 'BottleCap_FaceDown':
            color = (0, 255, 0)
        elif bb[4] == 'BottleCap_FaceUp':
            color = (0, 0, 255)
        else:
            color = (255, 0, 0)

        gt_frame = cv2.rectangle(gt_frame, (bb[0], bb[1]), (bb[2], bb[3]), color, 3)  # always draw the ground truth
    
    ############ Make my predictions image ###########
    img_result = frame.copy()

    # find roi corner points
    internal_roi_vertices, external_roi_vertices = find_roi_corner_points(frame)
    
    # draw the biggest contour
    img_result = draw_rectangle(img_result, internal_roi_vertices, color=(255, 0, 255), thickness=3) 
    img_result = draw_rectangle(img_result, external_roi_vertices, color=(255, 255, 0), thickness=3)

    internal_roi_vertices = internal_roi_vertices.reshape((4, 2)) 
    external_roi_vertices = external_roi_vertices.reshape((4, 2)) 

    # polygon used to check if both bb's top-left and bottom-right vertices
    # are inside the ROI
    roi_polygon = [external_roi_vertices[0], external_roi_vertices[1], external_roi_vertices[3], external_roi_vertices[2]]

    # define model
    print("Instantiating yolo model ...", end = '')
    model = yolo_net()
    print(" OK ")

    # load pretrained weights
    print("Loading pre-trained model weights ...", end = '')
    model.load_weights('../../weights/training_1_0.04312899.h5')
    print(" OK ")

    # pass image through the network
    print("Model prediction on selected frame ...",  end = '')
    boxes, scores, classes = predict(model, frame)
    print(" OK ")

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
                predicted = '\'BottleCap_FaceDown\''
                color = (0, 255, 0)
                caps_detected[0] += 1
            elif predicted_class == 1:
                predicted = '\'BottleCap_FaceUp\''
                color = (0, 0, 255)
                caps_detected[1] += 1
            else:
                predicted = '\'BottleCap_Deformed\''
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
            results.append([0, x_com, y_com, predicted]) # calculating center of mass of the bb

            # result image
            img_result = cv2.rectangle(img_result, (x, y), (x_prime, y_prime), color, 3)    
        
    print(" OK ")

    # show object detection results
    print('Bottle caps detected: \n Face down:', caps_detected[0], '\nFace up:', caps_detected[1], '\nDeformed:', caps_detected[2])
    
    # save or plot image
    frames_vertical_stack = get_one_image(img_result, gt_frame)
    if save:
        # save comparison image
        cv2.imwrite(args['results_folder_path'] + '\\' + file_name + '.png', frames_vertical_stack)
        # save predictions csv
        with open(args['results_folder_path'] + '\\' + file_name + '.csv', 'w', newline='') as file:
            writer = csv.writer(file, delimiter=',')
            writer.writerows(results)
    else:
        cv2.imshow('Comparison', cv2.resize(frames_vertical_stack, (0, 0), None, .4, .4))
        
    print("Testing completed.")
    
    # Closes all the frames
    if cv2.waitKey(0) & 0xFF == ord('q'):  
        cv2.destroyAllWindows() 