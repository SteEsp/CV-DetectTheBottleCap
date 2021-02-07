import sys
sys.path.append('../../src')
from roi_detection import *

import numpy as np
import math
import os
import sys
import argparse
import cv2
import csv
import glob

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
    gt_frame = frame.copy() # for comparison

    # find roi corner points
    internal_roi_vertices, external_roi_vertices = find_roi_corner_points(frame)
    
    # draw the biggest contour
    frame = draw_rectangle(frame, internal_roi_vertices, color=(255, 0, 255), thickness=3) 
    frame = draw_rectangle(frame, external_roi_vertices, color=(255, 255, 0), thickness=3)

    # read annotation json
    with open(args['annotation_path']) as json_file:
        annotation = json.load(json_file)

    # get original image buinding boxes
    bounding_boxes = get_bounding_boxes(annotation)
    
    internal_roi_vertices = internal_roi_vertices.reshape((4, 2)) 
    external_roi_vertices = external_roi_vertices.reshape((4, 2)) 
    
    # polygon used to check if both bb's top-left and bottom-right vertices
    # are inside the ROI
    roi_polygon = [external_roi_vertices[0], external_roi_vertices[1], external_roi_vertices[3], external_roi_vertices[2]]

    # plot them on the original image
    for bb in bounding_boxes:

        # check if both bb's top left and bottom right vertices
        # are inside the ROI
        cond_1 = is_within_polygon(roi_polygon, (bb[0], bb[1]))
        cond_2 = is_within_polygon(roi_polygon, (bb[2], bb[3]))

        if bb[4] == 'BottleCap_FaceDown':
            color = (0, 255, 0)
        elif bb[4] == 'BottleCap_FaceUp':
            color = (0, 0, 255)
        else:
            color = (255, 0, 0)

        if cond_1 and cond_2:
            frame = cv2.rectangle(frame, (bb[0], bb[1]), (bb[2], bb[3]), color, 3) # drawn only if inside ROI
        
        gt_frame = cv2.rectangle(gt_frame, (bb[0], bb[1]), (bb[2], bb[3]), color, 3)  # always draw the ground truth
    
    frames_vertical_stack = get_one_image(frame, gt_frame)
    
    if save:
        cv2.imwrite(args['results_folder_path'] + '\\' + file_name + '.png', frames_vertical_stack)
    else:
        cv2.imshow('Comparison', cv2.resize(frames_vertical_stack, (0, 0), None, .4, .4))

    print("Testing completed.")
    
    # Closes all the frames
    if cv2.waitKey(0) & 0xFF == ord('q'):  
        cv2.destroyAllWindows() 