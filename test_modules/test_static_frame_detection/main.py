import sys
sys.path.append('../../src')
from static_frame_detection import *

import numpy as np
import math
import os
import sys
import argparse
import cv2
import csv
import glob

# Usage
# python main.py -v ..\..\dataset\videos\CV20_video_1.mp4 -t ..\..\dataset\images\CV20_image_1.png -r .\results

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
    # Path to videos folder
    my_parser.add_argument('-v', '--video_path',
                            action='store',
                            help='the path to a directory containing the .mp4 input videos or to a single file',
                            required=True)
    # Path to images folder
    my_parser.add_argument('-t', '--gound_truth_path',
                            action='store',
                            help='the path to a directory containing the .png ground truth frames or to a single file',
                            required=True)
    # Path where to store results
    my_parser.add_argument('-r', '--results_folder_path',
                            action='store',
                            help='the path to a folder where to store the results')

    # Execute the parse_args() method
    args = vars(my_parser.parse_args())

    is_dir = True
    if not os.path.isdir(args['video_path']): # if not dir, check if file
        is_dir = False
        if not os.path.isfile(args['video_path']): # if not file, report error
            print('Please insert a valid path to a directory containing the .mp4 input videos or to a single file')
            sys.exit()
    
    if is_dir and not os.path.isdir(args['gound_truth_path']): # when dealing with folders, check if gt path is a folder
        print('Please insert a valid path to a directory containing the .png ground truth frames')
        sys.exit()

    if not is_dir and not os.path.isfile(args['gound_truth_path']): # when dealing with files, check if gt path is a file
        print('Please insert a valid path to the a .png file')
        sys.exit()

    save = True
    if not args['results_folder_path']:
        save = False
    else:
        if not os.path.isdir(args['results_folder_path']): # check for path validity
            print('The path to the folder where to store the results does not exists')
            sys.exit()
    
    if is_dir:
        videos = glob.glob(args['video_path'] + '*.mp4')
        videos.sort()
        gt_frames = glob.glob(args['gound_truth_path'] + '*.png')
        gt_frames.sort()
    else:
        videos = [args['video_path']]
        gt_frames = [args['gound_truth_path']]

    for i in range(len(videos)):

        video_name = videos[i].split("\\")[-1].split(".")[0]

        print("Processing " + video_name + " ... ")

        # detect still frame
        frame, frame_index = get_static_frame(videos[i])
        # load gt
        gt_frame = cv2.imread(gt_frames[i], cv2.IMREAD_COLOR)

        frames_vertical_stack = get_one_image(frame, gt_frame)
        
        if save:
            cv2.imwrite(args['results_folder_path'] + '\\' + video_name + '.png', frames_vertical_stack)
        else:
            cv2.imshow('Comparison', cv2.resize(frames_vertical_stack, (0, 0), None, .25, .25))

    print("Testing completed.")
    
    # Closes all the frames
    if cv2.waitKey(0) & 0xFF == ord('q'):  
        cv2.destroyAllWindows() 