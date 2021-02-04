import numpy as np
import cv2
import sys
import math
import time
from tqdm import tqdm

def get_static_frame(input_video_path):
    '''TODO'''
    
    # Opening video
    print("Opening video ...",  end = '')

    cap = cv2.VideoCapture(input_video_path)
    number_of_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Check if camera opened successfully
    if (cap.isOpened()== False): 
        print("Error opening video file")
        sys.exit()

    print(" OK ")

    original_frames = []
    grayscale_frames = []

    # Read until video is completed
    for _ in tqdm(range(number_of_frames), desc="Reading video and grayscale conversion"):

        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret==False :
            cap.release() 
            break 
        
        original_frames.append(frame)

        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        grayscale_frames.append(gray)

    # When everything done, release the video capture object
    cap.release()
  
    img_pixels = (original_frames[0].shape[0] * original_frames[0].shape[1])

    # focus only on the central part of the video, excluding most of the final part
    ub = 7/10
    lb = 0.2/10
    last_frame = math.floor(len(original_frames) * ub)
    first_frame = math.floor(len(original_frames) * lb)
    grayscale_frames = grayscale_frames[first_frame:last_frame]

    print("Number of frames in the video:", len(original_frames), "\nNumber of frames under analysis:", len(grayscale_frames))  

    # Smoothing
    smoothed_frames = []
    for i in tqdm(range(len(grayscale_frames)), desc="Video smoothing"):
        median_blur = cv2.medianBlur(grayscale_frames[i], 7)
        smoothed_frames.append(median_blur)
    grayscale_frames = smoothed_frames

    # Find frame with smallest difference from its
    # predecessors and successors
    window_width = 7
    min_pixel_value = 20 # for thresholding
    differences = []

    for i in tqdm(range(window_width // 2, len(grayscale_frames) - window_width // 2, 1), desc="Calculating frames differences"):
        
        accumulation = np.zeros(grayscale_frames[0].shape)
        for j in range(i - window_width // 2, i + window_width // 2, 1):
            if i != j: 
                weight = 1 / abs(j - i) 
                accumulation += weight * cv2.absdiff(grayscale_frames[i], grayscale_frames[j])

        _, thresholded = cv2.threshold(accumulation, min_pixel_value, 255, cv2.THRESH_BINARY)
        diff = (sum(sum(thresholded / 255)) / img_pixels) * 100
        differences.append(diff)
    
    differences = np.array(differences)
    differences = np.round(differences, 2) # round to 2 dec values

    # clamp differences at average diff value
    avg = np.average(differences)
    avg_cut_differences = differences.copy()
    avg_cut_differences[np.where(avg_cut_differences > avg)] = avg

    # find maximum in the first drop_ub % of frames
    # the frame with max differences should correspond to the moment caps are dropped
    drop_ub = 0.65
    drop_ub_frame = int(len(avg_cut_differences) * drop_ub)
    frame_caps_dropped = np.argmax(avg_cut_differences[:drop_ub_frame]) 

    # the static frame we are looking for will be the one with lowest error after 
    # the frame where caps are dropped

    # get all indices of frames with equal lowest error
    min_diff_after_caps_drop = avg_cut_differences[frame_caps_dropped:].min()
    selected_frames = np.where(avg_cut_differences == min_diff_after_caps_drop)

    # take the one with highest index
    selected_frame = np.max(selected_frames) 
    # print("Selected frame in trimmed video: ", selected_frame)
    # original_frame_caps_dropped = frame_caps_dropped + first_frame + window_width // 2
    original_selected_frame = selected_frame + first_frame + window_width // 2
    # print("ORIGINAL VIDEO: lower bound: ", first_frame, " selected frame: ", original_selected_frame, " upper bound: ", last_frame)

    frame = original_frames[original_selected_frame]

    return frame, original_selected_frame