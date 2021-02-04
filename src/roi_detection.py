import cv2
import matplotlib.pyplot as plt
import numpy as np
import json
import sys
import os
import math
import random
#%matplotlib inline
#from __future__ import print_function
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets
import glob

# Parameters
PADDING_SIZE = 60
K = 2 # k-means clustering (2 : separate background color from ROI area color)
ROI_OFFSET = 40
COLORS = [[0, 0, 0], [255, 255, 255]]

# Adapted from: https://gist.github.com/soply/f3eec2e79c165e39c9d540e916142ae1
def show_images(images, cols = 1, titles = None, figsize = None, cmap = None):
    '''Display a list of images in a single figure with matplotlib.
    
    Parameters
    ---------
    images: List of np.arrays compatible with plt.imshow.
    
    cols (Default = 1): Number of columns in figure (number of rows is 
                        set to np.ceil(n_images/float(cols))).
    
    titles: List of titles corresponding to each image. Must have
            the same length as titles.
    '''
    assert((titles is None)or (len(images) == len(titles)))
    n_images = len(images)
    if titles is None: titles = ['Image (%d)' % i for i in range(1,n_images + 1)]
    fig = plt.figure(figsize=figsize)
    for n, (image, title) in enumerate(zip(images, titles)):
        a = fig.add_subplot(cols, int(np.ceil(n_images/float(cols))), n + 1)
        if image.ndim == 2:
            plt.gray()
        plt.imshow(image, cmap=cmap)
        a.set_title(title)
        a.axis('off')
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
    plt.show()


def draw_rectangle(img, vertices, thickness=2, color=(255, 0, 255)):
    '''TODO'''
    cv2.line(img, (vertices[0][0][0], vertices[0][0][1]), (vertices[1][0][0], vertices[1][0][1]), color, thickness)
    cv2.line(img, (vertices[0][0][0], vertices[0][0][1]), (vertices[2][0][0], vertices[2][0][1]), color, thickness)
    cv2.line(img, (vertices[3][0][0], vertices[3][0][1]), (vertices[2][0][0], vertices[2][0][1]), color, thickness)
    cv2.line(img, (vertices[3][0][0], vertices[3][0][1]), (vertices[1][0][0], vertices[1][0][1]), color, thickness)
 
    return img


def postprocess_polygon_vertices(polygon_vertices, image_shape):
    '''Determine the top-left, top-right, bottom-right, 
    and bottom-left points for the ROI polygon and it's external polygon, 
    also removing the padding previously added from the points coordinates '''

    points_internal = np.clip(polygon_vertices.copy().reshape((4, 2)) - PADDING_SIZE, a_min=[0, 0], a_max=[image_shape[1], image_shape[0]])
    internal_polygon_vertices = np.zeros((4, 1, 2), dtype=np.int32)
    
    # the top-left point has the smallest sum whereas the
    # bottom-right has the largest sum
    add = points_internal.sum(1)
    internal_polygon_vertices[0] = points_internal[np.argmin(add)] 
    internal_polygon_vertices[3] = points_internal[np.argmax(add)]  
    
    # compute the difference between the points -- the top-right
    # will have the minumum difference and the bottom-left will
    # have the maximum difference
    diff = np.diff(points_internal, axis=1)
    internal_polygon_vertices[1] = points_internal[np.argmin(diff)]
    internal_polygon_vertices[2] = points_internal[np.argmax(diff)]

    return internal_polygon_vertices


def get_external_polygon_vertices(internal_polygon_vertices, image_shape):
    '''TODO'''

    external_polygon_vertices = internal_polygon_vertices.copy()
    external_polygon_vertices[0, 0] += [-ROI_OFFSET, -ROI_OFFSET] 
    external_polygon_vertices[1, 0] += [ ROI_OFFSET, -ROI_OFFSET] 
    external_polygon_vertices[3, 0] += [ ROI_OFFSET, +ROI_OFFSET] 
    external_polygon_vertices[2, 0] += [-ROI_OFFSET, +ROI_OFFSET]
    external_polygon_vertices = np.clip(external_polygon_vertices, a_min=[0, 0], a_max=[image_shape[1], image_shape[0]])

    return external_polygon_vertices

def full_image_polygon(image_shape):
    '''The ROI polygon covers the entire image. It is equal to its external polygon.'''

    polygon = np.zeros((4, 1, 2), dtype=np.int32)
    polygon[0] = [0, 0]                           # top-left     (0, 0)
    polygon[1] = [image_shape[1], 0]              # top-right    (xmax, 0) 
    polygon[2] = [0, image_shape[0]]              # bottom-left  (0, ymax) 
    polygon[3] = [image_shape[1], image_shape[0]] # bottom-right (xmax, ymax)
    
    return polygon


def find_roi_corner_points(image, plot=False):
    '''TODO'''
    # Sources: 
    #   https://www.pyimagesearch.com/2014/04/21/building-pokedex-python-finding-game-boy-screen-step-4-6/
    #   https://github.com/opencv/opencv/blob/master/samples/python/squares.py

    image_area = image.shape[0] * image.shape[1]
    lb_area = image_area * 0.05
    ub_area = image_area * 0.99
    min_vertical_length = image.shape[0] * 0.1
    min_horizontal_length = image.shape[1] * 0.1

    # list of all the valid 4-vertices polygons found
    polygons_found = []
    
    # https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_filtering/py_filtering.html#bilateral-filtering
    # Highly effective at noise removal while preserving edges
    bilateral_blur = cv2.bilateralFilter(image, 9, 75, 75) 

    # http://www.cse.msu.edu/~pramanik/research/papers/2002Papers/icip.hsv.pdf
    image_for_clustering = cv2.cvtColor(bilateral_blur, cv2.COLOR_BGR2HSV) 

    # vectorize image
    vectorized = np.float32(image_for_clustering.reshape((-1,3)))

     # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0) #  ( type, max_iter, epsilon )
    _, label, _ = cv2.kmeans(vectorized, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS) # alternative is cv2.KMEANS_PP_CENTERS
    
    colors = np.uint8(COLORS)

    # make image
    res = colors[label.flatten()]
    clustered_image = res.reshape((image.shape))

    # convert to grayscale to get a binary image
    gray = cv2.cvtColor(clustered_image, cv2.COLOR_BGR2GRAY)
    
    # postprocess (?)
    # https://docs.opencv.org/master/d9/d61/tutorial_py_morphological_ops.html
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    # threshold = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, kernel, iterations=1)
    # threshold = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, kernel, iterations=1)

    # for each padding_color (black or white)
    for padding_color in COLORS:
        
        # Add padding to the thresholded image
        padded_image = cv2.copyMakeBorder(gray.copy(),
                                          PADDING_SIZE, PADDING_SIZE, PADDING_SIZE, PADDING_SIZE,
                                          cv2.BORDER_CONSTANT, value=padding_color)
        
        # detecting shapes in image by selecting region with same colors or intensity 
        contours, _ = cv2.findContours(padded_image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE) # cv2.RETR_TREE
        contours = sorted(contours, key=cv2.contourArea)
        contours = contours[5:]
        
        # Draw all detected contours
        padded_image = cv2.drawContours(cv2.cvtColor(padded_image,cv2.COLOR_GRAY2RGB), contours, -1, (0, 255, 0), 10)
        
        arc_lenght_percentages = np.flip(np.arange(0.01, 0.1, 0.01))
        # arc_lenght_percentages = np.arange(0.01, 0.1, 0.01)
        for arc_lenght_percentage in arc_lenght_percentages: # arc_lenght_percentages: 
            
            for contour in contours:

                area = cv2.contourArea(contour) 
                
                # skip contour is area is outside the limits
                if area < lb_area or area > ub_area: # or area <= max_poly_area:
                    continue
                
                # Use of Ramer–Douglas–Peucker algorithm:
                # https://en.wikipedia.org/wiki/Ramer%E2%80%93Douglas%E2%80%93Peucker_algorithm
                # epsilon is maximum distance from contour to approximated contour, it is an accuracy parameter.
                
                epsilon = arc_lenght_percentage * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)

                # square contours should have 4 vertices after approximation and be convex.
                if len(approx) == 4 and cv2.isContourConvex(approx):

                    poly = postprocess_polygon_vertices(approx, image.shape)
                    approx = approx.reshape(-1, 2)

                    # Find the maximum cosine of the angle between joint edges
                    max_cos = np.max([angle_cos( approx[i], approx[(i+1) % 4], approx[(i+2) % 4] ) for i in range(4)])

                    # if cosines of all angles not too large 
                    # (let's say that all angles are between 45 and 90 degree)
                    if (max_cos < 0.7):

                        # check sides length
                        b_l = poly[0, 0]
                        b_r = poly[1, 0]
                        t_l = poly[2, 0]
                        t_r = poly[3, 0]

                        if not vertices_all_inside_image_or_corners([b_l, b_r, t_l, t_r], image.shape):
                            continue

                        bottom_horizontal_length = b_r[0] - b_l[0]
                        top_horizontal_length    = t_r[0] - t_l[0]
                        left_vertical_length     = t_l[1] - b_l[1]
                        right_vertical_length    = t_r[1] - b_r[1]

                        if ( bottom_horizontal_length > min_horizontal_length and
                             top_horizontal_length > min_horizontal_length and
                             left_vertical_length > min_vertical_length and
                             right_vertical_length > min_vertical_length):

                                polygons_found.append([poly, area, max_cos])

            # polygoins found, break from arc lenght percentage loop
            # if len(polygons_found) != 0:
            #    break
        
        # polygoins found, break from padding color loop
        if len(polygons_found) != 0:
            break
    
    # check if a polygon has been found
    if len(polygons_found) == 0: 
        # if not, then return a polygon big as the image
        internal_polygon_vertices = full_image_polygon(image.shape)
        external_polygon_vertices = get_external_polygon_vertices(internal_polygon_vertices, image.shape)
    else:
        # order found polygons by decreasing area and increasing max_cos
        polygons_found = sorted(polygons_found, key = lambda x: (-x[1], x[2]))
        #for f in polygons_found:
        #    print(f)
        internal_polygon_vertices = polygons_found[0][0]
        external_polygon_vertices = get_external_polygon_vertices(internal_polygon_vertices, image.shape)
    
    # if plot:
    #      # draw the biggest contour
    #      image_polygon_vertices = draw_rectangle(image.copy(), polygon_vertices, 5) 
    #      show_images([padded_image, image_polygon_vertices[...,::-1]], titles=["color separated (and padded) image", "approximated area polygon"]) 
    
    return internal_polygon_vertices, external_polygon_vertices

"""
def find_perspective_transf_matrix(vertices):
    ''' Return a perspective transform matrix given 4 rectangular vertices '''
    # adapted from: https://www.pyimagesearch.com/2014/05/05/building-pokedex-python-opencv-perspective-warping-step-5-6/
    
    # compute the width of the new image
    (tl, tr, bl, br) = vertices
    width_A = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    width_B = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))

    # and the height of the new image
    height_A = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    height_B = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))

    # take the maximum of the width and height values to reach our final dimensions
    max_width = max(int(width_A), int(width_B))
    max_height = max(int(height_A), int(height_B))

    # construct the destination points to be used to map the ROI to a top-down, "birds eye" view
    dst = np.array(
                    [
                        [0, 0],
                        [max_width - 1, 0],
                        [max_width - 1, max_height - 1],
                        [0, max_height - 1]
                    ], 
                    dtype = "float32"
                )
    
    # adjust rectangle representation
    rect = np.zeros((4, 2), dtype = "float32")
    rect[0] = vertices[0]
    rect[1] = vertices[1]
    rect[2] = vertices[3]
    rect[3] = vertices[2]
    
    # calculate the perspective transform matrix and warp the perspective to grab the ROI
    T = cv2.getPerspectiveTransform(rect, dst)
    
    return T, max_width, max_height
"""

def get_bounding_boxes(data):
    '''TODO'''

    bbs = []

    for shape in data['shapes']: 

        points = shape['points']
        points = np.array(points)
        
        bb = []

        xmin = np.min(points[:, 0])
        xmax = np.max(points[:, 0])
        ymin = np.min(points[:, 1])
        ymax = np.max(points[:, 1])

        bb.append(math.floor(xmin))
        bb.append(math.floor(ymin))
        bb.append(math.floor(xmax))
        bb.append(math.floor(ymax))
        bb.append(shape['label'])
        
        bbs.append(bb)
        
    return bbs

"""
def warp_bounding_boxes(bbs, T):
    '''TODO'''

    new_bbs = []
    for bb in bbs:
        
        pointsIn = np.array([[bb[0], bb[1]], [bb[2], bb[3]]], dtype='float32')
        pointsIn = np.array([pointsIn])
        pointsOut = cv2.perspectiveTransform(pointsIn, T)

        new_bb = []
        top_left = pointsOut[0][0]
        new_bb.append(math.floor(top_left[0])) # xmin
        new_bb.append(math.floor(top_left[1])) # ymin
        bottom_right = pointsOut[0][1]
        new_bb.append(math.floor(bottom_right[0])) # xmax
        new_bb.append(math.floor(bottom_right[1])) # ymax
        new_bb.append(bb[4])
        
        new_bbs.append(new_bb)

    return new_bbs
"""

def is_point_contained(point, xmax, ymax):
    '''TODO'''
    
    p_x = point[0]
    p_y = point[1]
    cond_1 = 0 <= p_x and p_x <= xmax
    cond_2 = 0 <= p_y and p_y <= ymax
    return cond_1 and cond_2


# credits: https://algorithmtutor.com/Computational-Geometry/Check-if-a-point-is-inside-a-polygon/
# usage: is_within_polygon([(0, 0), (5, 0), (6, 7), (2, 3)], point)
def is_within_polygon(polygon, point):
    '''
    polygon : a CCW list of points defining the convex border
    point : (x, y)
    '''

    A = []
    B = []
    C = []  
    for i in range(len(polygon)):
        p1 = polygon[i]
        p2 = polygon[(i + 1) % len(polygon)]
        
        # calculate A, B and C
        a = -(p2[1] - p1[1])
        b = p2[0] - p1[0]
        c = -(a * p1[0] + b * p1[1])

        A.append(a)
        B.append(b)
        C.append(c)

    D = []
    for i in range(len(A)):
        d = A[i] * point[0] + B[i] * point[1] + C[i]
        D.append(d)

    t1 = all(d >= 0 for d in D)
    t2 = all(d <= 0 for d in D)
    return t1 or t2


def angle_cos(p0, p1, p2):
    '''TODO'''

    d1, d2 = (p0-p1).astype('float'), (p2-p1).astype('float')
    return abs( np.dot(d1, d2) / np.sqrt( np.dot(d1, d1)*np.dot(d2, d2) ) )


def vertices_all_inside_image_or_corners(vertices, image_shape):
    '''TODO'''
    
    # print(vertices)
    corner_points = [[0, 0], [0, image_shape[0] - 1], [image_shape[1] - 1, image_shape[0] - 1], [image_shape[1] - 1, 0]]
    
    for vertex in vertices:
        
        found = False
        for i in range(len(corner_points)):
            cp = corner_points[i]
            if (vertex[0] == cp[0] and vertex[1] == cp[1]):
                found = True
                index = i
                break
        
        if found: 
            del corner_points[index]
    
    # print(corner_points)
    
    if len(corner_points) == 4 or len(corner_points) == 0:
        return True
    else:
        return False