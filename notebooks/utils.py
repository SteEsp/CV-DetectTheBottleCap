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


# Adapted from: https://gist.github.com/soply/f3eec2e79c165e39c9d540e916142ae1
def show_images(images, cols = 1, titles = None, figsize = None, cmap = None):
    """Display a list of images in a single figure with matplotlib.
    
    Parameters
    ---------
    images: List of np.arrays compatible with plt.imshow.
    
    cols (Default = 1): Number of columns in figure (number of rows is 
                        set to np.ceil(n_images/float(cols))).
    
    titles: List of titles corresponding to each image. Must have
            the same length as titles.
    """
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


def postprocess_polygon_vertices(polygon_vertices, image_shape, PADDING_SIZE):
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


def get_external_polygon_vertices(internal_polygon_vertices, image_shape, ROI_OFFSET):
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
    
    return polygon, polygon


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


def is_point_contained(point, xmax, ymax):
    '''TODO'''
    
    p_x = point[0]
    p_y = point[1]
    cond_1 = 0 <= p_x and p_x <= xmax
    cond_2 = 0 <= p_y and p_y <= ymax
    return cond_1 and cond_2