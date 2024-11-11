"""
Implementation of the cross-image co-occurrence matrix (CICM) in python, 
which computes a version of the common gray-level co-ocurrence matrix (GLCM) 
between different images, or channels of the same image.

Author: Victor Medina-Heierle
Date: 2024/11/11
"""

import numpy as np
import math

from typing import List, Union, Tuple, Any

def shift_array(array: np.ndarray, step: Union[int,Tuple,List], padding: Any = -1) -> np.ndarray:
    """Shifts an array a given number of rows and columns, padding the remaining
    elements with the provided value.

    Args:
        array (np.ndarray): The input array.
        step (Union[int,Tuple,List]): The number of rows and columns to shift the input array, with negative values indicating backward shifting.
                                      If only an integer value is provided, the same value will be used for rows and columns.
        padding (Any, optional): Any valid nupy array value to be used as padding after shifting the input array. Defaults to -1.

    Returns:
        np.ndarray: The shifted and padded array.
        
    Notes:
        This method implements the same operation as scipy.ndimage's shift method. 
        It's been implemented for two reasons:
            1. To reduce package dependency 
            2. To fix a bug found during testing where the added padding is always 0 
               (ignoring the value of the cval parameter) when the input array is generated randomly.
    """
     
    if (type(step) == int):
        step = (step, step)
            
    step_row = step[0]
    step_col = step[1]
    
    # Declare an array where all cells contain
    # the padding element.
    new_array = np.ones_like(array)*padding
    rows, cols = array.shape[0:2]
    
    # We cannot use 0 as a slice limit, 
    # so we replace zero indices with the
    # actual array size to be used for slicing.
    step_row2 = -rows if (step_row==0) else step_row
    step_col2 = -cols if (step_col==0) else step_col    
        
    # Check each combination of values for rows and columns.
    if (step_row >= 0 and step_col >= 0):
        new_array[step_row:, step_col:] = array[:-step_row2,:-step_col2]
    if (step_row >= 0 and step_col < 0):
        new_array[step_row:, :step_col] = array[:-step_row2:, -step_col2:]
    if (step_row < 0 and step_col >= 0):
        new_array[:-step_row+1, step_col:] = array[-step_row2:,:-step_col2]
    if (step_row < 0 and step_col < 0):
        new_array[:-step_row+1, :-step_col+1] = array[-step_row2:,-step_col2:]
        
    return new_array


def cicm(src_img: np.ndarray, dst_image: np.ndarray, distances: List[int], radian_angles: List[float], levels: int, sum_angles: bool=False) -> np.ndarray:
    """ Cross-image co-occurrence matrix (CICM), a version of the gray-level co-ocurrence matrix (GLCM) 
    that allows finding pixel co-occurrences between different images, or channels of the same image.

    Args:
        src_img (np.ndarray): The reference image.
        dst_image (np.ndarray): The destination image to check for pixel co-occurrence.
        distances (List[int]): The list of pixel distances to consider.
        radian_angles (List[float]): The list of pixel angles to consider, given in radians.
        levels (int): The total number of gray level values that can occur in the image.
        sum_angles (bool, optional): When set to True, the result component for each distance combines the sum of all the pixel counts 
                                     for each angle into one array; If set to False, the counts for each individual angle are kept in 
                                     separate arrays. Defaults to False.

    Returns:
        np.ndarray: The result matrix, of size [levels x levels x len(distances) x len(radian_angles)], which contains a component
                    for each distance and angle combination, with each component being a [levels x levels] matrix where each cell [i,j]
                    stores the total number of times where gray-level i in src_img occurs at distance D and angle A from gray-level j 
                    in dst_img.
    """
    
    cicm_array = np.zeros((levels, levels, len(distances), len(radian_angles)), dtype = np.uint8)
    
    for idx_d, D in enumerate(distances):
        for idx_a, A in enumerate(radian_angles):
            
            src = src_img
            
            # For consistency with the results from scikit's graycomatrix method,
            # here we transpose the angles 180 degrees anti-clockwise, so that
            # 0 degrees moves straight backwards on the horizontal, 
            # 45 moves backwards up on the diagonal, 
            # 90 moves straight upwards on the vertical,
            # and 135 moves forward up on the diagonal.
            
            corrected_angle = math.radians(180)-A
            
            step_row = math.ceil(D * round(math.sin(-corrected_angle)))
            step_col = math.ceil(D * round(math.cos(corrected_angle)))
            
            # Define dst as the input image after shifting
            # each pixel D positions in the direction of angle A.
            dst = shift_array(dst_image, (-step_row, -step_col), padding=-1)
            
            # Loop over each gray-level value
            for l1 in range(levels):
                
                # Define a logical array where True indicates
                # that a pixel contains gray level l1 in src, 
                # and False indicates that it doesn't.
                src_bin = (src==l1)
                
                for l2 in range(levels):
                    # Define a logical array where True indicates
                    # that a pixel contains gray level l2 in dst, 
                    # and False indicates that it doesn't.
                    dst_bin = (dst==l2)
                    
                    # Define a new logical array where True indicates
                    # that a cell is True in both src_bin and dst_bin,
                    # whereas False indicates that the cell is not True
                    # in at least one of the arrays.
                    matches = src_bin & dst_bin
                    
                    # Get total number of matching (True) pixels.
                    count = matches.sum()
                    
                    # Add the number of matching pixels to the array
                    # for the current distance and angle.
                    cicm_array[l2,l1,idx_d,idx_a] += count
    
    if (sum_angles):
        # Sum all the angles for each distance
        # keeping distance arrays separate.
        cicm_array = np.sum(cicm_array, axis=3)
      
    return cicm_array

