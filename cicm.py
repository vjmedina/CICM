"""
Implementation of the cross-image co-occurrence matrix (CICM) in python, 
which computes a version of the common gray-level co-ocurrence matrix (GLCM) 
between different images, or channels of the same image.

Author: Victor Medina-Heierle
Date: 2024/11/11
"""

import numpy as np
import math
import time

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

def agg_duplicates(arr:np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Finds duplicated columns in the input array, adds them,
    and returns the same input array without the duplicated 
    columns and another array with the total count of each aggregated column.

    Args:
        arr (np.ndarray): the input array, with the value combinations that needs to be examined arranged in columns.

    Returns:
        np.ndarray: the input array without duplicated columns
        np.ndarray: the total count of each aggregated column.
    """
    
    data = np.ones_like(arr[0,:])
    arr2 = arr.T
    
    order = np.lexsort(arr2.T)
    diff = np.diff(arr2[order], axis=0)
    uniq_mask = np.append(True, (diff != 0).any(axis=1))

    uniq_inds = order[uniq_mask]
    inv_idx = np.zeros_like(order)
    inv_idx[order] = np.cumsum(uniq_mask) - 1
    
    data = np.bincount(inv_idx, weights=data)
    
    return arr2[uniq_inds].T, data

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
    
    cicm_array = np.zeros((levels, levels, len(distances), len(radian_angles)))
    
    rows, cols = src_img.shape[:2]
    
    for idx_d, D in enumerate(distances):
        
        for idx_a, A in enumerate(radian_angles):
            
            src = src_img
            
            # For consistency with the results from scikit's graycomatrix method,
            # here we transpose the angles 180 degrees counter-clockwise, so that
            # 0 degrees moves straight backwards on the horizontal, 
            # 45 moves backwards up on the diagonal, 
            # 90 moves straight upwards on the vertical,
            # and 135 moves forward up on the diagonal.
            
            corrected_angle = math.radians(180) - A
            
            step_row = math.ceil(D * round(math.sin(-corrected_angle)))
            step_col = math.ceil(D * round(math.cos(corrected_angle)))
            
            # Define dst as the input image after shifting
            # each pixel D positions in the direction of angle A.
            dst = shift_array(dst_image, (-step_row, -step_col), padding=-1)
                                    
            # Ravel matrices to simplify calculations
            src_rav = src.ravel()
            dst_rav = dst.ravel()
                        
            # Define mask to ignore negative entries, 
            # as they are used for padding during
            # shifting, and should not be confused with
            # numpy's negative indexing.
            mask = (dst_rav >= 0) 
            
            # Convert arrays to indices
            rows = src_rav[mask].T
            cols = dst_rav[mask].T
            
            # Stack indices to operate on 
            # one single array
            data = np.stack((rows, cols))
            
            # Find repeated row-column combinations
            # and add one
            agg_data, counts = agg_duplicates(data)
            rows, cols = agg_data
            
            # Add the number of matching pixels to the array
            # for the current distance and angle.
            cicm_array[rows, cols, idx_d, idx_a] += counts
    
    if (sum_angles):
        # Sum all the angles for each distance
        # keeping distance arrays separate.
        cicm_array = np.sum(cicm_array, axis=3)
      
    return cicm_array
