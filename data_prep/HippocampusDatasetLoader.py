"""
Module loads the hippocampus dataset into RAM
"""
import os
from os import listdir
from os.path import isfile, join

import numpy as np
from medpy.io import load

from utils.utils import med_reshape

def LoadHippocampusData(root_dir, y_shape, z_shape):
    '''
    This function loads our dataset form disk into memory,
    reshaping output to common size

    Arguments:
        volume {Numpy array} -- 3D array representing the volume

    Returns:
        Array of dictionaries with data stored in seg and image fields as 
        Numpy arrays of shape [AXIAL_WIDTH, Y_SHAPE, Z_SHAPE]
    '''

    image_dir = os.path.join(root_dir, 'images')
    label_dir = os.path.join(root_dir, 'labels')

    images = [f for f in listdir(image_dir) if (
        isfile(join(image_dir, f)) and f[0] != ".")]

    out = []
    for f in images:

        # We would benefit from mmap load method here if dataset doesn't fit into memory
        # Images are loaded here using MedPy's load method. We will ignore header 
        # since we will not use it
        # from scipy.ndimage import zoom
        image, _ = load(os.path.join(image_dir, f)) # 3D descriptive features
        label, _ = load(os.path.join(label_dir, f)) # 3D labels
        # print(np.unique(label))    
        # TASK: normalize all images (but not labels) so that values are in [0..1] range
        # <YOUR CODE GOES HERE>
        # image = np.clip(image, 0., 1.)
        image = image.astype(np.single)/np.max(image)

        #image = zoom(image, (1., .9, .9))
        #label = zoom(label, (1., .9, .9))
        
        #print("check image max and min...")
        #print(np.max(image), np.min(image))
        
        # We need to reshape data since CNN tensors that represent minibatches
        # in our case will be stacks of slices and stacks need to be of the same size.
        # In the inference pathway we will need to crop the output to that
        # of the input image.
        # Note that since we feed individual slices to the CNN, we only need to 
        # extend 2 dimensions out of 3. We choose to extend coronal and sagittal here
        
        ################################################# My Comments ###############################################
        # why don't you just say we need to keep each image with the same size of coronal and sagittal dimensions   #
        # because the raw data has different sizes of coronal and sagittal dimensions                               #
        ################################################# My Guess ##################################################
        # My guess is the images and labels should be converted into (num_of_2D_images, 64, 64)                     #                                                                                            #
        #############################################################################################################
        
        # TASK: med_reshape function is not complete. Go and fix it!
        new_shape=(image.shape[0], y_shape, z_shape) # x_shape = axial, y_shape = coronal, z_shape = sagittal
        image = med_reshape(image, new_shape=new_shape)
        label = med_reshape(label, new_shape=new_shape).astype(int)

        # TASK: Why do we need to cast label to int?
        # ANSWER: Label to be used as mask in the project is for classification purposes.
        # For example, we have 3 classes in this project "background", "anterior", and "posterior".
        # Also NumPy is better at handling numeric data, it makes more sense to use integers to represent classes.

        # "labels": { 
        #   "0": "background", 
        #   "1": "Anterior", 
        #   "2": "Posterior"
        # }, 

        out.append({"image": image, "seg": label, "filename": f})

    # Hippocampus dataset only takes about 300 Mb RAM, so we can afford to keep it all in RAM
    print(f"Processed {len(out)} files, total {sum([x['image'].shape[0] for x in out])} image slices, total {sum([x['seg'].shape[0] for x in out])} mask slices")
    
    return np.array(out)