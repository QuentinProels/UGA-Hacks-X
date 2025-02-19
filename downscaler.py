import os
import scipy.ndimage
import numpy as np
import cv2
#not scaling one of the dimensions right now I presume to be the z axis
def resize_voxel_cv(voxel_array, scale_factor):

    voxel_array = voxel_array.astype(np.float32)
    # Rescale each dimension separately with cubic interpolation
    new_shape = (int(voxel_array.shape[0] * scale_factor), 
                 int(voxel_array.shape[1] * scale_factor),
                 int(voxel_array.shape[2] * scale_factor))

    # Resize each slice (z-axis) individually
    resized_voxel = np.stack([
        cv2.resize(voxel_array[z], (new_shape[1], new_shape[2]), interpolation=cv2.INTER_CUBIC)
        for z in range(voxel_array.shape[0])
    ], axis=0)
    
    return resized_voxel

def resize_voxel(directory, output_size):
    size = output_size
    real_max_dimension = 0
    for i in os.listdir(directory):
        if i.endswith(".npy"):

            file_path = os.path.join(directory, i)

            # Convert mesh to voxel grid

            voxel_array = np.load(file_path) 
          
            max_dim = max(voxel_array.shape) 

            if (max_dim > real_max_dimension):
                real_max_dimension = max_dim
            
            #print(f"File: {i}, Shape: { max_dim}")
    print (real_max_dimension)
    scale_factor = (output_size/real_max_dimension)
    print (scale_factor)

    for i in os.listdir(directory):
        if i.endswith(".npy"):
            output_directory = f"rescaled_{directory}"
            os.makedirs(output_directory, exist_ok=True)
            file_path = os.path.join(directory, i)
            voxel_array = np.load(file_path) 
            #resized_voxel_array = scipy.ndimage.zoom(voxel_array, (scale_factor, scale_factor, scale_factor), order = 5)
            #resized_voxel_array = np.clip(resized_voxel_array, 0, 1)

            resized_voxel_array = resize_voxel_cv(voxel_array, scale_factor)



            output_path = os.path.join(f"rescaled_{directory}", i)
            np.save(output_path, resized_voxel_array)

resize_voxel("dataset", 32)

