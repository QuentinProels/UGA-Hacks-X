import os
import scipy.ndimage
import numpy as np



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
    scale_factor = output_size/real_max_dimension

    for i in os.listdir(directory):
        if i.endswith(".npy"):
            output_directory = f"rescaled_{directory}"
            os.makedirs(output_directory, exist_ok=True)
            file_path = os.path.join(directory, i)
            voxel_array = np.load(file_path) 
            resized_voxel_array = scipy.ndimage.zoom(voxel_array, scale_factor, order = 1)
            output_path = os.path.join(f"rescaled_{directory}", i)
            np.save(output_path, resized_voxel_array)
             

            




resize_voxel("dataset", 32)
resize_voxel("rescaled_dataset", 16)