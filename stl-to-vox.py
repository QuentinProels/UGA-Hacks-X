import trimesh
import numpy as np
import torch
import os
from scipy.ndimage import binary_dilation, binary_erosion 
from scipy.spatial import ConvexHull, Delaunay



def fill_holes(voxel_array):
    # Get the coordinates of the non-zero points (points in the voxel grid)
    points = np.array(np.nonzero(voxel_array)).T
    
    # Create the convex hull of the points
    hull = ConvexHull(points)
    
    # Create a Delaunay triangulation from the convex hull
    delaunay = Delaunay(points[hull.vertices])  # Delaunay over the convex hull vertices
    
    # Initialize an empty voxel grid to fill
    filled_voxel_array = np.zeros_like(voxel_array)

    # Iterate over every voxel in the grid
    for i in range(voxel_array.shape[0]):
        for j in range(voxel_array.shape[1]):
            for k in range(voxel_array.shape[2]):
                # Check if the point (i, j, k) is inside the convex hull
                if delaunay.find_simplex([i, j, k]) >= 0:
                    filled_voxel_array[i, j, k] = 1
    
    return filled_voxel_array

directory = "stls"
count = 1

for i in os.listdir(directory):
    if i.endswith(".stl"):
        print(f"ran {count}")
        file_path = os.path.join(directory, i)
        mesh = trimesh.load_mesh(file_path)
        voxelized = trimesh.voxel.creation.voxelize(mesh, pitch=1)
        voxel_array = voxelized.matrix
        voxel_array = fill_holes(voxel_array)
        np.save(f"dataset/voxel_{count}.npy", voxel_array)
        count = count+1        


           
    