import trimesh
import numpy as np
import torch
import os


directory = "stls"
count = 1

for i in os.listdir(directory):
    if i.endswith(".stl"):
        print(f"ran {count}")
        file_path = os.path.join(directory, i)
        mesh = trimesh.load_mesh(file_path)
        voxelized = trimesh.voxel.creation.voxelize(mesh, pitch=1)
        voxel_array = voxelized.matrix
        np.save(f"dataset/voxel_{count}.npy", voxel_array)
        count = count+1        


           
    