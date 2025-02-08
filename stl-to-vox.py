import trimesh
import numpy as np
import torch
import os

tensor_list = []

directory = "stls/"

for i in os.listdir(directory):
    if i.endswith(".stl"):
        file_path = os.path.join(directory, i)
        mesh = trimesh.load_mesh(file_path)
        voxelized = trimesh.voxel.creation.voxelize(mesh, pitch=0.5)
        voxel_array = voxelized.matrix
        tensor = torch.from_numpy(voxel_array)
        tensor_list.append(tensor)
           
    