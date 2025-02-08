import numpy as np 
import trimesh

voxel_grid = np.load("dataset/voxel_1.npy")

vox_mesh = trimesh.voxel.VoxelGrid(voxel_grid)
mesh = vox_mesh.as_boxes()
mesh.show()