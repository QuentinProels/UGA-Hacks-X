import numpy as np 
import trimesh

def visualize_the_voxel_goddamnit(voxel_grid):
    vox_mesh = trimesh.voxel.VoxelGrid(voxel_grid)
    mesh = vox_mesh.as_boxes()
    mesh.show()

voxel_grid = np.load("rescaled_dataset/voxel_1.npy")
visualize_the_voxel_goddamnit(voxel_grid)
#loads the voxel mesh for visualization
