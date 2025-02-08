import numpy as np
import trimesh

def stl_to_voxels(stl_file, voxel_size):
    # Load the STL file using trimesh
    mesh_data = trimesh.load_mesh(stl_file)
    
    # Define the bounding box of the mesh
    min_bound = mesh_data.bounds[0]
    max_bound = mesh_data.bounds[1]
    
    # Create a grid of points (voxels) within the bounding box
    x_range = np.arange(min_bound[0], max_bound[0], voxel_size)
    y_range = np.arange(min_bound[1], max_bound[1], voxel_size)
    z_range = np.arange(min_bound[2], max_bound[2], voxel_size)
    
    # Initialize an empty voxel grid
    voxel_grid = np.zeros((len(x_range), len(y_range), len(z_range)), dtype=bool)
    
    # Iterate over each voxel and check if it intersects with the mesh
    for i, x in enumerate(x_range):
        for j, y in enumerate(y_range):
            for k, z in enumerate(z_range):
                # Create a small cube at the voxel center
                voxel_center = [x + voxel_size / 2, y + voxel_size / 2, z + voxel_size / 2]
                
                # Check if the voxel center is inside the mesh
                if mesh_data.contains(voxel_center):
                    voxel_grid[i, j, k] = 1
    
    return voxel_grid

# Example usage:
stl_file = 'stls/BH_10x5_37.stl'
voxel_size = 0.1  # Adjust voxel size
voxel_grid = stl_to_voxels(stl_file, voxel_size)

print("Voxel grid shape:", voxel_grid.shape)
