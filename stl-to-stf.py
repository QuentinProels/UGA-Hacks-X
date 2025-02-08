import trimesh
import numpy as np
from scipy.spatial import cKDTree
import skimage.measure

def compute_sdf(mesh, grid_size=64):
    # Ensure the mesh is watertight (no holes)
    if not mesh.is_watertight:
        print("Warning: Mesh is not watertight! This may cause incorrect SDF.")

    # Get bounding box
    bounds = mesh.bounds
    x_min, y_min, z_min = bounds[0]
    x_max, y_max, z_max = bounds[1]

    # Create a 3D grid of points
    x = np.linspace(x_min, x_max, grid_size)
    y = np.linspace(y_min, y_max, grid_size)
    z = np.linspace(z_min, z_max, grid_size)
    grid_points = np.array(np.meshgrid(x, y, z)).T.reshape(-1, 3)

    # Compute **signed** distances
    sdf_values = trimesh.proximity.signed_distance(mesh, grid_points)

    # Reshape into 3D grid
    sdf_values = sdf_values.reshape((grid_size, grid_size, grid_size))

    return sdf_values

# Load STL and compute SDF
mesh = trimesh.load_mesh("stls/BH_10x5_37.stl")
mesh.apply_translation(-mesh.centroid) 
sdf_grid = compute_sdf(mesh, grid_size=64)

# Save the SDF for future use
np.save("stls/sdf.npy", sdf_grid)

print("SDF computed and saved as 'sdf.npy'.")
print("Negative values in SDF:", np.any(sdf_grid < 0))


# Load the saved SDF
sdf_grid = np.load("stls/sdf.npy")

# Apply Marching Cubes to extract the surface
verts, faces, _, _ = skimage.measure.marching_cubes(sdf_grid, level=0)

# Create a mesh from the extracted surface
mesh = trimesh.Trimesh(vertices=verts, faces=faces)

# Export the reconstructed mesh as STL
mesh.export("stls/reconstructed.stl")

print("Reconstructed STL saved as 'reconstructed.stl'")