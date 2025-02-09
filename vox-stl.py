import numpy as np
import trimesh
from skimage.measure import marching_cubes
import pyvista
#loads the mesh
voxel_grid = np.load("dataset/voxel_36.npy")
#applies marching cubes algoithm to the mesh
vertices, faces, _, _ = marching_cubes(voxel_grid, level=0.5)

mesh = trimesh.Trimesh(vertices=vertices,faces=faces)

pv_mesh = pyvista.wrap(mesh)
#smooths and outputs mesh to a test file
smooth = pv_mesh.smooth(n_iter=100, relaxation_factor=0.1)

smooth.save("outputs/outputTest.stl")