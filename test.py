import numpy as np
import trimesh
from stl import mesh
from perlin_noise import PerlinNoise

# Initialize Perlin noise generator
noise_gen = PerlinNoise(octaves=1, seed=42)

# Generate a base shape (e.g., an icosphere)
base_mesh = trimesh.creation.icosphere(subdivisions=5, radius=50)


# Apply Perlin noise to vertices for organic deformation
for i, vertex in enumerate(base_mesh.vertices):
    noise_value = noise_gen([vertex[0] * 0.05, vertex[1] * 0.05, vertex[2] * 0.05])  # Scale for smoother variation
    base_mesh.vertices[i] += noise_value * 10  # Adjust strength of noise


# Export as STL
base_mesh.export("C:/Users/quent/perlin_hold4.stl")

print("Climbing hold STL file generated: perlin_hold.stl")
