import trimesh
import numpy as np
import torch
import os

# Directory containing STL files
directory = "stls"

# List to store voxelized tensors
tensor_list = []

count = 1

for i in os.listdir(directory):
    if i.endswith(".stl"):
        print(f"Processing {count}: {i}")
        count += 1

        # Load STL mesh
        file_path = os.path.join(directory, i)
        mesh = trimesh.load_mesh(file_path)

        # Convert mesh to voxel grid
        voxelized = trimesh.voxel.creation.voxelize(mesh, pitch=1)

        voxel_array = voxelized.matrix.astype(np.float32)  # Ensure float format

        # Check and pad/resize if necessary
        target_size = (32, 32, 32)  # Adjust based on your model
        current_shape = voxel_array.shape

        if current_shape != target_size:
            padded_array = np.zeros(target_size, dtype=np.float32)
            min_shape = tuple(min(s, t) for s, t in zip(current_shape, target_size))
            padded_array[:min_shape[0], :min_shape[1], :min_shape[2]] = voxel_array[:min_shape[0], :min_shape[1], :min_shape[2]]
            voxel_array = padded_array

        # Convert to PyTorch tensor
        tensor = torch.tensor(voxel_array).unsqueeze(0)  # Add channel dimension (1, 32, 32, 32)
        tensor_list.append(tensor)

# Convert list to a single tensor batch (N, 1, 32, 32, 32)
voxel_data = torch.stack(tensor_list)
print(f"Final tensor shape: {voxel_data.shape}")  # Expected: (num_samples, 1, 32, 32, 32)

import torch.nn as nn

class Generator3D(nn.Module):
    def __init__(self, latent_dim=200):
        super(Generator3D, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 32 * 32 * 32),  # Output voxel grid size
            nn.Tanh()  # Normalize output between -1 and 1
        )

    def forward(self, z):
        x = self.model(z)
        return x.view(-1, 1, 32, 32, 32)  # Reshape to 3D volume


class Discriminator3D(nn.Module):
    def __init__(self):
        super(Discriminator3D, self).__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 32 * 32, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1),
            nn.Sigmoid()  # Output probability of real/fake
        )

    def forward(self, x):
        return self.model(x)


import torch.optim as optim

# Set up models
latent_dim = 200
generator = Generator3D(latent_dim)
discriminator = Discriminator3D()

# Set up optimizers
optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

# Loss function
criterion = nn.BCELoss()

# Training loop
epochs = 1000
batch_size = 16

for epoch in range(epochs):
    # Sample random noise
    z = torch.randn(batch_size, latent_dim)
    fake_data = generator(z)
    
    # Get real samples from dataset
    real_data = voxel_data[:batch_size]  # Assuming enough data exists

    # Train Discriminator
    optimizer_D.zero_grad()
    real_labels = torch.ones(batch_size, 1)
    fake_labels = torch.zeros(batch_size, 1)

    real_loss = criterion(discriminator(real_data), real_labels)
    fake_loss = criterion(discriminator(fake_data.detach()), fake_labels)
    d_loss = real_loss + fake_loss
    d_loss.backward()
    optimizer_D.step()

    # Train Generator
    optimizer_G.zero_grad()
    g_loss = criterion(discriminator(fake_data), real_labels)  # Want to fool D
    g_loss.backward()
    optimizer_G.step()

    if epoch % 100 == 0:
        print(f"Epoch {epoch}: D Loss: {d_loss.item()}, G Loss: {g_loss.item()}")

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def visualize_voxels(voxel_data):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    voxel_data = voxel_data.squeeze().detach().cpu().numpy()
    ax.voxels(voxel_data > 0.5, edgecolor='k')
    plt.show()

# Generate a sample
z = torch.randn(1, latent_dim)
generated_voxel = generator(z)
visualize_voxels(generated_voxel)

