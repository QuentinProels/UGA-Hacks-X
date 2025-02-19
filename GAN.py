import trimesh
import numpy as np
import torch
import os
from vox2stl import exporter
from downscaler import resize_voxel
from vox_visualizer import visualize_the_voxel_goddamnit

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
#function to visualize the voxels
def visualize_voxels(voxel_data):
    voxel_data = (voxel_data - voxel_data.min()) / (voxel_data.max() - voxel_data.min()) * 2 - 1
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    voxel_data = voxel_data.squeeze().detach().cpu().numpy()
    ax.voxels(voxel_data > 0.5, edgecolor='k')
    
    exporter(voxel_data)
    
    plt.show()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print(f"GPU available: {torch.cuda.is_available()}")
print(f"Device name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
# Directory containing STL files
directoryOne = "dataset"
size = 32
resize_voxel(directoryOne, size)
directory = f"rescaled_{directoryOne}"

# List to store voxelized tensors
tensor_list = []

count = 1

for i in os.listdir(directory):
    if i.endswith(".npy"):
        print(f"Processing {count}: {i}")
        count += 1

        # Load STL mesh
        file_path = os.path.join(directory, i)

        # Convert mesh to voxel grid

        voxel_array = np.load(file_path).astype(np.float32)  # Ensure float32 format
        voxel_array = (voxel_array * 2) - 1  # Normalize from [0,1] to [-1,1]


        # Check and pad/resize if necessary
        target_size = (size, size, size)  # Adjust based on your model
        current_shape = voxel_array.shape

        if current_shape != target_size:
            padded_array = np.zeros(target_size, dtype=np.float32)
            min_shape = tuple(min(s, t) for s, t in zip(current_shape, target_size))
            padded_array[:min_shape[0], :min_shape[1], :min_shape[2]] = voxel_array[:min_shape[0], :min_shape[1], :min_shape[2]]
            voxel_array = padded_array


        # Convert to PyTorch tensor
        print("about to visualize the voxel")
        #visualize_the_voxel_goddamnit(voxel_array)
        tensor = torch.tensor(voxel_array).unsqueeze(0)  # Add channel dimension (1, 32, 32, 32)
        tensor_list.append(tensor)
        print("about to visualize the tensor")
        #visualize_voxels(tensor)

# Convert list to a single tensor batch (N, 1, 32, 32, 32)
voxel_data = torch.stack(tensor_list).to(device)
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
            nn.Linear(1024, size * size * size),  # Output voxel grid size
            nn.Tanh()  # Normalize output between -1 and 1
        )

    def forward(self, z):
        x = self.model(z)
        return x.view(-1, 1, size, size, size)  # Reshape to 3D volume


class Discriminator3D(nn.Module):
    def __init__(self):
        super(Discriminator3D, self).__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(size * size * size, 1024),
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
generator.to(device)
discriminator = Discriminator3D()
discriminator.to(device)
tensor = torch.randn(3,3).to(device)

# Set up optimizers
optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0001, betas=(0.5, 0.999))

# Loss function
criterion = nn.BCELoss()

# Training loop
epochs = 700
batch_size = 8

# Directory for saving models
save_dir = "checkpoints"
os.makedirs(save_dir, exist_ok=True)



# Function to save the model state
def save_checkpoint(epoch, generator, discriminator, optimizer_G, optimizer_D):
    checkpoint_path = f"{save_dir}/gan_checkpoint_{epoch}.pth"
    checkpoint = {
        'epoch': epoch,
        'generator_state_dict': generator.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
        'optimizer_G_state_dict': optimizer_G.state_dict(),
        'optimizer_D_state_dict': optimizer_D.state_dict(),
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved at {checkpoint_path}")


# Function to load the latest checkpoint
def load_checkpoint(generator, discriminator, optimizer_G, optimizer_D):
    checkpoint_files = [f for f in os.listdir(save_dir) if f.startswith("gan_checkpoint_") and f.endswith(".pth")]

    if not checkpoint_files:
        print("🛠 No checkpoint found, starting from scratch.")
        return 0  # Start training from epoch 0

    # Find the latest checkpoint based on epoch number
    latest_checkpoint = max(checkpoint_files, key=lambda x: int(x.split("_")[-1].split(".")[0]))
    latest_epoch = int(latest_checkpoint.split("_")[-1].split(".")[0])

    checkpoint_path = os.path.join(save_dir, latest_checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location=device)

    generator.load_state_dict(checkpoint['generator_state_dict'])
    discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
    optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
    optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])

    print(f"Resumed training from checkpoint: {checkpoint_path} (Epoch {latest_epoch + 1})")
    return latest_epoch + 1  # Resume from the next epoch


# Example usage before training starts
checkpoint_path = "checkpoints/gan_checkpoint_900.pth"  # Change to the latest checkpoint
start_epoch = load_checkpoint(generator, discriminator, optimizer_G, optimizer_D)


# If final checkpoint exists, skip training
if start_epoch >= epochs:
    print("Training is already complete. Skipping training...")
else:
    print(f"Starting training from epoch {start_epoch}...")

    # Training loop
    for epoch in range(start_epoch, epochs):
        # Sample random noise
        z = torch.randn(batch_size, latent_dim).to(device)
        fake_data = generator(z)
        
        # Get real samples from dataset
        real_data = voxel_data[:batch_size]
        real_data = real_data.to(device)  # Ensure enough data exists

        # Train Discriminator   
        optimizer_D.zero_grad()
        real_labels = torch.ones(batch_size, 1).to(device)
        fake_labels = torch.zeros(batch_size, 1).to(device)

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
            z = torch.randn(1, latent_dim).to(device)
            generated_voxel = generator(z)
            visualize_voxels(generated_voxel)

        if epoch % 50 == 0:
            print(f"Epoch {epoch}: D Loss: {d_loss.item()}, G Loss: {g_loss.item()}")
            save_checkpoint(epoch, generator, discriminator, optimizer_G, optimizer_D)
    # Final checkpoint after completion
    save_checkpoint(epochs - 1, generator, discriminator, optimizer_G, optimizer_D)
    print("Training completed and final model saved!")




# Generate a sample
z = torch.randn(1, latent_dim).to(device)
generated_voxel = generator(z)
visualize_voxels(generated_voxel)

