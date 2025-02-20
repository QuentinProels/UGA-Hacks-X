import torch
import torch.nn as nn

class MiniBatchLayer(nn.Module):
    def __init__(self, input_dim, n_kernels=20, kernel_dim=5):
        super(MiniBatchLayer, self).__init__()
        self.weights = nn.Parameter(torch.randn(input_dim, n_kernels * kernel_dim))

    # self is model and x is tensor
    def forward(self, x):  

        print(f"x.shape before unpacking: {x.shape}")
        batch_size, channels, depth, height, width = x.size()
        # converts each voxel grid into a 1 dimensional vector
        x = x.view(batch_size, -1)
        # condenses the features of the entire batch into one matrix and multiplies it by their corresponding weights
        features = torch.matmul(x, self.weights)
        # turns the feature representation back into tensor
        features = features.view(batch_size, -1, 5)
        # finds the pairwise difference of the tensors with an extra dimension in index 0 for the first and index 1 for the second
        difference = features.unsqueeze(0) - features.unsqueeze(1)
        # gets the sum across each kernel of the absolute values of the differences
        distance = torch.sum(torch.abs(difference), dim = 2)
        # returns a tensor across the kernels that shows how dissimiliar each sample in the batch is from the other
        return torch.sum(torch.exp(-distance), dim=2)