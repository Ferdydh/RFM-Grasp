"""TODO: Not tested at all, just a rough draft based on https://elib.dlr.de/197492/1/winkelbauer23_copyright.pdf"""

import torch
import torch.nn as nn
from einops import rearrange


class VoxelSDFEncoder(nn.Module):
    def __init__(self, input_size=32,output_size=512):
        super(VoxelSDFEncoder, self).__init__()

        # Define the convolutional encoder
        self.encoder = nn.Sequential(
            # First 3D Conv Block
            nn.Conv3d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),
            # Second 3D Conv Block
            nn.Conv3d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),
            # Third 3D Conv Block
            nn.Conv3d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),
            # Fourth 3D Conv Block
            nn.Conv3d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),
        )

        # Calculate flattened size: after 4 MaxPool layers, spatial dims are reduced by 16
        # Final shape will be (batch_size, 256, 3, 3, 3) #average the last 3 dimensions.
        #final_spatial_size = input_size // 16  # 48/16 = 3
        #Maybe average pooling, 256 in CNN instead of this 6912*512 layer -> so it will be 256*512
        flatten_size = 256 # (final_spatial_size**3)  # 256 * 3 * 3 * 3 = 6912

        self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        # Linear layer to get to desired output size of 512
        self.fc = nn.Linear(flatten_size, 512)

    def forward(self, x):
        # Input shape: (batch_size, 1, 48, 48, 48)
        x = self.encoder(x)
        x = self.avg_pool(x)
        # Flatten using einops: (batch, channels, depth, height, width) -> (batch, channels * depth * height * width)
        x = rearrange(x, "b c d h w -> b (c d h w)")
        # Project to 512 dimensions
        x = self.fc(x)
        return x


# Example usage:
if __name__ == "__main__":
    # Create a sample input tensor
    batch_size = 4
    input_tensor = torch.randn(batch_size, 1, 48, 48, 48)

    # Initialize the model
    model = VoxelSDFEncoder()

    # Forward pass
    output = model(input_tensor)

    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape: {output.shape}")  # Should be (batch_size, 512)
