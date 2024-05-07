import torch
import math

# Generate 100 points in the range [0, 1]
train_x = torch.linspace(0, 1, 100)

# True function: sin(2*pi*x) with Gaussian noise
# Adding Gaussian noise with mean 0 and standard deviation sqrt(0.04)
train_y = torch.sin(train_x * (2 * math.pi)) + torch.randn(train_x.size()) * math.sqrt(0.04)
