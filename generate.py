import torch
import math

# Set the random seed for reproducibility
torch.manual_seed(42)

# Generate 100 points in the range [0, 1]
train_x = torch.linspace(0, 1, 100)

# True function: sin(2*pi*x) with Gaussian noise
# Adding Gaussian noise with mean 0 and standard deviation sqrt(0.04)
noise_std_dev = math.sqrt(0.04)
train_y = torch.sin(train_x * (2 * math.pi)) + torch.randn(train_x.size()) * noise_std_dev

# Display the generated data points
print("Generated data points (train_x, train_y):")
for x, y in zip(train_x[:5], train_y[:5]):  # Displaying the first 5 points as a sample
    print(f"x: {x:.3f}, y: {y:.3f}")
