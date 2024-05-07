import torch

# Generate fake data
torch.manual_seed(42)
X_linear = torch.randn(100, 1)  # 100 samples, 1 feature
y_linear = 2 * X_linear + 3 + torch.randn(100, 1) * 0.1  # y = 2x + 3 + noise

# Define the model
class LinearRegressionModel(torch.nn.Module):
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        self.linear = torch.nn.Linear(1, 1)  # Input size: 1, Output size: 1

    def forward(self, x):
        return self.linear(x)

# Instantiate the model
model_linear = LinearRegressionModel()
