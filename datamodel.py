import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

# Generate fake data
torch.manual_seed(42)
X_linear = torch.randn(100, 1)  # 100 samples, 1 feature
y_linear = 2 * X_linear + 3 + torch.randn(100, 1) * 0.1  # y = 2x + 3 + noise

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_linear, y_linear, test_size=0.2, random_state=42)

# Define the model
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)

# Instantiate the model
model_linear = LinearRegressionModel()

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(model_linear.parameters(), lr=0.01)

# Training the model
num_epochs = 100
for epoch in range(num_epochs):
    # Forward pass
    outputs = model_linear(X_train)
    loss = criterion(outputs, y_train)
    
    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Evaluate the model
with torch.no_grad():
    model_linear.eval()
    predicted = model_linear(X_test)
    test_loss = criterion(predicted, y_test)
    print(f'Test Loss: {test_loss.item():.4f}')
