import torch
import torch.nn as nn
import torch.optim as optim

# Step 1: Define the dataset
# Toy dataset for demonstration
x_train = torch.tensor([[1.0], [2.0], [3.0], [4.0]], dtype=torch.float32)
y_train = torch.tensor([[2.0], [4.0], [6.0], [8.0]], dtype=torch.float32)

# Step 2: Define the model
class LinearRegression(nn.Module):
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(1, 1)  # 1 input feature, 1 output

    def forward(self, x):
        return self.linear(x)

model = LinearRegression()

# Step 3: Define loss function and optimizer
criterion = nn.MSELoss()  # Mean Squared Error loss
optimizer = optim.SGD(model.parameters(), lr=0.01)  # Stochastic Gradient Descent

# Step 4: Train the model
num_epochs = 100
for epoch in range(num_epochs):
    # Forward pass
    y_pred = model(x_train)

    # Compute loss
    loss = criterion(y_pred, y_train)

    # Backpropagation and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Print progress
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# Step 5: Evaluate the model
model.eval()
with torch.no_grad():
    y_pred = model(x_train)
    print(f'Predictions: {y_pred}')

# Step 6: Save the model
torch.save(model.state_dict(), 'linear_model.pth')
