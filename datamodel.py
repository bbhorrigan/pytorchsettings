import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

# Set random seed for reproducibility
torch.manual_seed(42)

# Generate synthetic linear data
X = torch.randn(100, 1)  # 100 samples, 1 feature
y = 2 * X + 3 + torch.randn(100, 1) * 0.1  # y = 2x + 3 + some noise

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define a simple linear regression model
class LinearRegression(nn.Module):
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(1, 1)  # 1 input, 1 output

    def forward(self, x):
        return self.linear(x)

# Instantiate the model, define the loss function and optimizer
model = LinearRegression()
loss_fn = nn.MSELoss()  # Mean Squared Error Loss
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Function to train the model
def train(model, loss_fn, optimizer, X_train, y_train, epochs=100):
    model.train()
    for epoch in range(epochs):
        # Forward pass
        predictions = model(X_train)
        loss = loss_fn(predictions, y_train)

        # Backpropagation and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print loss every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}')

# Function to evaluate the model on test data
def evaluate(model, loss_fn, X_test, y_test):
    model.eval()  # Switch to evaluation mode
    with torch.no_grad():
        predictions = model(X_test)
        test_loss = loss_fn(predictions, y_test)
    return test_loss.item()

# Train the model
train(model, loss_fn, optimizer, X_train, y_train)

# Evaluate the model
test_loss = evaluate(model, loss_fn, X_test, y_test)
print(f'Test Loss: {test_loss:.4f}')
