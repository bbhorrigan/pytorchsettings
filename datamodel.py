import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

# Set random seed for reproducibility
torch.manual_seed(42)

# Generate fake data
X_linear = torch.randn(100, 1)  # 100 samples, 1 feature
y_linear = 2 * X_linear + 3 + torch.randn(100, 1) * 0.1  # y = 2x + 3 + noise

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_linear, y_linear, test_size=0.2, random_state=42)

# Define the model
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

# Instantiate the model
model_linear = LinearRegressionModel()

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(model_linear.parameters(), lr=0.01)

# Define training function
def train_model(model: nn.Module, criterion: nn.Module, optimizer: optim.Optimizer, 
                X_train: torch.Tensor, y_train: torch.Tensor, num_epochs: int = 100) -> None:
    model.train()
    for epoch in range(num_epochs):
        # Forward pass
        outputs = model(X_train)
        loss = criterion(outputs, y_train)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Define evaluation function
def evaluate_model(model: nn.Module, criterion: nn.Module, X_test: torch.Tensor, y_test: torch.Tensor) -> float:
    model.eval()
    with torch.no_grad():
        predicted = model(X_test)
        test_loss = criterion(predicted, y_test)
    return test_loss.item()

# Train the model
train_model(model_linear, criterion, optimizer, X_train, y_train, num_epochs=100)

# Evaluate the model
test_loss = evaluate_model(model_linear, criterion, X_test, y_test)
print(f'Test Loss: {test_loss:.4f}')
