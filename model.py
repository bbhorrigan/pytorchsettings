import torch
import gpytorch
import math
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from gpytorch.models import ExactGP
from gpytorch.means import ConstantMean
from gpytorch.kernels import ScaleKernel, RBFKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood

class GPRegressionModel(ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(GPRegressionModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = ConstantMean()
        self.covar_module = ScaleKernel(RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

def train_model(model, likelihood, train_x, train_y, training_iter=50):
    model.train()
    likelihood.train()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    mll = ExactMarginalLogLikelihood(likelihood, model)
    
    for i in range(training_iter):
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        loss.backward()
        print(f'Iter {i + 1}/{training_iter} - Loss: {loss.item():.3f}')
        optimizer.step()

def evaluate_model(model, likelihood, test_x, test_y):
    model.eval()
    likelihood.eval()
    
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        observed_pred = likelihood(model(test_x))
        mean = observed_pred.mean
        lower, upper = observed_pred.confidence_region()
        
        rmse = torch.sqrt(torch.mean((mean - test_y) ** 2))
        print(f'RMSE: {rmse.item():.3f}')
        
        return mean, lower, upper

def plot_results(train_x, train_y, test_x, test_y, mean, lower, upper):
    plt.figure(figsize=(12, 6))
    plt.plot(train_x.numpy(), train_y.numpy(), 'k*')
    plt.plot(test_x.numpy(), mean.numpy(), 'b')
    plt.fill_between(test_x.numpy(), lower.numpy(), upper.numpy(), alpha=0.5)
    plt.legend(['Observed Data', 'Mean', 'Confidence'])
    plt.show()

# Assuming train_x, train_y, test_x, and test_y are already defined

# Normalize the data
scaler_x = StandardScaler()
scaler_y = StandardScaler()

train_x_normalized = torch.tensor(scaler_x.fit_transform(train_x), dtype=torch.float32)
train_y_normalized = torch.tensor(scaler_y.fit_transform(train_y.reshape(-1, 1)).squeeze(), dtype=torch.float32)
test_x_normalized = torch.tensor(scaler_x.transform(test_x), dtype=torch.float32)
test_y_normalized = torch.tensor(scaler_y.transform(test_y.reshape(-1, 1)).squeeze(), dtype=torch.float32)

# Initialize likelihood and model
likelihood = GaussianLikelihood()
model = GPRegressionModel(train_x_normalized, train_y_normalized, likelihood)

# Train the model
train_model(model, likelihood, train_x_normalized, train_y_normalized)

# Evaluate the model
mean, lower, upper = evaluate_model(model, likelihood, test_x_normalized, test_y_normalized)

# Denormalize the predictions
mean_denorm = torch.tensor(scaler_y.inverse_transform(mean.numpy().reshape(-1, 1)).squeeze())
lower_denorm = torch.tensor(scaler_y.inverse_transform(lower.numpy().reshape(-1, 1)).squeeze())
upper_denorm = torch.tensor(scaler_y.inverse_transform(upper.numpy().reshape(-1, 1)).squeeze())

# Plot the results
plot_results(train_x, train_y, test_x, test_y, mean_denorm, lower_denorm, upper_denorm)
