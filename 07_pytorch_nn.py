# ==============================================================================
# 07 — Neural Network with PyTorch nn.Module
# ==============================================================================
# In 04 we built a 2-layer network by hand (numpy arrays + manual backprop).
# Now we build the SAME network using PyTorch's building blocks.
#
# NEW concepts:
#   nn.Module   — base class for every PyTorch neural network
#   nn.Linear   — a fully-connected layer (replaces our W, b matrices)
#   nn.Sequential — stack layers in a list (clean and readable)
#   Optimizer   — torch.optim handles the "w -= lr * grad" update for us
#   DataLoader  — utility to feed data in batches (important for large datasets)
#
# By the end: same XOR result, much cleaner code, ready to scale to MNIST.
# ==============================================================================

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# --- XOR dataset (same as file 04) ---
X = torch.tensor([[0., 0.],
                   [0., 1.],
                   [1., 0.],
                   [1., 1.]])
y = torch.tensor([[0.], [1.], [1.], [0.]])

# ==============================================================================
# Defining a network with nn.Module
# ==============================================================================

class XORNet(nn.Module):
    """2-layer network that solves XOR."""

    def __init__(self):
        super().__init__()
        # nn.Linear(in_features, out_features) = weight matrix + bias
        self.hidden = nn.Linear(2, 3)    # input → hidden (2 inputs, 3 neurons)
        self.output = nn.Linear(3, 1)    # hidden → output

    def forward(self, x):
        # forward() defines what happens on each call: net(input)
        x = torch.sigmoid(self.hidden(x))
        x = torch.sigmoid(self.output(x))
        return x


model = XORNet()
print("Network architecture:")
print(model)
total_params = sum(p.numel() for p in model.parameters())
print(f"\nTotal trainable parameters: {total_params}")
print("(weight matrices + biases → same 11 numbers we had in file 04)")

# ==============================================================================
# Training loop — the same 3 steps, now using PyTorch tools
# ==============================================================================

# Loss function and optimiser
criterion = nn.MSELoss()                      # mean squared error
optimizer = optim.SGD(model.parameters(), lr=0.5)  # gradient descent

losses = []

for step in range(5000):
    # 1. Forward pass
    y_pred = model(X)
    loss = criterion(y_pred, y)

    # 2. Backward pass
    optimizer.zero_grad()   # reset gradients from last step
    loss.backward()         # compute gradients

    # 3. Update
    optimizer.step()        # applies: param -= lr * param.grad

    losses.append(loss.item())

print("\nTrained! Predictions:")
with torch.no_grad():          # disable gradient tracking for inference
    preds = model(X)
for inputs, pred, expected in zip(X, preds, y):
    print(f"  {int(inputs[0])} XOR {int(inputs[1])} = {pred.item():.3f}  (expected {int(expected.item())})")

# ==============================================================================
# Showing what's inside: weights and activations
# ==============================================================================

print("\nLearned weights in the hidden layer:")
print(model.hidden.weight.data)
print("Hidden biases:")
print(model.hidden.bias.data)

print("\nActivations (hidden layer output) for each input:")
with torch.no_grad():
    hidden_acts = torch.sigmoid(model.hidden(X))
    for inp, act in zip(X, hidden_acts):
        print(f"  {inp.tolist()} → hidden neurons: {[f'{v:.2f}' for v in act.tolist()]}")

# ==============================================================================
# Visualise
# ==============================================================================
import numpy as np

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Decision boundary
ax = axes[0]
xx, yy = np.meshgrid(np.linspace(-0.2, 1.2, 300), np.linspace(-0.2, 1.2, 300))
grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)
with torch.no_grad():
    probs = model(grid).numpy().reshape(xx.shape)

ax.contourf(xx, yy, probs, levels=50, cmap="RdYlGn", alpha=0.8)
ax.contour(xx, yy, probs, levels=[0.5], colors="black", linewidths=2)
for (xi, yi), label in zip(X.numpy(), y.numpy().ravel()):
    color = "green" if label == 1 else "red"
    ax.scatter(xi, yi, color=color, s=200, zorder=5, edgecolors="white", linewidths=2)
ax.set_title("XOR — PyTorch decision boundary")
ax.set_xlabel("Input A"); ax.set_ylabel("Input B")

# Loss curve
axes[1].plot(losses, color="darkorange")
axes[1].set_yscale("log")
axes[1].set_title("Training loss")
axes[1].set_xlabel("Step"); axes[1].set_ylabel("MSE Loss")

plt.tight_layout()
plt.savefig("07_output.png", dpi=120)
plt.show()
print("\nPlot saved to 07_output.png")

# ------------------------------------------------------------------------------
# KEY LESSON
# ------------------------------------------------------------------------------
# The 3-step loop is always the same in PyTorch:
#
#     optimizer.zero_grad()
#     loss = criterion(model(x), y)
#     loss.backward()
#     optimizer.step()
#
# Change the network, change the loss, change the optimiser — this loop stays.
#
# We're ready to apply this to real data: 70,000 handwritten digits.
# → See 08_mnist.ipynb
# ------------------------------------------------------------------------------
