# ==============================================================================
# 04 — Neural Network from Scratch
# ==============================================================================
# In 03 we saw that one neuron can't learn XOR.
# The fix: stack neurons in LAYERS.
#
# Our network:
#
#   Input (2)  →  Hidden layer (3 neurons)  →  Output (1 neuron)
#
# Each arrow is a weight. Each neuron has a bias.
# This is called a "fully connected" or "dense" network.
#
# NEW concept: backpropagation
#   How do we know which weight to update, and by how much?
#   We use the chain rule from calculus to propagate the error backwards
#   through the layers. PyTorch does this automatically later — here we do
#   it by hand so you can see exactly what's happening.
# ==============================================================================

import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_d(x):
    s = sigmoid(x)
    return s * (1 - s)

# ==============================================================================
# XOR dataset — impossible for a single neuron, easy for 2 layers
# ==============================================================================
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])

y = np.array([[0],
              [1],
              [1],
              [0]])   # XOR outputs

# ==============================================================================
# Network architecture: input(2) → hidden(3) → output(1)
# W1: (2, 3)  b1: (1, 3)
# W2: (3, 1)  b2: (1, 1)
# ==============================================================================
np.random.seed(42)
W1 = np.random.randn(2, 3) * 0.5
b1 = np.zeros((1, 3))
W2 = np.random.randn(3, 1) * 0.5
b2 = np.zeros((1, 1))

learning_rate = 0.5
losses = []

for step in range(5000):
    # ---- FORWARD PASS --------------------------------------------------------
    # Layer 1: linear transform + sigmoid
    z1 = X @ W1 + b1          # (4, 3)
    a1 = sigmoid(z1)           # (4, 3)  ← activations of hidden neurons

    # Layer 2: linear transform + sigmoid
    z2 = a1 @ W2 + b2          # (4, 1)
    a2 = sigmoid(z2)           # (4, 1)  ← final prediction

    # Loss: mean squared error
    loss = ((a2 - y) ** 2).mean()
    losses.append(loss)

    # ---- BACKWARD PASS (backpropagation) ------------------------------------
    # How much does the output need to change?
    d_a2 = 2 * (a2 - y) / len(X)         # gradient of loss w.r.t. a2
    d_z2 = d_a2 * sigmoid_d(z2)           # chain rule through sigmoid

    d_W2 = a1.T @ d_z2                    # gradient for W2
    d_b2 = d_z2.sum(axis=0, keepdims=True)

    # Propagate gradient back through layer 1
    d_a1 = d_z2 @ W2.T
    d_z1 = d_a1 * sigmoid_d(z1)

    d_W1 = X.T @ d_z1
    d_b1 = d_z1.sum(axis=0, keepdims=True)

    # ---- UPDATE WEIGHTS ------------------------------------------------------
    W2 -= learning_rate * d_W2
    b2 -= learning_rate * d_b2
    W1 -= learning_rate * d_W1
    b1 -= learning_rate * d_b1

# --- Results ---
print("Neural Network learned XOR:")
print("-" * 30)
for inputs, expected in zip(X, y):
    pred = sigmoid(sigmoid(inputs @ W1 + b1) @ W2 + b2)[0, 0]
    print(f"  {inputs[0]} XOR {inputs[1]} = {pred:.3f}  (expected {int(expected[0])})")

# --- Visualise ---
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Decision boundary
ax = axes[0]
xx, yy = np.meshgrid(np.linspace(-0.2, 1.2, 300), np.linspace(-0.2, 1.2, 300))
grid = np.c_[xx.ravel(), yy.ravel()]
a1_g = sigmoid(grid @ W1 + b1)
probs = sigmoid(a1_g @ W2 + b2).reshape(xx.shape)

ax.contourf(xx, yy, probs, levels=50, cmap="RdYlGn", alpha=0.8)
ax.contour(xx, yy, probs, levels=[0.5], colors="black", linewidths=2)
for (xi, yi), label in zip(X, y.ravel()):
    color = "green" if label == 1 else "red"
    ax.scatter(xi, yi, color=color, s=200, zorder=5, edgecolors="white", linewidths=2)
    text_label = "1 (True)" if label else "0 (False)"
    ax.annotate(text_label, (xi + 0.03, yi + 0.04))
ax.set_title("XOR decision boundary\n(one neuron couldn't do this!)")
ax.set_xlabel("Input A"); ax.set_ylabel("Input B")

# Loss curve
axes[1].plot(losses, color="darkorange")
axes[1].set_title("Training loss")
axes[1].set_xlabel("Step"); axes[1].set_ylabel("MSE Loss")
axes[1].set_yscale("log")

plt.tight_layout()
plt.savefig("04_output.png", dpi=120)
plt.show()
print("\nPlot saved to 04_output.png")

# ------------------------------------------------------------------------------
# KEY LESSON
# ------------------------------------------------------------------------------
# By adding a hidden layer we gave the network the ability to form a curved
# decision boundary. It now solves problems a straight line never could.
#
# We implemented backpropagation by hand: ~10 lines of calculus.
# Modern frameworks (PyTorch) do this automatically for ANY network.
#
# The 3 ingredients of training any neural network:
#   1. Forward pass  → compute predictions
#   2. Loss          → measure error
#   3. Backward pass → compute gradients, update weights
#
# Everything from here on builds on exactly this loop.
# But first: a completely different approach to learning.
# → See 05_decision_trees.ipynb
# ------------------------------------------------------------------------------
