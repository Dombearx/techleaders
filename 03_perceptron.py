# ==============================================================================
# 03 — The Perceptron (a Single Artificial Neuron)
# ==============================================================================
# In 02 we predicted a number. Now we want to predict a category: yes / no.
#
# The perceptron is the building block of all neural networks.
# It was inspired by how a biological neuron works:
#
#   inputs → multiply by weights → sum → activation → output
#
# Think of it as a weighted vote:
#   - Each input has a weight (how important is this signal?).
#   - Add all weighted inputs together.
#   - If the total is above a threshold → fire (output 1), else → don't (output 0).
#
# NEW concept: sigmoid activation function
#   Instead of a hard 0/1, we use sigmoid to get a smooth probability (0 to 1).
#   sigmoid(x) = 1 / (1 + e^(-x))
# ==============================================================================

import numpy as np
import matplotlib.pyplot as plt

# --- Sigmoid: squashes any number into (0, 1) ---
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

# ==============================================================================
# TASK 1: Learn the OR gate
# Input: two bits (0 or 1)
# Output: 1 if EITHER input is 1, else 0
# ==============================================================================

# All possible inputs and expected outputs for OR
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])

y_or = np.array([0, 1, 1, 1])  # OR truth table

# Initialise weights and bias randomly
np.random.seed(0)
w = np.random.randn(2) * 0.1   # one weight per input
b = 0.0                         # bias (shifts the threshold)

learning_rate = 0.5
loss_history_or = []

print("Learning the OR gate...")
for step in range(1000):
    # Forward pass: compute prediction
    z = X @ w + b           # weighted sum  (shape: [4])
    y_hat = sigmoid(z)      # apply sigmoid → probability

    # Loss: Binary Cross-Entropy (standard loss for yes/no problems)
    # BCE = -mean(y*log(ŷ) + (1-y)*log(1-ŷ))
    loss = -np.mean(y_or * np.log(y_hat + 1e-9) + (1 - y_or) * np.log(1 - y_hat + 1e-9))
    loss_history_or.append(loss)

    # Backward pass: gradients
    error = y_hat - y_or             # shape [4]
    grad_w = X.T @ error / len(X)    # shape [2]
    grad_b = error.mean()

    w -= learning_rate * grad_w
    b -= learning_rate * grad_b

print("Done! Final predictions:")
for i, (inputs, expected) in enumerate(zip(X, y_or)):
    pred = sigmoid(inputs @ w + b)
    print(f"  {inputs[0]} OR {inputs[1]} = {pred:.3f}  (expected {expected})")

# ==============================================================================
# TASK 2: Learn the AND gate  (just change the labels)
# ==============================================================================

y_and = np.array([0, 0, 0, 1])  # AND truth table

np.random.seed(0)
w2 = np.random.randn(2) * 0.1
b2 = 0.0

for _ in range(1000):
    z = X @ w2 + b2
    y_hat = sigmoid(z)
    error = y_hat - y_and
    w2 -= learning_rate * (X.T @ error / len(X))
    b2 -= learning_rate * error.mean()

print("\nFinal predictions for AND:")
for inputs, expected in zip(X, y_and):
    pred = sigmoid(inputs @ w2 + b2)
    print(f"  {inputs[0]} AND {inputs[1]} = {pred:.3f}  (expected {expected})")

# --- Visualise the decision boundary ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, (w_p, b_p, y_truth, title) in zip(axes, [
    (w,  b,  y_or,  "OR  gate"),
    (w2, b2, y_and, "AND gate"),
]):
    # Draw background: what does the perceptron predict for every point?
    xx, yy = np.meshgrid(np.linspace(-0.5, 1.5, 200), np.linspace(-0.5, 1.5, 200))
    grid = np.c_[xx.ravel(), yy.ravel()]
    probs = sigmoid(grid @ w_p + b_p).reshape(xx.shape)
    ax.contourf(xx, yy, probs, levels=50, cmap="RdYlGn", alpha=0.8)
    ax.contour(xx, yy, probs, levels=[0.5], colors="black", linewidths=2)

    # Draw data points
    for inputs, label in zip(X, y_truth):
        color = "green" if label == 1 else "red"
        ax.scatter(*inputs, color=color, s=200, zorder=5, edgecolors="white", linewidths=2)
        ax.text(inputs[0] + 0.05, inputs[1] + 0.05, f"{'True' if label else 'False'}")

    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(-0.5, 1.5)
    ax.set_xlabel("Input A")
    ax.set_ylabel("Input B")
    ax.set_title(f"Perceptron: {title}\n(green=1, red=0, line=decision boundary)")

plt.tight_layout()
plt.savefig("03_output.png", dpi=120)
plt.show()
print("\nPlot saved to 03_output.png")

# ------------------------------------------------------------------------------
# KEY LESSON
# ------------------------------------------------------------------------------
# A single perceptron can only separate data with a STRAIGHT LINE.
# OR and AND are linearly separable → the perceptron learns them easily.
#
# But what about XOR?
#   0 XOR 0 = 0     0 XOR 1 = 1
#   1 XOR 0 = 1     1 XOR 1 = 0
#
# No single straight line can separate the 1s from the 0s in XOR.
# A single neuron CANNOT learn XOR. We need multiple neurons (layers).
# → See 04_neural_net_scratch.py
# ------------------------------------------------------------------------------
