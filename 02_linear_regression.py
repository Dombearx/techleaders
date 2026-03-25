# ==============================================================================
# 02 — Linear Regression & Gradient Descent
# ==============================================================================
# Rule-based systems need human expertise. Can a machine find rules on its own?
#
# The simplest possible answer: fit a straight line to data.
# We'll do this from scratch (no ML libraries) so every step is visible.
#
# The idea:
#   We have a line:      y = w * x + b
#   We want to find w and b such that the line fits our data.
#   We do this by measuring how wrong we are (the *loss*),
#   then nudging w and b in the direction that reduces the error.
#   That nudging process is called *gradient descent*.
# ==============================================================================

import numpy as np
import matplotlib.pyplot as plt

# --- 1. Generate some toy data ---
# Imagine: hours studied (x) → exam score (y).
# True relationship: score ≈ 10 * hours + 30 (plus some noise).

np.random.seed(42)
x = np.linspace(0, 10, 30)                  # 30 students, 0–10 hours studied
y = 10 * x + 30 + np.random.randn(30) * 8   # true line + random noise

# --- 2. Initialise our parameters randomly ---
w = 0.0   # slope (weight)
b = 0.0   # intercept (bias)

# --- 3. Define how we measure error ---
# Mean Squared Error: average of (prediction - truth)²
# Big error → big loss. Perfect fit → loss = 0.

def loss(w, b, x, y):
    predictions = w * x + b
    errors = predictions - y
    return (errors ** 2).mean()

# --- 4. Gradient descent loop ---
# The gradient tells us: "if I increase w by a tiny bit, does the loss go up or down?"
# We move w and b in the opposite direction of the gradient (downhill).

learning_rate = 0.01   # how big each step is
n = len(x)
loss_history = []

print("Training: finding the best line")
print(f"  Start: w={w:.2f}, b={b:.2f}, loss={loss(w, b, x, y):.2f}")

for step in range(200):
    # Compute predictions
    predictions = w * x + b
    errors = predictions - y

    # Gradients (calculus, but just two lines)
    grad_w = (2 / n) * (errors * x).sum()
    grad_b = (2 / n) * errors.sum()

    # Update parameters
    w -= learning_rate * grad_w
    b -= learning_rate * grad_b

    loss_history.append(loss(w, b, x, y))

    if (step + 1) % 50 == 0:
        print(f"  Step {step+1:3d}: w={w:.2f}, b={b:.2f}, loss={loss(w,b,x,y):.2f}")

print(f"\n  Learned: y = {w:.1f} * x + {b:.1f}")
print(f"  Truth:   y = 10.0 * x + 30.0")

# --- 5. Visualise ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# Plot data + fitted line
ax1.scatter(x, y, color="steelblue", label="Data (students)")
ax1.plot(x, w * x + b, color="crimson", linewidth=2, label=f"Learned line\ny = {w:.1f}x + {b:.1f}")
ax1.set_xlabel("Hours studied")
ax1.set_ylabel("Exam score")
ax1.set_title("Linear Regression")
ax1.legend()

# Plot how loss decreased
ax2.plot(loss_history, color="darkorange")
ax2.set_xlabel("Training step")
ax2.set_ylabel("Loss (MSE)")
ax2.set_title("Loss going down = model improving")
ax2.set_yscale("log")

plt.tight_layout()
plt.savefig("02_output.png", dpi=120)
plt.show()
print("\nPlot saved to 02_output.png")

# ------------------------------------------------------------------------------
# KEY LESSON
# ------------------------------------------------------------------------------
# We didn't write any rules. We gave the algorithm data and it found w and b.
#
# Gradient descent is the engine behind almost all of modern AI:
#   1. Make a prediction.
#   2. Measure how wrong you are (loss).
#   3. Compute which direction makes the loss smaller (gradient).
#   4. Take a small step in that direction.
#   5. Repeat thousands of times.
#
# Our model has 2 parameters (w and b).
# A large neural network has billions of parameters.
# The algorithm is exactly the same.
#
# Next: what if the output is not a number but a category?
# → See 03_perceptron.py
# ------------------------------------------------------------------------------
