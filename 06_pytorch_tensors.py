# ==============================================================================
# 06 — PyTorch: Tensors and Autograd
# ==============================================================================
# In files 02–04 we computed gradients by hand.
# PyTorch does this automatically for ANY computation — that's the superpower.
#
# NEW concepts:
#   Tensor    — like a NumPy array but can run on a GPU
#   autograd  — automatic differentiation: PyTorch tracks all operations and
#               can compute gradients for free with .backward()
#
# This file maps everything you already know to PyTorch equivalents.
# ==============================================================================

import torch
import numpy as np
import matplotlib.pyplot as plt

print("=" * 55)
print("PART 1 — Tensors (the NumPy you already know)")
print("=" * 55)

# Creating tensors — just like NumPy arrays
a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([4.0, 5.0, 6.0])

print(f"\nnp.array  → torch.tensor")
print(f"  a = {a}")
print(f"  b = {b}")
print(f"  a + b = {a + b}")
print(f"  a * b = {a * b}")
print(f"  a.mean() = {a.mean()}")

# Shapes — exactly like NumPy
matrix = torch.zeros(3, 4)
print(f"\ntorch.zeros(3, 4): shape = {matrix.shape}")

# Converting between NumPy and PyTorch
np_array = np.array([1.0, 2.0, 3.0])
tensor = torch.from_numpy(np_array)
back_to_numpy = tensor.numpy()
print(f"\nNumPy ↔ PyTorch conversion: {np_array} → {tensor} → {back_to_numpy}")

print("\n" + "=" * 55)
print("PART 2 — autograd: automatic gradient computation")
print("=" * 55)

# requires_grad=True tells PyTorch: "track operations on this tensor"
w = torch.tensor(2.0, requires_grad=True)
x = torch.tensor(3.0)        # input (we don't need its gradient)

# A simple computation: y = w * x^2 + 5
y = w * x**2 + 5
print(f"\ny = w * x² + 5  where w={w.item()}, x={x.item()}")
print(f"y = {y.item()}")

# .backward() computes all gradients automatically
y.backward()

# dy/dw = x² = 9  (PyTorch computes this for us)
print(f"\ndy/dw (computed by PyTorch): {w.grad.item()}")
print(f"dy/dw (by hand: x² = 3²)  : 9.0")

print("\n" + "=" * 55)
print("PART 3 — Redoing linear regression with autograd")
print("=" * 55)

# Same dataset as 02_linear_regression.py
np.random.seed(42)
x_np = np.linspace(0, 10, 30)
y_np = 10 * x_np + 30 + np.random.randn(30) * 8

x_t = torch.tensor(x_np, dtype=torch.float32)
y_t = torch.tensor(y_np, dtype=torch.float32)

# Parameters — now PyTorch tracks everything
w = torch.tensor(0.0, requires_grad=True)
b = torch.tensor(0.0, requires_grad=True)

learning_rate = 0.01
losses = []

for step in range(200):
    # Forward pass
    y_pred = w * x_t + b
    loss = ((y_pred - y_t) ** 2).mean()

    # Backward pass — one line instead of the manual gradient code in file 02
    loss.backward()

    # Update weights (without tracking these operations)
    with torch.no_grad():
        w -= learning_rate * w.grad
        b -= learning_rate * b.grad

    # Zero gradients (PyTorch accumulates them by default)
    w.grad.zero_()
    b.grad.zero_()

    losses.append(loss.item())

print(f"\nLearned: y = {w.item():.1f} * x + {b.item():.1f}")
print(f"Truth:   y = 10.0 * x + 30.0")

# --- Plot ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.scatter(x_np, y_np, color="steelblue", label="Data")
ax1.plot(x_np, w.item() * x_np + b.item(), color="crimson", lw=2,
         label=f"y = {w.item():.1f}x + {b.item():.1f}")
ax1.set_title("Linear Regression (PyTorch autograd)")
ax1.set_xlabel("x"); ax1.set_ylabel("y"); ax1.legend()

ax2.plot(losses, color="darkorange")
ax2.set_title("Training loss"); ax2.set_xlabel("Step"); ax2.set_yscale("log")

plt.tight_layout()
plt.savefig("06_output.png", dpi=120)
plt.show()
print("\nPlot saved to 06_output.png")

print("\n" + "=" * 55)
print("PART 4 — GPU check")
print("=" * 55)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\nUsing device: {device}")
t = torch.ones(3, 3).to(device)
print(f"Tensor on {device}:\n{t}")
print("\nMoving computation to GPU is just: tensor.to('cuda')")
print("On GPU, training is 10–100× faster for large models.")

# ------------------------------------------------------------------------------
# KEY LESSON
# ------------------------------------------------------------------------------
# Manual backprop (file 04) vs PyTorch autograd:
#   File 04: 10 lines of calculus that break if you change the architecture
#   PyTorch: loss.backward() — works for ANY network, ANY complexity
#
# This is why PyTorch exists.
# Next: rebuild our XOR network using PyTorch's nn.Module.
# → See 07_pytorch_nn.py
# ------------------------------------------------------------------------------
