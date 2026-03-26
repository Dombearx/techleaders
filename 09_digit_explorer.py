# ==============================================================================
# 09 — Interactive Digit Explorer (Gradio)
# ==============================================================================
# We've trained a network that recognises handwritten digits.
# Now let's make it interactive: draw a digit in the browser,
# and watch the network's output probabilities update in real time.
#
# Run:  python 09_digit_explorer.py
# Then open the URL printed in the terminal (usually http://127.0.0.1:7860)
#
# Prerequisites: run 08_mnist.ipynb first to create mnist_model.pt
# ==============================================================================

from typing import Any

import torch
import torch.nn as nn
import numpy as np
import gradio as gr
import matplotlib
matplotlib.use("Agg")   # non-interactive backend (Gradio handles the display)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ==============================================================================
# Re-define the same network architecture (must match what we trained)
# ==============================================================================

class MNISTNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# Load the trained weights
model = MNISTNet()
try:
    model.load_state_dict(torch.load("mnist_model.pt", map_location="cpu", weights_only=True))
    print("Loaded mnist_model.pt")
except FileNotFoundError:
    print("WARNING: mnist_model.pt not found.")
    print("Please run 08_mnist.ipynb first to train and save the model.")

model.eval()


# ==============================================================================
# Preprocessing: Gradio sketchpad → PyTorch tensor (same as MNIST training)
# ==============================================================================

from torchvision import transforms

preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Grayscale(),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])

def predict(sketch: dict[str, Any] | None) -> tuple[plt.Figure | None, str]:
    """
    sketch: dict with 'composite' key containing an RGBA numpy array
            (what Gradio's sketchpad returns)
    Returns:
      - a bar chart figure showing probabilities
      - a text label
    """
    if sketch is None or sketch.get("composite") is None:
        return None, "Draw a digit above ↑"

    img_rgba = sketch["composite"]         # H × W × 4

    # Convert to grayscale numpy (use alpha channel as intensity — white drawing on black)
    alpha = img_rgba[:, :, 3].astype(np.float32) / 255.0
    # Invert: MNIST has white digit on black background
    gray = (alpha * 255).astype(np.uint8)

    if gray.max() == 0:
        return None, "Canvas is empty"

    # Preprocess
    tensor = preprocess(gray)           # (1, 28, 28)
    tensor = tensor.unsqueeze(0)        # (1, 1, 28, 28)

    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1).squeeze().numpy()

    predicted = int(probs.argmax())
    confidence = float(probs.max())

    # ---- Build the bar chart ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 4),
                             gridspec_kw={"width_ratios": [1, 3]})

    # Left: the 28×28 image the network actually sees
    with torch.no_grad():
        display_img = tensor.squeeze().numpy()
    axes[0].imshow(display_img, cmap="gray", interpolation="nearest")
    axes[0].set_title("What the network sees\n(28×28 normalised)", fontsize=10)
    axes[0].axis("off")

    # Right: probability bars
    bar_colors = ["crimson" if i == predicted else "steelblue" for i in range(10)]
    bars = axes[1].bar(range(10), probs * 100, color=bar_colors, edgecolor="white")
    axes[1].set_xticks(range(10))
    axes[1].set_xticklabels([str(i) for i in range(10)], fontsize=12)
    axes[1].set_xlabel("Digit", fontsize=12)
    axes[1].set_ylabel("Probability (%)", fontsize=12)
    axes[1].set_ylim(0, 105)
    axes[1].set_title(f"Output probabilities  →  predicted: {predicted}  ({confidence:.1%})",
                      fontsize=12)

    # Annotate bars with values above 1%
    for bar, p in zip(bars, probs):
        if p > 0.01:
            axes[1].text(bar.get_x() + bar.get_width() / 2,
                         bar.get_height() + 1,
                         f"{p:.1%}", ha="center", va="bottom", fontsize=9)

    legend_elements = [
        mpatches.Patch(color="crimson",   label="Predicted class"),
        mpatches.Patch(color="steelblue", label="Other classes"),
    ]
    axes[1].legend(handles=legend_elements, loc="upper right")

    plt.tight_layout()

    label = f"Predicted: **{predicted}**  (confidence {confidence:.1%})"
    return fig, label


# ==============================================================================
# Build the Gradio UI
# ==============================================================================

with gr.Blocks(title="MNIST Digit Explorer") as demo:
    gr.Markdown(
        """
        # MNIST Digit Explorer
        Draw a digit (0–9) in the box below and press **Classify ➜**  
        The bar chart shows what the network thinks — watch how confidence changes as you draw!
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            sketch = gr.Sketchpad(
                label="Draw a digit here",
                canvas_size=(280, 280),
                brush=gr.Brush(default_size=20, default_color="#ffffff"),
            )
            btn = gr.Button("Classify ➜", variant="primary")
            clear = gr.ClearButton([sketch], value="Clear canvas")

        with gr.Column(scale=2):
            chart = gr.Plot(label="Network output")
            label_out = gr.Markdown("Draw a digit and press Classify ➜")

    btn.click(fn=predict, inputs=sketch, outputs=[chart, label_out])

    gr.Markdown(
        """
        ---
        **How it works:**  
        1. Your drawing is resized to 28×28 pixels  
        2. Each pixel becomes a number fed into the network (784 inputs)  
        3. The network outputs 10 scores — one per digit  
        4. Softmax converts those scores into probabilities summing to 100%
        """
    )


if __name__ == "__main__":
    demo.launch()
