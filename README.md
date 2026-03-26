# AI Introduction — Hands-on Course

A step-by-step introduction to Artificial Intelligence, starting from simple rule-based systems and building up to neural networks and large language models.

## Setup

**Requirements:** Python 3.11+, [Poetry](https://python-poetry.org/), [Ollama](https://ollama.com/)

```bash
# Install dependencies
poetry install

# Activate the virtual environment
poetry shell
```

For the LLM example (file 10) you also need Ollama running locally:

```bash
# Install Ollama from https://ollama.com, then pull a small model
ollama pull llama3.2
ollama serve   # starts the API on http://localhost:11434
```

---

## Curriculum

| Notebook | Topic | How to open |
|---|---|---|
| `01_rule_based.ipynb` | What is AI? Rule-based classifier | `jupyter notebook 01_rule_based.ipynb` |
| `02_linear_regression.ipynb` | Finding patterns with gradient descent | `jupyter notebook 02_linear_regression.ipynb` |
| `03_perceptron.ipynb` | The artificial neuron | `jupyter notebook 03_perceptron.ipynb` |
| `04_neural_net_scratch.ipynb` | 2-layer net solves XOR | `jupyter notebook 04_neural_net_scratch.ipynb` |
| `05_decision_trees.ipynb` | A completely different approach | `jupyter notebook 05_decision_trees.ipynb` |
| `06_pytorch_tensors.ipynb` | PyTorch: tensors & autograd | `jupyter notebook 06_pytorch_tensors.ipynb` |
| `07_pytorch_nn.ipynb` | Same net, the PyTorch way | `jupyter notebook 07_pytorch_nn.ipynb` |
| `08_mnist.ipynb` | Recognising handwritten digits | `jupyter notebook 08_mnist.ipynb` |
| `09_digit_explorer.py` | Interactive digit explorer (Gradio) | `python 09_digit_explorer.py` |
| `10_llm_pydantic_ai.ipynb` | Calling a local LLM (Ollama) | `jupyter notebook 10_llm_pydantic_ai.ipynb` |

> **Note:** `09_digit_explorer.py` remains a Python script because it runs a Gradio web app that must be launched from the terminal.

---

## Learning Path

```
Rules → Numbers → One Neuron → Many Neurons → Different model entirely (Trees)
     → PyTorch → Real dataset (MNIST) → Interactive UI → Language Models
```

Each notebook is self-contained and heavily commented. Run them **in order**, executing every cell top-to-bottom. Each important step produces a visualisation inline so you can see the effect immediately.
