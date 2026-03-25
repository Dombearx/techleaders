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

| File | Topic | How to run |
|---|---|---|
| `01_rule_based.py` | What is AI? Rule-based classifier | `python 01_rule_based.py` |
| `02_linear_regression.py` | Finding patterns with gradient descent | `python 02_linear_regression.py` |
| `03_perceptron.py` | The artificial neuron | `python 03_perceptron.py` |
| `04_neural_net_scratch.py` | 2-layer net solves XOR | `python 04_neural_net_scratch.py` |
| `05_decision_trees.ipynb` | A completely different approach | `jupyter notebook 05_decision_trees.ipynb` |
| `06_pytorch_tensors.py` | PyTorch: tensors & autograd | `python 06_pytorch_tensors.py` |
| `07_pytorch_nn.py` | Same net, the PyTorch way | `python 07_pytorch_nn.py` |
| `08_mnist.ipynb` | Recognising handwritten digits | `jupyter notebook 08_mnist.ipynb` |
| `09_digit_explorer.py` | Interactive digit explorer (Gradio) | `python 09_digit_explorer.py` |
| `10_llm_pydantic_ai.py` | Calling a local LLM (Ollama) | `python 10_llm_pydantic_ai.py` |

---

## Learning Path

```
Rules → Numbers → One Neuron → Many Neurons → Different model entirely (Trees)
     → PyTorch → Real dataset (MNIST) → Interactive UI → Language Models
```

Each file is self-contained and heavily commented. Read and run them in order.
