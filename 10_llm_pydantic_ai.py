# ==============================================================================
# 10 — Calling a Local LLM with pydantic-ai + Ollama
# ==============================================================================
# All the networks we built so far work on numbers (pixels, measurements).
# Large Language Models (LLMs) work on TEXT — and they're built on the same
# neural net ideas, just scaled up enormously (billions of parameters).
#
# Instead of training one ourselves, we call a pre-trained model via Ollama,
# which runs it locally on your machine (no internet, no API key needed).
#
# SETUP:
#   1. Install Ollama from https://ollama.com
#   2. ollama pull llama3.2       ← downloads a ~2GB model
#   3. ollama serve               ← starts the local API server
#   4. python 10_llm_pydantic_ai.py
#
# pydantic-ai wraps the model call and lets us get STRUCTURED output
# (a Pydantic model) instead of raw text — great for building real apps.
# ==============================================================================

from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.ollama import OllamaProvider

# ==============================================================================
# Connect to the local Ollama instance
# Default model: llama3.2 — small and fast, good for demos.
# Change MODEL_NAME to any model you've pulled (ollama list to see them).
# ==============================================================================

MODEL_NAME = "llama3.2"

ollama_model = OpenAIChatModel(
    model_name=MODEL_NAME,
    provider=OllamaProvider(base_url="http://localhost:11434/v1"),
)

# ==============================================================================
# DEMO 1 — Plain text conversation
# ==============================================================================

plain_agent = Agent(ollama_model)

print("=" * 60)
print("DEMO 1 — Plain text chat")
print("=" * 60)

result = plain_agent.run_sync(
    "Explain what a neural network is in 2 sentences, as if talking to a 10-year-old."
)
print(f"\nQuestion: What is a neural network? (explain to a 10-year-old)\n")
print(f"Answer: {result.output}")
print(f"\n[Tokens used: {result.usage()}]")

# ==============================================================================
# DEMO 2 — Structured output with a Pydantic model
#
# Instead of getting back raw text, we define EXACTLY what we want:
# a Python object with typed fields. pydantic-ai tells the model to fill them.
# ==============================================================================

class ConceptExplanation(BaseModel):
    """A beginner-friendly explanation of an AI concept."""
    concept: str           = Field(description="The AI concept being explained")
    one_line_summary: str  = Field(description="What it is in one sentence")
    real_world_analogy: str = Field(description="An analogy a non-technical person would understand")
    example: str           = Field(description="A concrete example (2-3 sentences)")
    difficulty: str        = Field(description="Beginner / Intermediate / Advanced")


structured_agent = Agent(ollama_model, output_type=ConceptExplanation)

concepts = [
    "gradient descent",
    "overfitting",
    "neural network",
]

print("\n" + "=" * 60)
print("DEMO 2 — Structured output (Pydantic model)")
print("=" * 60)

for concept in concepts:
    print(f"\n--- {concept.upper()} ---")
    result = structured_agent.run_sync(
        f"Explain the following AI concept for a beginner: {concept}"
    )
    explanation: ConceptExplanation = result.output

    print(f"  Summary:  {explanation.one_line_summary}")
    print(f"  Analogy:  {explanation.real_world_analogy}")
    print(f"  Example:  {explanation.example}")
    print(f"  Level:    {explanation.difficulty}")

# ==============================================================================
# DEMO 3 — Multi-turn conversation (memory across turns)
# ==============================================================================

print("\n" + "=" * 60)
print("DEMO 3 — Multi-turn conversation")
print("=" * 60)

class QuizQuestion(BaseModel):
    """A quiz question about what we've learned."""
    question: str
    options: list[str] = Field(description="Exactly 4 multiple choice options")
    correct_option_index: int  = Field(description="0-based index of the correct option")
    explanation: str

quiz_agent = Agent(
    ollama_model,
    output_type=QuizQuestion,
    system_prompt=(
        "You are a friendly AI teacher. Generate short, clear multiple-choice "
        "quiz questions about basic AI and machine learning concepts."
    ),
)

topics = ["What gradient descent does", "What overfitting means"]

for topic in topics:
    result = quiz_agent.run_sync(f"Create a quiz question about: {topic}")
    q: QuizQuestion = result.output

    print(f"\nQ: {q.question}")
    for i, opt in enumerate(q.options):
        marker = "✓" if i == q.correct_option_index else " "
        print(f"  [{marker}] {i+1}. {opt}")
    print(f"  → {q.explanation}")

# ==============================================================================
# KEY LESSON
# ==============================================================================
print("\n" + "=" * 60)
print("KEY LESSON")
print("=" * 60)
print("""
LLMs are neural networks trained on massive amounts of text.
They predict the next token — the same idea as linear regression
(predict the next value), just with 7+ billion parameters and
trained on most of the internet.

pydantic-ai adds structure: instead of free text, you get back
a Python object you can use in your code — no string parsing needed.

Ollama lets you run these models 100% locally:
  - Your data never leaves your machine
  - No API costs
  - Works offline
""")
