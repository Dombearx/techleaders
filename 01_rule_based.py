# ==============================================================================
# 01 — Rule-Based AI
# ==============================================================================
# The very first question: what IS artificial intelligence?
#
# Before machine learning existed, programmers wrote AI by hand.
# They observed patterns in data and translated them into IF/ELSE rules.
#
# This is called a "rule-based system" or "expert system".
# It works — but someone has to write every single rule.
# ==============================================================================

# Let's build a tiny animal classifier.
# A domain expert (a biologist) told us:
#   - has feathers  → bird
#   - has scales + lives in water → fish
#   - has fur + warm-blooded → mammal
#   - otherwise → unknown


def classify_animal(has_feathers, has_scales, has_fur, lives_in_water, warm_blooded):
    """Classify an animal using hand-written rules."""
    if has_feathers:
        return "bird"
    if has_scales and lives_in_water:
        return "fish"
    if has_fur and warm_blooded:
        return "mammal"
    return "unknown"


# --- Test our rule-based classifier ---

animals = [
    # name            feathers  scales  fur    water  warm
    ("Eagle",         True,     False,  False, False, True),
    ("Salmon",        False,    True,   False, True,  False),
    ("Dog",           False,    False,  True,  False, True),
    ("Mystery beast", False,    False,  False, False, False),
]

print("Rule-Based Animal Classifier")
print("=" * 35)
for name, *features in animals:
    result = classify_animal(*features)
    print(f"  {name:<20} → {result}")

# ------------------------------------------------------------------------------
# Now let's do something more interesting: classify e-mails as spam or not spam.
# A human expert wrote these rules after looking at many e-mails.
# ------------------------------------------------------------------------------

SPAM_WORDS = ["free", "winner", "click here", "guarantee", "limited offer"]

def is_spam(email_text: str) -> bool:
    """Simple spam detector using keyword rules."""
    text = email_text.lower()
    matches = [word for word in SPAM_WORDS if word in text]
    # Rule: 2 or more spam keywords → spam
    return len(matches) >= 2

emails = [
    "Hi, are you free for lunch tomorrow?",
    "FREE winner! Click here for your guaranteed limited offer prize!",
    "Your invoice is attached. Please review.",
    "CLICK HERE for a free limited offer — you are the winner!",
]

print("\nSpam Classifier")
print("=" * 35)
for email in emails:
    label = "SPAM" if is_spam(email) else "OK  "
    preview = email[:45] + "..." if len(email) > 45 else email
    print(f"  [{label}]  {preview}")

# ------------------------------------------------------------------------------
# KEY LESSON
# ------------------------------------------------------------------------------
# + Rule-based systems are easy to understand and explain.
# + They work perfectly when rules are simple and well-defined.
#
# - Someone has to write every rule by hand.
# - Rules break down on edge cases ("what about a penguin with feathers?").
# - They don't scale: real spam has millions of patterns.
#
# The next question: can a computer LEARN the rules from examples?
# → That's machine learning. See 02_linear_regression.py
# ------------------------------------------------------------------------------
