"""Prompt templates for ROFA experiments."""

SYSTEM = (
    "You are an expert medical assistant."
    "You are to be a helpful, respectful, and honest assistant."
)


def build_user(q):
    """Build the user prompt for a single dataset example."""
    return (
        "For the following multiple-choice question, select one correct answer. "
        "Let's think step by step. "
        f"Question: {q['question']} "
        f"Options: A. {q['opa']} B. {q['opb']} C. {q['opc']} D. {q['opd']}"
    )
