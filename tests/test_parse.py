from rofa.parse import extract_choice_letter


def test_extract_choice_letter_examples():
    samples = [
        ("Final answer: A", "A"),
        ("Answer - b", "B"),
        ("So the answer is C.", "C"),
        ("Correct option is D", "D"),
        ("Option B is correct because...", "B"),
        ("After thinking, the answer is A", "A"),
        ("(C) is best", "C"),
        ("We choose D!", "D"),
        ("\nA. first\nB. second\nC. third\nD. fourth\nTherefore: B", "B"),
        ("Reasoning... final answer: d", "D"),
        ("The answer is A but maybe B", "A"),
    ]

    for text, expected in samples:
        assert extract_choice_letter(text) == expected
