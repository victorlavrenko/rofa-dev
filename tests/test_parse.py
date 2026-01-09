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


def test_extract_choice_letter_from_options():
    options = {
        "A": "9 minutes",
        "B": "4.5 minutes",
        "C": "27 minutes",
        "D": "13.5 minutes",
    }
    text = "The final answer is 27 minutes after scaling the rate to 1/3."
    assert extract_choice_letter(text, options=options) == "C"

    list_options = ["cyan", "magenta", "yellow", "black"]
    text = "Printing uses black ink for the key color."
    assert extract_choice_letter(text, options=list_options) == "D"
