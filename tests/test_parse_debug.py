from rofa.core.parse import extract_choice_letter_debug


def test_extract_choice_letter_debug_empty():
    debug = extract_choice_letter_debug("")
    assert debug.letter is None
    assert debug.method == "empty"


def test_extract_choice_letter_debug_tail_score():
    debug = extract_choice_letter_debug("We considered options. Therefore B.")
    assert debug.letter == "B"
    assert debug.method == "tail-score"
    assert debug.scores["B"] >= 5


def test_extract_choice_letter_debug_option_text():
    options = ["cyan", "magenta", "yellow", "black"]
    debug = extract_choice_letter_debug(
        "Printing uses black ink for the key color.", 
        options=options
    )
    assert debug.letter == "D"
    assert debug.method == "option-text"
    assert debug.matched_option == "D"
