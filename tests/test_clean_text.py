# tests/test_clean_text.py
import re
import pytest
from src.preprocessing.clean_text import clean_text

@pytest.mark.parametrize("input_text,expected_substr", [
    ("This is a TEST!!!", "test"),
    ("Visit http://example.com now", "visit now"),
    ("Numbers 123 and symbols #$%", "numbers and symbols"),
])
def test_clean_text(input_text, expected_substr):
    cleaned = clean_text(input_text)
    assert isinstance(cleaned, str), "Should return a string"
    # check that expected key words are present
    assert expected_substr in cleaned
    # ensure uppercase removed
    assert cleaned.lower() == cleaned
    # ensure no punctuation or URLs
    assert not re.search(r"http\S+", cleaned)
    assert not re.search(r"[^a-zA-Z ]", cleaned)

def test_non_string_input():
    with pytest.raises(ValueError):
        clean_text(None)
