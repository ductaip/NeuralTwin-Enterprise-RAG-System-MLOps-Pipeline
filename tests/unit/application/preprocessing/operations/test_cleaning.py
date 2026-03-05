import pytest
from llm_engineering.application.preprocessing.operations.cleaning import clean_text

def test_clean_text_basic():
    """Test basic text cleaning without special characters."""
    assert clean_text("Hello World") == "Hello World"

def test_clean_text_special_chars():
    """Test removal of special characters."""
    # Keeps letters, numbers, whitespace, dots, commas, exclamation and question marks
    text = "Hello @#& World!"
    # The regex replaces special chars with space, then collapses spaces
    assert clean_text(text) == "Hello World!"

def test_clean_text_whitespace():
    """Test whitespace handling."""
    text = "  Hello   World  \n "
    assert clean_text(text) == "Hello World"

def test_clean_text_punctuation():
    """Test punctuation preservation."""
    text = "Hello, World! How are you?"
    assert clean_text(text) == "Hello, World! How are you?"

def test_clean_text_mixed():
    """Test mixed case with special chars and whitespace."""
    text = "  Hello @# World! 123...  "
    # Note: re.sub(r"[^\w\s.,!?]", " ", text) replaces special chars with space
    # 123... should be preserved
    cleaned = clean_text(text)
    assert "Hello" in cleaned
    assert "World" in cleaned
    assert "123..." in cleaned
    assert "@" not in cleaned
    assert "#" not in cleaned
