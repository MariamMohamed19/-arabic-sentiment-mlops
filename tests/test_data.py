import pandas as pd
import pytest

from src.data.prepare import clean_text, verify_labels


def test_clean_text_normalizes():
    assert clean_text("جميييييل  جداااا") == "جمييل جداا"
    assert clean_text("مَنْتَجٌ") == "منتج"


def test_verify_labels_rejects_unknown():
    with pytest.raises(ValueError):
        verify_labels(pd.DataFrame({"label": [0], "label_text": ["Weird"]}))


def test_verify_labels_ok():
    verify_labels(pd.DataFrame({"label": [0, 1, 2], "label_text": ["Negative", "Positive", "Mixed"]}))
