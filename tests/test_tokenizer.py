from __future__ import annotations

import json
from pathlib import Path

import pytest

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "tokenizer"


def test_tokenizer_round_trip_if_fixture():
    tok_path = FIXTURE_DIR / "tokenizer.json"
    if not tok_path.is_file():
        pytest.skip("tokenizer fixture missing")
    from tokenizers import Tokenizer

    from tokenizer import TextTokenizer

    meta = json.loads((FIXTURE_DIR / "meta.json").read_text())
    tok = TextTokenizer(Tokenizer.from_file(str(tok_path)), vocab_size=meta["vocab_size"], eos_id=meta["eos_id"])
    text = "Once upon a time"
    ids = tok.encode(text)
    assert tok.decode(ids) == text
