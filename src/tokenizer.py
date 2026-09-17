from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel


DEFAULT_TOKENIZER_DIR = Path(__file__).resolve().parents[1] / "data" / "tokenizer"
EOS_TOKEN = "<|endoftext|>"


def tokenizer_hash(tokenizer_path: Path, vocab_size: int) -> str:
    digest = hashlib.sha256(tokenizer_path.read_bytes()).hexdigest()
    return f"{digest}:{vocab_size}"


class TextTokenizer:
    def __init__(self, tokenizer: Tokenizer, *, vocab_size: int, eos_id: int):
        self._tokenizer = tokenizer
        self.vocab_size = vocab_size
        self.eos_id = eos_id

    @classmethod
    def load(cls, tokenizer_dir: Path | None = None) -> TextTokenizer:
        root = tokenizer_dir or DEFAULT_TOKENIZER_DIR
        meta_path = root / "meta.json"
        tok_path = root / "tokenizer.json"
        if not tok_path.is_file():
            raise FileNotFoundError(f"{tok_path} not found; run train-tokenizer first")
        meta = json.loads(meta_path.read_text()) if meta_path.is_file() else {}
        tokenizer = Tokenizer.from_file(str(tok_path))
        if tokenizer.decoder is None:
            tokenizer.decoder = ByteLevel()
        vocab_size = int(meta.get("vocab_size", tokenizer.get_vocab_size()))
        eos_id = int(meta.get("eos_id", tokenizer.token_to_id(EOS_TOKEN) or 0))
        return cls(tokenizer, vocab_size=vocab_size, eos_id=eos_id)

    def encode(self, text: str) -> list[int]:
        return self._tokenizer.encode(text).ids

    def decode(self, ids: list[int] | list[list[int]]) -> str:
        if ids and isinstance(ids[0], list):
            return self._tokenizer.decode(ids)  # type: ignore[arg-type]
        return self._tokenizer.decode(ids)  # type: ignore[arg-type]

    def meta(self) -> dict:
        root = DEFAULT_TOKENIZER_DIR
        tok_path = root / "tokenizer.json"
        return {
            "vocab_size": self.vocab_size,
            "eos_id": self.eos_id,
            "tokenizer_hash": tokenizer_hash(tok_path, self.vocab_size),
        }
