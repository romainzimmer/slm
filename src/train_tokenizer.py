from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_from_disk
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.normalizers import NFKC
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer

from tokenizer import DEFAULT_TOKENIZER_DIR, EOS_TOKEN, TextTokenizer, tokenizer_hash

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "tinystories" / "raw"


def iter_texts(raw_dir: Path, *, max_docs: int | None) -> list[str]:
    train_path = raw_dir / "train"
    if not train_path.is_dir():
        raise FileNotFoundError(f"{train_path} not found; run download-dataset first")
    ds = load_from_disk(str(train_path))
    texts = [row["text"] for row in ds]
    if max_docs is not None:
        texts = texts[:max_docs]
    return texts


def train_bpe_tokenizer(
    texts: list[str],
    *,
    vocab_size: int,
    output_dir: Path,
) -> TextTokenizer:
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = Tokenizer(BPE(unk_token=None))
    tokenizer.normalizer = NFKC()
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=[EOS_TOKEN],
        show_progress=True,
    )
    tokenizer.train_from_iterator(texts, trainer=trainer, length=len(texts))
    tok_path = output_dir / "tokenizer.json"
    tokenizer.save(str(tok_path))
    eos_id = tokenizer.token_to_id(EOS_TOKEN)
    if eos_id is None:
        raise RuntimeError(f"failed to register special token {EOS_TOKEN}")
    meta = {
        "vocab_size": vocab_size,
        "eos_id": eos_id,
        "special_tokens": [EOS_TOKEN],
        "train_docs": len(texts),
        "tokenizer_hash": tokenizer_hash(tok_path, vocab_size),
    }
    (output_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return TextTokenizer(tokenizer, vocab_size=vocab_size, eos_id=eos_id)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train BPE tokenizer on TinyStories")
    parser.add_argument("--vocab-size", type=int, default=8192)
    parser.add_argument("--max-docs", type=int, default=None)
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_TOKENIZER_DIR)
    args = parser.parse_args()
    texts = iter_texts(args.raw_dir, max_docs=args.max_docs)
    tok = train_bpe_tokenizer(texts, vocab_size=args.vocab_size, output_dir=args.output_dir)
    print(f"trained tokenizer vocab={tok.vocab_size} eos_id={tok.eos_id} -> {args.output_dir}")


if __name__ == "__main__":
    main()
