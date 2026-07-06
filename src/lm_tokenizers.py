"""Tokenizer helpers for language-model benchmarks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


def _get_tiktoken_encoding(name: str = "gpt2"):
    try:
        import tiktoken
    except ImportError as exc:
        raise RuntimeError(
            "GPT-2 BPE tokenization requires the 'tiktoken' package. "
            "Install code/requirements.txt to enable BPE-tokenized LM runs."
        ) from exc
    return tiktoken.get_encoding(name)


@dataclass(frozen=True, slots=True)
class TokenizerMetadata:
    name: str
    vocab_size: int
    token_unit: str


class ByteTokenizer:
    def metadata(self) -> TokenizerMetadata:
        return TokenizerMetadata(name="byte", vocab_size=256, token_unit="byte")

    def encode_bytes(self, payload: bytes) -> list[int]:
        return list(payload)

    def encode_text(self, text: str) -> list[int]:
        return self.encode_bytes(text.encode("utf-8"))

    def encode_file(self, path: str | Path) -> list[int]:
        return self.encode_bytes(Path(path).read_bytes())

    def decode_tokens(self, tokens: list[int]) -> str:
        return bytes(int(token) % 256 for token in tokens).decode("utf-8", errors="replace")


class GPT2BPETokenizer:
    def __init__(self) -> None:
        self.encoding = _get_tiktoken_encoding("gpt2")

    def metadata(self) -> TokenizerMetadata:
        return TokenizerMetadata(name="gpt2_bpe", vocab_size=int(self.encoding.n_vocab), token_unit="bpe")

    def encode_text(self, text: str) -> list[int]:
        return list(self.encoding.encode_ordinary(text))

    def encode_file(self, path: str | Path) -> list[int]:
        return self.encode_text(Path(path).read_text(encoding="utf-8"))

    def decode_tokens(self, tokens: list[int]) -> str:
        return self.encoding.decode(list(int(token) for token in tokens))


def build_tokenizer(name: str):
    key = str(name).strip().lower()
    if key == "byte":
        return ByteTokenizer()
    if key in {"gpt2", "gpt2_bpe"}:
        return GPT2BPETokenizer()
    raise ValueError(f"Unsupported tokenizer {name!r}.")
