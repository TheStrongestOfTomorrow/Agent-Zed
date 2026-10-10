"""Character-level tokenizer for TinyCodeGPT (zero dependencies)."""

from typing import Dict, List

class CharTokenizer:
    """Maps characters to integer ids. Tiny on purpose: a char-level vocab of
    ~100 symbols keeps the embedding table small enough for pure-Python
    inference on low-memory devices (e.g. Termux / 4GB Android)."""

    def __init__(self, vocab: List[str], unk: str = "?"):
        if unk not in vocab:
            vocab = sorted(set(vocab) | {unk})
        self.vocab: List[str] = list(vocab)
        self.unk = unk
        self.stoi: Dict[str, int] = {ch: i for i, ch in enumerate(self.vocab)}
        self.itos: List[str] = self.vocab
        self.unk_id = self.stoi[unk]

    @classmethod
    def build(cls, text: str, unk: str = "?") -> "CharTokenizer":
        """Build a tokenizer from corpus text (deterministic sorted vocab)."""
        vocab = sorted(set(text))
        return cls(vocab, unk=unk)

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def encode(self, text: str) -> List[int]:
        unk = self.unk_id
        return [self.stoi.get(ch, unk) for ch in text]

    def decode(self, ids: List[int]) -> str:
        n = len(self.vocab)
        return "".join(self.itos[i] if 0 <= i < n else self.unk for i in ids)

    def to_dict(self) -> Dict:
        return {"type": "char", "vocab": self.vocab, "unk": self.unk}

    @classmethod
    def from_dict(cls, data: Dict) -> "CharTokenizer":
        return cls(list(data["vocab"]), unk=data.get("unk", "?"))
