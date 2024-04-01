import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class Tokenizer:
    """Simple dictionary-lookup tokenizer for char level models."""
    vocab_size: int
    int_to_str: dict[int, str]
    str_to_int: dict[str, int]

    def encode(self, prompt: str) -> list[int]:
        """Encode a string prompt to integer tokens."""
        return [self.str_to_int[x] for x in prompt]

    def decode(self, tokens: list[int]) -> str:
        """Decode integer tokens to a string prompt."""
        return "".join(self.int_to_str[x] for x in tokens)

    @classmethod
    def from_pretrained(cls, path: Path | str):
        """Load a tokenizer from a JSON file."""
        path = Path(path)
        assert path.suffix == ".json", f"Only JSON files supported, found {path}"

        with path.open(mode="r", encoding="utf-8") as f:
            parameters = json.load(f)

        return cls(
            vocab_size=parameters["vocab_size"],
            int_to_str={int(k): str(v) for k, v in parameters["int_to_str"].items()},
            str_to_int={str(k): int(v) for k, v in parameters["str_to_int"].items()},
        )

    def save(self, path: Path | str) -> None:
        """Store a Tokenizer to a JSON file."""
        path = Path(path)
        assert path.suffix == ".json", f"Only JSON files supported, found {path}"

        with path.open(mode="w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=4)
