import argparse
import json
import random
import string
from pathlib import Path
from typing import Dict, List


def build_random_mapping(num_chars: int = 26) -> Dict[str, str]:
    letters = list(string.ascii_lowercase)[:num_chars]
    shuffled = letters.copy()
    random.shuffle(shuffled)
    return {src: dst for src, dst in zip(letters, shuffled)}


def apply_mapping(text: str, mapping: Dict[str, str]) -> str:
    return "".join(mapping.get(ch, ch) for ch in text)


def generate_samples(n: int, length: int, mapping: Dict[str, str]) -> List[Dict[str, str]]:
    letters = string.ascii_lowercase
    rows: List[Dict[str, str]] = []
    for _ in range(n):
        s = "".join(random.choice(letters) for _ in range(length))
        t = apply_mapping(s, mapping)
        rows.append({"question": s, "answer": t})
    return rows


def save_as_jsonl(rows: List[Dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, required=True)
    parser.add_argument("--num_chars", type=int, required=False, default=26)
    parser.add_argument("--string_length", type=int, required=True)
    parser.add_argument("--save_path", type=str, required=True)
    args = parser.parse_args()

    mapping = build_random_mapping(args.num_chars)
    rows = generate_samples(args.num_samples, args.string_length, mapping)

    save_path = Path(args.save_path)
    save_as_jsonl(rows, save_path)

    # Also save mapping for reference
    with (save_path.parent / "mapping.json").open("w") as f:
        json.dump(mapping, f, indent=2)


if __name__ == "__main__":
    main()


