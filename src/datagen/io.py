from pathlib import Path
from typing import Dict, Any, Iterable, List, Set
import json


def ensure_output_dir(path: str) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def save_config(out_dir: Path, cfg: Dict[str, Any]) -> None:
    with open(out_dir / "data_gen_config.json", "w") as f:
        json.dump(cfg, f, indent=2)


def get_existing_sample_ids(output_dir: Path) -> Set[str]:
    existing_ids: Set[str] = set()
    if output_dir.exists():
        for json_file in output_dir.glob("*.json"):
            if json_file.stem != "data_gen_config":
                existing_ids.add(json_file.stem)
    return existing_ids


def write_per_sample_json(output_dir: Path, sample_id: str, row: Dict[str, Any]) -> None:
    p = output_dir / f"{sample_id}.json"
    with open(p, "w") as f:
        json.dump(row, f, indent=2)


def load_existing_rows(output_dir: Path, existing_ids: Iterable[str]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for sid in existing_ids:
        p = output_dir / f"{sid}.json"
        if p.exists():
            with open(p, "r") as f:
                rows.append(json.load(f))
    return rows


def write_dataset_jsonl(output_dir: Path, filename: str, rows: List[Dict[str, Any]]) -> Path:
    out_path = output_dir / filename
    with open(out_path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    return out_path


