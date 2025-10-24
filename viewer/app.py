import json
from pathlib import Path
from typing import Dict, List, Tuple
import html

import streamlit as st


DEFAULT_DATA_DIR = "/mnt/data/shayan_memory/finer_data_gen_shuffled_256x100"


def list_triplet_files(data_dir: str) -> List[Path]:
    data_path = Path(data_dir)
    if not data_path.exists():
        return []
    return sorted(
        [p for p in data_path.iterdir() if p.is_file() and p.name.startswith("triplet_") and p.suffix == ".json"],
        key=lambda p: p.name,
    )


def parse_triplet_id(file_name: str) -> Tuple[int, int]:
    # Expected pattern: triplet_{triplet_id}_shuffle_{shuffle}.json
    # Fallback: return (-1, -1) on failure
    try:
        base = file_name.replace("triplet_", "").replace(".json", "")
        left, right = base.split("_shuffle_")
        return int(left), int(right)
    except Exception:
        return -1, -1


def load_json_head_safe(path: Path, max_bytes: int = 2_000_000) -> Dict:
    # Load safely to handle very large files
    try:
        size = path.stat().st_size
        if size > max_bytes:
            with path.open("r") as f:
                # Stream read but still need valid JSON; fall back to full read
                # because files appear to be single JSON objects.
                content = f.read()
        else:
            content = path.read_text()
        return json.loads(content)
    except Exception as e:
        return {"__error__": str(e)}


def group_by_triplet(files: List[Path]) -> Dict[int, List[Path]]:
    groups: Dict[int, List[Path]] = {}
    for p in files:
        triplet_id, _ = parse_triplet_id(p.name)
        if triplet_id < 0:
            continue
        groups.setdefault(triplet_id, []).append(p)
    # Sort each group's files by shuffle index
    for triplet_id, paths in groups.items():
        groups[triplet_id] = sorted(paths, key=lambda p: parse_triplet_id(p.name)[1])
    return groups


def render_horizontal_scroller(children: List[Tuple[str, str]]):
    # children: list of (title, content)
    # Use CSS grid for horizontal scroll of equally sized panels
    st.markdown(
        """
        <style>
        .hscroll {
            display: flex;
            flex-direction: row;
            overflow-x: auto;
            padding-bottom: 8px;
            gap: 0;
            -webkit-overflow-scrolling: touch;
        }
        .panel {
            flex: 0 0 calc(100% / 3);
            height: 60vh;
            display: flex;
            flex-direction: column;
            background: transparent;
            border: none;
            padding: 0 8px 0 0; /* minimal right padding for readability */
            box-sizing: border-box;
        }
        .panel-title {
            font-weight: 600;
            margin-bottom: 6px;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        .panel-body {
            overflow: auto;
            flex: 1;
            white-space: pre-wrap;
            word-break: break-word;
            overflow-wrap: anywhere;
        }
        .panel-body pre {
            margin: 0;
            font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
            font-size: 0.9rem;
            line-height: 1.4;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="hscroll">', unsafe_allow_html=True)
    for title, content in children:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown(f'<div class="panel-title">{title}</div>', unsafe_allow_html=True)
        escaped = html.escape(content)
        st.markdown(f'<div class="panel-body"><pre>{escaped}</pre></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


def main():
    st.set_page_config(page_title="Triplet Output Viewer", layout="wide")
    st.title("Triplet Output Viewer")

    data_dir = st.text_input("Data directory", value=DEFAULT_DATA_DIR)

    files = list_triplet_files(data_dir)
    if not files:
        st.warning("No triplet_*.json files found.")
        return

    groups = group_by_triplet(files)
    all_triplet_ids = sorted(groups.keys())

    # Select triplet id
    default_index = 0
    if 2 in groups:  # convenience default if exists
        default_index = all_triplet_ids.index(2)
    triplet_id = st.selectbox("Triplet id", options=all_triplet_ids, index=default_index)

    st.caption(f"Found {len(groups[triplet_id])} files for triplet {triplet_id}")

    # Build panels for selected triplet
    panels: List[Tuple[str, str]] = []
    for p in groups[triplet_id]:
        data = load_json_head_safe(p)
        if "__error__" in data:
            content = f"Error reading file: {data['__error__']}"
        else:
            # Prefer output_message; fallback to entire JSON pretty
            msg = data.get("output_message")
            if isinstance(msg, str) and len(msg) > 0:
                content = msg
            else:
                content = json.dumps(data, indent=2)
        title = p.name
        panels.append((title, content))

    render_horizontal_scroller(panels)


if __name__ == "__main__":
    main()


