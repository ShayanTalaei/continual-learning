"""
Simple script to chat with one or more cartridges via Tokasaurus, using
cartridge specs loaded from a YAML config file.

Example YAML additions (e.g., in examples/10k/eval_perplexity_config.yaml):

tokasaurus_url: "http://localhost:10210"
cartridges:
  - id: "amd_10k"
    source: "local"

You can also define named groups:

amd_cartridges:
  - id: "amd_10k"
    source: "local"
pepsi_cartridges:
  - id: "pepsi_10k"
    source: "local"

Then run:

  python examples/10k/tokasaurus_chat_from_config.py \\
    --config examples/10k/eval_perplexity_config.yaml \\
    --group amd
"""

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
import yaml


def load_config(config_path: str) -> Dict[str, Any]:
    """Load a YAML config file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_cartridges_from_config(
    cfg: Dict[str, Any],
    group: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Load cartridge specs from config.

    Precedence:
      1) If group is provided, look for `<group>_cartridges` key.
      2) Otherwise, or as fallback, use top-level `cartridges` key.
    """
    if group:
        if group == 'both':
            key1 = "amd_gemini"
            key2 = "pepsi_gemini"
            cartridge_path1 = cfg.get(key1)
            cartridge_path2 = cfg.get(key2)
            cartridges = [
            {
                'id': key1,
                'source':'local'
            },
            {
                'id': key2,
                'source':'local'
            }]
        else:
            key = f"{group}"
            cartridges = [{
                'id': key,
                'source':'local'
            }]
        if cartridges:
            return cartridges

    cartridges = cfg.get("cartridges", [])
    if not isinstance(cartridges, list):
        raise ValueError("Expected 'cartridges' to be a list of {id, source} objects.")
    return cartridges


def get_tokasaurus_url(cfg: Dict[str, Any]) -> str:
    """Determine Tokasaurus base URL from config or environment."""
    # 1) YAML key
    url = cfg.get("tokasaurus_url")
    # 2) Environment variable
    if not url:
        url = os.environ.get("TOKASAURUS_URL")
    # 3) Default
    if not url:
        url = "http://localhost:10210"
    return url.rstrip("/")


def chat_with_tokasaurus(
    base_url: str,
    cartridges: List[Dict[str, Any]],
    question: str,
    max_tokens: int = 256,
    system_prompt: str = "You are a helpful financial analyst assistant.",
    endpoint: str = "/custom/cartridge/chat/completions",
) -> str:
    """Send a single question to Tokasaurus with the given cartridge specs.

    Default endpoint matches the path used in evaluate_financebench_qa.py. If
    your server expects the OpenAI-style path, pass endpoint="/v1/cartridge/chat/completions".
    """
    url = base_url + endpoint

    payload: Dict[str, Any] = {
        "model": "meta-llama/Llama-3.2-3B-Instruct",
        "messages": [
            # {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ],
        "cartridges":cartridges,
        "max_tokens": max_tokens,
    }
    print(payload)

    resp = requests.post(url, json=payload, timeout=600)
    resp.raise_for_status()
    data = resp.json()

    # Tokasaurus is OpenAI-compatible; expect choices[0].message.content
    try:
        return data["choices"][0]["message"]["content"]
    except Exception:
        # Fallback: print full response for debugging
        raise RuntimeError(f"Unexpected response format from Tokasaurus: {json.dumps(data, indent=2)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interactive chat with cartridges via Tokasaurus using a YAML config."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config (e.g., examples/10k/eval_perplexity_config.yaml).",
    )
    parser.add_argument(
        "--group",
        type=str,
        default=None,
        help=(
            "Optional cartridge group name. If provided, looks for '<group>_cartridges' "
            "in the YAML; otherwise uses top-level 'cartridges'."
        ),
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Maximum tokens to generate per answer.",
    )
    parser.add_argument(
        "--endpoint",
        type=str,
        default="/custom/cartridge/chat/completions",
        help="Tokasaurus endpoint path (default: /custom/cartridge/chat/completions).",
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        default="You are a helpful financial analyst assistant.",
        help="System prompt to use in the chat.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    base_url = get_tokasaurus_url(cfg)
    cartridges = load_cartridges_from_config(cfg, group=args.group)

    print(f"Tokasaurus URL: {base_url}")
    print(f"Loaded {len(cartridges)} cartridge spec(s) from config.")
    if cartridges:
        print("Cartridges:")
        print(json.dumps(cartridges, indent=2))
    else:
        print("No cartridges configured; chatting with base model only.")

    print("\nType your questions below. Empty line or 'exit' to quit.\n")

    while True:
        try:
            question = input("User> ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not question or question.lower() in {"exit", "quit"}:
            break

        try:
            answer = chat_with_tokasaurus(
                base_url=base_url,
                cartridges=cartridges,
                question=question,
                max_tokens=args.max_tokens,
                system_prompt=args.system_prompt,
                endpoint=args.endpoint,
            )
        except Exception as e:
            print(f"[ERROR] {e}")
            continue

        print("Cartridge>", answer.strip(), "\n")


if __name__ == "__main__":
    main()
