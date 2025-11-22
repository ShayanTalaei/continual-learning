"""
Score QA predictions using Gemini (Vertex AI) as a judge.

Input: a JSON list of objects with keys: question, answer (gold), pred (model output).
Output: same list with an added field `correct` (1 or 0) and optional `reason`.

Usage:
  python examples/10k/gemini_score_qa.py \
    --input data/10k/eval/tokasaurus_qa_results.json \
    --output data/10k/eval/tokasaurus_qa_results_scored.json

Credentials: looks for GCP_PROJECT / GCP_REGION / GCP_CREDENTIALS in the
environment. If `python-dotenv` is installed, it will also load a `.env` file next
to this script. If not, a minimal fallback parser reads the same file.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict
import pdb
try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - tqdm is optional
    tqdm = None

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

try:
    import vertexai
    try:
        from vertexai.generative_models import GenerativeModel, GenerationConfig
    except ImportError:
        from vertexai.preview.generative_models import GenerativeModel, GenerationConfig  # type: ignore
except ImportError:
    vertexai = None
    GenerativeModel = None  # type: ignore
    GenerationConfig = None  # type: ignore


def init_vertex_ai(project: str, region: str, credentials: str | None = None) -> None:
    if vertexai is None or GenerativeModel is None:
        raise ImportError("vertexai not installed; pip install google-cloud-aiplatform")
    if credentials and not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials
    vertexai.init(project=project, location=region)


def load_local_env(env_path: Path) -> None:
    """Minimal .env loader if python-dotenv is not available."""
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if key and val and key not in os.environ:
            os.environ[key] = val


def resolve_credentials_path(credentials: str | None, search_dirs: list[Path]) -> str | None:
    """Return an existing credentials path, searching common roots if needed."""
    if not credentials:
        return None
    cred_path = Path(credentials)
    if cred_path.is_file():
        return str(cred_path)
    for base in search_dirs:
        candidate = base / credentials
        if candidate.is_file():
            return str(candidate)
    return None


def judge_answer(model_name: str, question: str, gold: str, pred: str) -> Dict[str, Any]:
    model = GenerativeModel(model_name)

    def extract_json_block(text: str) -> str:
        """Extract the first plausible JSON object from a string."""
        import re

        # Strip fences if present
        if text.startswith("```"):
            text = text.strip("`").strip()

        # Exact parse first
        try:
            json.loads(text)
            return text
        except Exception:
            pass

        # Look for a {...} block
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            return m.group(0)
        return text

    def make_config(max_tokens: int = 64):
        """Build a GenerationConfig with JSON expectations."""
        try:
            return GenerationConfig(
                response_mime_type="application/json",
                response_schema=schema,
                max_output_tokens=max_tokens,
                temperature=0.0,
            )
        except TypeError:
            try:
                return GenerationConfig(
                    response_mime_type="application/json",
                    response_json_schema=schema,
                    max_output_tokens=max_tokens,
                    temperature=0.0,
                )
            except TypeError:
                return GenerationConfig(max_output_tokens=max_tokens, temperature=0.0)

    prompt = f"""
You are a grader. Given a question, the true answer, and a model's predicted answer, output 1 if the prediction is correct, else 0.

Respond with JSON only:
{{"correct": 1 or 0, "reason": "short justification"}}

Question: {question}
True answer: {gold}
Predicted answer: {pred}
"""

    schema = {
        "type": "object",
        "properties": {
            "correct": {"type": "integer"},
            "reason": {"type": "string"},
        },
        "required": ["correct"],
    }

    gen_config = make_config(max_tokens=64)

    def run_once(prompt_text: str):
        resp = model.generate_content([prompt_text], generation_config=gen_config)
        text = ""
        try:
            text = (resp.text or "").strip()
        except Exception:
            text = ""
        return resp, text

    resp, text = run_once(prompt)

    # Robustly extract text; safety filters sometimes return empty candidates.
    if not text:
        # Try a shallow look at candidates for debugging context
        reason = "blocked_or_empty"
        try:
            cand = resp.candidates[0]
            finish = getattr(cand, "finish_reason", "")
            reason = f"blocked_or_empty:{finish}" if finish else reason
        except Exception:
            pass
        return {"correct": 0, "reason": reason, "raw": ""}

    def parse_text(txt: str):
        txt = extract_json_block(txt)
        try:
            data = json.loads(txt)
            return int(data.get("correct", 0)), data.get("reason", ""), txt
        except Exception:
            return None, None, txt

    correct, reason, raw = parse_text(text)
    if correct is None:
        # Fallback retry with a minimal instruction to force JSON
        minimal_prompt = (
            "Return only JSON of the form {\"correct\":0 or 1, \"reason\":\"...\"}.\n"
            f"Question: {question}\nTrue answer: {gold}\nPredicted answer: {pred}"
        )
        resp, text2 = run_once(minimal_prompt)
        correct, reason, raw = parse_text(text2)
        if correct is None:
            correct, reason, raw = 0, "parse_failed", raw

    return {"correct": correct, "reason": reason, "raw": raw}


def main():
    parser = argparse.ArgumentParser(description="Score QA predictions with Gemini judge (0/1)")
    parser.add_argument("--input", required=True, help="Path to input JSON with question/answer/pred entries")
    parser.add_argument("--output", required=True, help="Where to write scored JSON")
    parser.add_argument("--model", default="gemini-2.5-flash", help="Gemini model name")
    parser.add_argument("--project", default=os.environ.get("GCP_PROJECT"), help="GCP project")
    parser.add_argument("--region", default=os.environ.get("GCP_REGION", "us-central1"), help="GCP region")
    parser.add_argument("--credentials", default=os.environ.get("GCP_CREDENTIALS"), help="Path to service account JSON (or set GCP_CREDENTIALS in .env)")
    args = parser.parse_args()

    env_candidates = [
        Path(__file__).parent / ".env",
        Path(__file__).parent.parent.parent / ".env",
        Path(__file__).parent.parent.parent / "cartridges" / ".env",
    ]
    search_dirs = [
        Path.cwd(),
        Path(__file__).parent,
        Path(__file__).parent.parent,
        Path(__file__).parent.parent.parent,
        Path(__file__).parent.parent.parent / "cartridges",
    ]
    for p in env_candidates:
        if load_dotenv is not None:
            load_dotenv(p, override=False)
        else:
            load_local_env(p)

    project = args.project or os.environ.get("GCP_PROJECT")
    region = args.region or os.environ.get("GCP_REGION")
    credentials = resolve_credentials_path(args.credentials or os.environ.get("GCP_CREDENTIALS"), search_dirs)

    if not project:
        tried = ", ".join(str(p) for p in env_candidates)
        raise ValueError(
            "GCP project not set (use --project, GCP_PROJECT env, or add .env); "
            f"tried: {tried}"
        )
    if credentials is None:
        raise ValueError(
            "GCP credentials file not found; set --credentials / GCP_CREDENTIALS or place gcp.json in one of: "
            + ", ".join(str(d) for d in search_dirs)
        )

    init_vertex_ai(project, region, credentials)

    with open(args.input, "r", encoding="utf-8") as f:
        records = json.load(f)

    iterator = tqdm(records, desc="Scoring QA", unit="qa") if tqdm else records

    scored = []
    for rec in iterator:
        q = rec.get("question", "")
        gold = rec.get("answer", "")
        pred = rec.get("pred", "")
        verdict = judge_answer(args.model, q, gold, pred)
        rec.update(verdict)
        scored.append(rec)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(scored, f, indent=2, ensure_ascii=False)

    print(f"Wrote scored results to {out_path} ({len(scored)} rows)")


if __name__ == "__main__":
    main()
