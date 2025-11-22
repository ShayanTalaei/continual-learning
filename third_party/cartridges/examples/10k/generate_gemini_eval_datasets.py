"""
Generate GenConvo-style evaluation QA datasets for AMD and PepsiCo 10-Ks using Gemini (Vertex AI).

For each company document (AMD / PepsiCo) and each seed prompt type, this script:
  - Loads the full 10-K text as a single context.
  - Asks Gemini (via Vertex AI) to generate N unique question–answer pairs
    that follow the given seed prompt template.
  - Saves all QAs to JSON files:
        data/10k/eval/amd_qa_gemini.json
        data/10k/eval/pepsi_qa_gemini.json

Environment:
  - Expects Vertex AI credentials and project configuration to be available, e.g.:
        GCP_PROJECT, GCP_REGION, GCP_CREDENTIALS
    Optionally, `GOOGLE_APPLICATION_CREDENTIALS` can be set directly.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Any

from tqdm.auto import tqdm

try:
    # Vertex AI Python SDK (google-cloud-aiplatform >= 1.38)
    import vertexai
    try:
        # Newer versions
        from vertexai.generative_models import GenerativeModel, GenerationConfig
    except ImportError:
        # Fallback for preview namespace
        from vertexai.preview.generative_models import (  # type: ignore
            GenerativeModel,
            GenerationConfig,
        )
except ImportError as e:  # pragma: no cover - import error surfaced at runtime
    vertexai = None
    GenerativeModel = None  # type: ignore[assignment]
    GenerationConfig = None  # type: ignore[assignment]

THIS_DIR = Path(__file__).parent
CARTRIDGES_DIR = THIS_DIR.parent.parent
DATA_DIR = CARTRIDGES_DIR / "data" / "10k"
EVAL_DIR = DATA_DIR / "eval"


# Seed prompt templates taken from the user's specification (GenConvo-style)
SEED_TEMPLATES: Dict[str, str] = {
    "factual": """Factual Prompt Template
Please generate a question to test someone’s ability to remember factual details from the document. The
answer should be a few tokens long and be a factual detail from the statement, such as a number, entity,
date, title, or name.
This question should not be common knowledge: instead, it should be something that is only answerable
via information in the document.""",
    "knowledge": """Knowledge Prompt Template
Please generate a question that requires combining information mentioned both inside and outside the
document.
This question should require using a fact from the document and also a fact that you are confident about,
but is not mentioned in the document. For instance: - What are the founding dates of the companies
that got acquired this year? This is a good question because the names of the acquired companies are
mentioned in the document and the founding dates are not mentioned. - What is the name of the CEO’s spouse? This is a good question because the name of the CEO is mentioned in the document and the
spouse’s name is not mentioned.
The answer should be a fact that is a few tokens long such as a number, entity, date, title, or name.""",
    "disjoint": """Disjoint Prompt Template
Please generate a multi-hop question that tests someone’s ability to use factual information mentioned
in at least two very different sub-sections of the document.
This question shouldn’t be a standard question about this kind of document. Instead, it should ask
about two particularly disconnected ideas, like comparing information about the amount of owned space
for the company headquarters with the amount of dollars of estimated liability or comparing the revenue
number with the number of employees.
This question should also test one’s ability to do retrieval: do not give away part of the answer in
the question. Ensure that for one to get the correct answer to the question, they need to understand
the document.
The answer should be a short: for example, a number, entity, date, title, or name.""",
    "synthesize": """Synthesize Prompt Template
Please generate a question that requires synthesizing and aggregating information in the document.
For instance, you could ask someone to summarize a page of the document, list all the key competitors
mentioned in the document, or summarize the company’s business mode""",
    "structure": """Structure Prompt Template
Please generate a question that requires understanding the structure of the document.
This question should be more about the structure of the document, rather than the precise statement
details. For instance, you could ask someone to list the titles of all the sections in the document,
describe the document structure, report the total number of pages, ask which section amongst two sections
comes first, or report the section with the largest number of tables.""",
    "creative": """Creative Prompt Template
Please generate a question about the document to test someone’s ability to comprehend the content of the
document. This question specifically should be focused on their ability to generalize the information
about the document to a strange question of sorts.
This question shouldn’t be a standard question about this kind of document, it should ask to do something
abnormal and creative, like writing a poem about a financial document.""",
    "counting": """Counting Prompt Template
Please generate a question that requires counting how frequently different events occur in the document.
This question should be about statistical properties of the document, rather than the statement details.
For instance, you could ask someone to count the number of times the word "million" is mentioned or
count the length of the shortest section title.
The answer should be a number.""",
    "reasoning": """Reasoning Prompt Template
Please generate a question that requires mathematical reasoning over the values in the document.
This question should require going beyond the facts directly mentioned in the statement, such as asking
to compute the percentage increase in revenue between two years, find the largest expense category, or
calculate difference in profit between two years.
The answer should be a number.""",
}


def _load_local_env() -> None:
    """Load key=value pairs from .env next to this script into os.environ (if present)."""
    env_path = THIS_DIR / ".env"
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        key = key.strip()
        val = val.strip()
        # Strip surrounding quotes if present
        if len(val) >= 2 and val[0] in {"'", '"'} and val[-1] == val[0]:
            val = val[1:-1]
        if key and val and key not in os.environ:
            os.environ[key] = val


def init_vertex_ai(project: str, region: str, credentials_path: str | None = None) -> None:
    """Initialize Vertex AI client using environment / provided credentials."""
    if vertexai is None or GenerativeModel is None:
        raise ImportError(
            "vertexai is not installed. Please `pip install google-cloud-aiplatform` "
            "and ensure the Vertex AI SDK is available."
        )

    # Allow using a custom credentials JSON if provided
    if credentials_path and not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path

    vertexai.init(project=project, location=region)


def build_prompt(
    document_text: str,
    seed_name: str,
    seed_template: str,
    num_questions: int,
) -> str:
    """Compose a single prompt to ask Gemini for N QA pairs for a given seed."""
    return f"""
You are an expert question writer creating evaluation data for long 10-K financial documents.

You will be given:
1. The full text of a single 10-K document.
2. A seed prompt template that describes the *type* of question to ask.

Your task:
- Using ONLY the information in the document (except where the seed explicitly asks to combine with outside knowledge, e.g. the Knowledge prompt),
  generate **{num_questions} unique question–answer pairs** that follow the seed template.
- Questions must be non-duplicated, specific, and diverse.
- Each answer should be short (a few tokens), e.g., a number, entity, date, title, or name, unless the seed suggests otherwise.

Seed prompt type: {seed_name}

Seed prompt template:
\"\"\"{seed_template}\"\"\"

Document:
\"\"\"{document_text}\"\"\"

Output format (JSON only, no extra text):
{{
  "pairs": [
    {{"question": "...", "answer": "..."}},
    ...
  ]
}}

Constraints:
- Return exactly a JSON object with the top-level key "pairs".
- Each element in "pairs" must have string fields "question" and "answer".
- Do NOT include any explanatory text, markdown fences, or commentary outside the JSON.
"""


def call_gemini(
    model_name: str,
    prompt: str,
    max_output_tokens: int = 4096,
    temperature: float = 0.3,
) -> str:
    """Call Gemini via Vertex AI and return the raw text response.

    Tries to request structured JSON output using response_mime_type/JSON schema
    when supported by the installed Vertex AI SDK, and falls back gracefully
    otherwise.
    """
    model = GenerativeModel(model_name)

    # Simple JSON schema for the expected response:
    # {"pairs": [{"question": "...", "answer": "..."}, ...]}
    qa_schema: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "pairs": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "question": {"type": "string"},
                        "answer": {"type": "string"},
                    },
                    "required": ["question", "answer"],
                },
            }
        },
        "required": ["pairs"],
    }

    # Try with structured JSON config first, then fall back if unsupported
    gen_config = None
    # try:
    gen_config = GenerationConfig(
        max_output_tokens=max_output_tokens,
        temperature=temperature,
        response_mime_type="application/json",
        response_schema=qa_schema,  # newer Vertex AI SDKs
        )
    # except TypeError:
    #     try:
    #         gen_config = GenerationConfig(
    #             max_output_tokens=max_output_tokens,
    #             temperature=temperature,
    #             response_mime_type="application/json",
    #             response_json_schema=qa_schema,  # some client variants
    #         )
    #     except TypeError:
    #         gen_config = GenerationConfig(
    #             max_output_tokens=max_output_tokens,
    #             temperature=temperature,
    #         )

    response = model.generate_content(
        [prompt],
        generation_config=gen_config,
    )
    return response.text


def parse_qa_pairs(raw_text: str, expected_num: int) -> List[Dict[str, str]]:
    """Parse Gemini output into a list of QA dicts, enforcing uniqueness and count.

    Expected JSON template:
      - Either an object: {"pairs": [{"question": "...", "answer": "..."} , ...]}
      - Or directly a list: [{"question": "...", "answer": "..."}, ...]

    Raises on malformed JSON or mismatched template so the caller can retry.
    """
    text = raw_text.strip()
    # Strip common accidental wrappers like markdown fences
    if text.startswith("```"):
        text = text.strip("`").strip()

    data: Any
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # Try to recover by extracting the first JSON object substring
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        data = json.loads(text[start : end + 1])

    if isinstance(data, dict):
        if "pairs" not in data or not isinstance(data["pairs"], list):
            raise ValueError("JSON object must have key 'pairs' mapping to a list.")
        pairs = data["pairs"]
    elif isinstance(data, list):
        pairs = data
    else:
        raise ValueError("Unexpected JSON structure: expected object with 'pairs' or a list.")

    cleaned: List[Dict[str, str]] = []
    seen_questions = set()
    for item in pairs:
        if not isinstance(item, dict):
            continue
        q = str(item.get("question", "")).strip()
        a = str(item.get("answer", "")).strip()
        if not q or not a:
            continue
        if q in seen_questions:
            continue
        seen_questions.add(q)
        cleaned.append({"question": q, "answer": a})
        if len(cleaned) >= expected_num:
            break

    return cleaned


def generate_for_company(
    company: str,
    doc_path: Path,
    model_name: str,
    num_per_seed: int,
    temperature: float,
    max_output_tokens: int,
) -> List[Dict[str, Any]]:
    """Generate QA pairs for a single company document across all seed templates."""
    if not doc_path.exists():
        raise FileNotFoundError(f"Document not found: {doc_path}")

    document_text = doc_path.read_text(encoding="utf-8")

    all_examples: List[Dict[str, Any]] = []
    for seed_name, template in tqdm(
        SEED_TEMPLATES.items(),
        desc=f"Generating for {company}",
        total=len(SEED_TEMPLATES),
    ):
        base_prompt = build_prompt(
            document_text=document_text,
            seed_name=seed_name,
            seed_template=template,
            num_questions=num_per_seed,
        )

        # Retry Gemini generation / JSON parsing up to 12 times for this seed type.
        # On each failure, we append a short self-correction hint (with truncated
        # previous output + error) to the prompt to nudge the model away from
        # repeating the same mistake.
        pairs: List[Dict[str, str]] | None = None
        last_err: Exception | None = None
        last_raw: str | None = None
        extra_hint = ""
        max_attempts = 12
        for attempt in range(max_attempts):
            try:
                prompt = base_prompt + extra_hint
                raw = call_gemini(
                    model_name=model_name,
                    prompt=prompt,
                    max_output_tokens=max_output_tokens,
                    temperature=temperature,
                )
                last_raw = raw
                pairs = parse_qa_pairs(raw, expected_num=num_per_seed)
                break
            except Exception as e:  # noqa: PERF203
                last_err = e
                print(
                    f"[WARN] {company} / {seed_name}: "
                    f"generation/parse attempt {attempt + 1}/{max_attempts} failed: {e}"
                )
                # Build a brief correction hint for the next attempt
                raw_preview = (last_raw or "").replace("\n", " ")[:500]
                extra_hint = (
                    "\n\nNOTE: The previous response could not be parsed as valid JSON "
                    f"and resulted in the following error: {type(e).__name__}: {e}. "
                    "Here is a truncated version of the invalid response:\n"
                    f"```{raw_preview}```\n"
                    "Please regenerate a NEW response that strictly follows the JSON template "
                    'with a top-level "pairs" list and no trailing commas or extra fields.'
                )
        if pairs is None:
            print(
                f"[ERROR] Skipping seed '{seed_name}' for company '{company}' "
                f"after {max_attempts} failed attempts. Last error: {last_err}"
            )
            if last_raw is not None:
                preview = last_raw.replace("\n", " ")[:2000]
                print(f"[DEBUG] Last raw response (truncated): {preview}")
            continue

        for idx, qa in enumerate(pairs):
            all_examples.append(
                {
                    "company": company,
                    "seed_type": seed_name,
                    "seed_index": idx,
                    "seed_template": template,
                    "question": qa["question"],
                    "answer": qa["answer"],
                    "model": model_name,
                    "document_path": str(doc_path),
                }
            )

    return all_examples


def main() -> None:
    # Ensure local .env (in examples/10k) is loaded before reading defaults
    _load_local_env()

    parser = argparse.ArgumentParser(
        description="Generate GenConvo-style eval QA datasets for AMD/Pepsi 10-Ks using Gemini (Vertex AI)."
    )
    parser.add_argument(
        "--company",
        choices=["amd", "pepsi", "both"],
        default="both",
        help="Which company document(s) to process.",
    )
    parser.add_argument(
        "--num-per-seed",
        type=int,
        default=16,
        help="Number of unique QA pairs to generate per seed prompt type.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gemini-2.5-pro",
        help="Vertex AI Gemini model name.",
    )
    parser.add_argument(
        "--project",
        type=str,
        default=os.environ.get("GCP_PROJECT"),
        help="GCP project ID (default: from GCP_PROJECT env var).",
    )
    parser.add_argument(
        "--region",
        type=str,
        default=os.environ.get("GCP_REGION", "us-central1"),
        help="GCP region for Vertex AI (default: us-central1 or GCP_REGION env var).",
    )
    parser.add_argument(
        "--credentials",
        type=str,
        default=os.environ.get("GCP_CREDENTIALS", str(THIS_DIR / "gcp.json")),
        help="Path to service account JSON (default: from GCP_CREDENTIALS env var or local gcp.json).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature for Gemini.",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=4096,
        help="Maximum tokens Gemini can output for a single call.",
    )

    args = parser.parse_args()

    if not args.project:
        raise ValueError("GCP project must be provided via --project or GCP_PROJECT env var.")

    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    init_vertex_ai(project=args.project, region=args.region, credentials_path=args.credentials)

    companies = []
    if args.company in ("amd", "both"):
        companies.append(
            ("amd", DATA_DIR / "amd_10k_financebench.txt", EVAL_DIR / "amd_qa_gemini.json")
        )
    if args.company in ("pepsi", "both"):
        companies.append(
            ("pepsi", DATA_DIR / "pepsi_10k_financebench.txt", EVAL_DIR / "pepsi_qa_gemini.json")
        )

    for company, doc_path, out_path in companies:
        examples = generate_for_company(
            company=company,
            doc_path=doc_path,
            model_name=args.model,
            num_per_seed=args.num_per_seed,
            temperature=args.temperature,
            max_output_tokens=args.max_output_tokens,
        )
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(examples, f, indent=2, ensure_ascii=False)
        print(f"Saved {len(examples)} QA pairs for {company} to {out_path}")


if __name__ == "__main__":
    main()
