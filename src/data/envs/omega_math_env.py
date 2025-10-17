from typing import Dict, Any, Optional, List, Tuple, Literal
from logging import Logger, getLogger

from pydantic import BaseModel
import json

from datasets import load_dataset

from src.data.env import EnvDataset, EnvDatasetConfig, Environment
from src.data.envs.qa_env import QAEnv
from src.lm.language_model import LanguageModel, LMConfig
from src.lm.lm_factory import get_lm_client
from src.utils import logger as jsonlogger
import re

import sympy as sp


def _strip_latex_markers(text: str) -> str:
    s = (text or "").strip()
    if s.startswith("$") and s.endswith("$"):
        s = s[1:-1].strip()
    if s.startswith("\\boxed{") and s.endswith("}"):
        s = s[len("\\boxed{"):-1].strip()
    return s


def _extract_boxed(text: str) -> str:
    if "\\boxed{" in text:
        try:
            return text.split("\\boxed{")[1].split("}")[0].strip()
        except Exception:
            return text
    return text


def _normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def _try_parse_number(text: str) -> Optional[float]:
    s = (text or "").replace(",", "").strip()
    if not s:
        return None
    try:
        return float(s)
    except Exception:
        pass
    if sp is not None:
        try:
            expr = sp.sympify(s)
            if expr.is_number:
                return float(expr.evalf())
        except Exception:
            return None
    return None


def _try_parse_tuple(text: str) -> Optional[List[float]]:
    s = _strip_latex_markers(text)
    if not ((s.startswith("(") and s.endswith(")")) or (s.startswith("[") and s.endswith("]"))):
        return None
    parts = [p.strip() for p in s[1:-1].split(",") if p.strip()]
    vals: List[float] = []
    for p in parts:
        v = _try_parse_number(p)
        if v is None:
            return None
        vals.append(v)
    return vals


def _try_parse_set(text: str) -> Optional[List[float]]:
    s = _strip_latex_markers(text)
    if not (s.startswith("{") and s.endswith("}")):
        return None
    parts = [p.strip() for p in s[1:-1].split(",") if p.strip()]
    vals: List[float] = []
    for p in parts:
        v = _try_parse_number(p)
        if v is None:
            return None
        vals.append(v)
    return sorted(vals)


def _try_parse_matrix(text: str) -> Optional[List[List[float]]]:
    s = _strip_latex_markers(text)
    if not (s.startswith("[[") and s.endswith("]]")):
        return None
    try:
        raw = eval(s, {"__builtins__": {}})
        if not isinstance(raw, list):
            return None
        mat: List[List[float]] = []
        for row in raw:
            if not isinstance(row, list):
                return None
            vals: List[float] = []
            for x in row:
                v = _try_parse_number(str(x))
                if v is None:
                    return None
                vals.append(v)
            mat.append(vals)
        return mat
    except Exception:
        return None


def _eq_num(a: float, b: float, tol: float = 1e-6) -> bool:
    return abs(a - b) <= tol


def _eq_vec(a: List[float], b: List[float], tol: float = 1e-6) -> bool:
    return len(a) == len(b) and all(_eq_num(x, y, tol) for x, y in zip(a, b))


def _eq_set(a: List[float], b: List[float], tol: float = 1e-6) -> bool:
    if len(a) != len(b):
        return False
    used = [False] * len(b)
    for x in a:
        ok = False
        for j, y in enumerate(b):
            if not used[j] and _eq_num(x, y, tol):
                used[j] = True
                ok = True
                break
        if not ok:
            return False
    return True


def _eq_mat(a: List[List[float]], b: List[List[float]], tol: float = 1e-6) -> bool:
    return len(a) == len(b) and all(_eq_vec(ra, rb, tol) for ra, rb in zip(a, b))


def _expr_equiv(a: str, b: str) -> bool:
    if sp is None:
        return False
    try:
        ea = sp.simplify(sp.sympify(a))
        eb = sp.simplify(sp.sympify(b))
        return bool(sp.simplify(ea - eb) == 0)
    except Exception:
        return False


def evaluate_answer(
    predicted: str,
    target: str,
    mode: str = "auto",
    tol: float = 1e-6,
    expect_boxed: bool = False,
) -> Dict[str, Any]:
    pred_raw = predicted or ""
    if expect_boxed:
        pred_raw = _extract_boxed(pred_raw)
    pred_norm = _strip_latex_markers(_normalize_ws(pred_raw))
    tgt_norm = _strip_latex_markers(_normalize_ws(target or ""))

    def resp(correct: bool, msg: str, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        # Create descriptive message similar to qa_env.py
        if correct:
            message = "Correct!"
        else:
            message = f"Incorrect due to {msg}! The correct answer is {target}."
        
        out = {"score": 1 if correct else 0, "target": target, "message": message}
        if extra:
            out["extra"] = extra
        return out

    if mode == "normalized_exact":
        return resp(pred_norm == tgt_norm, "normalized-exact")
    if mode == "numeric_tol":
        ap = _try_parse_number(pred_norm)
        at = _try_parse_number(tgt_norm)
        if ap is None or at is None:
            return resp(False, "numeric-parse-failed")
        return resp(_eq_num(ap, at, tol), "numeric-compare", {"pred": ap, "tgt": at})
    if mode == "tuple_tol":
        vp = _try_parse_tuple(pred_norm)
        vt = _try_parse_tuple(tgt_norm)
        if vp is None or vt is None:
            return resp(False, "tuple-parse-failed")
        return resp(_eq_vec(vp, vt, tol), "tuple-compare", {"pred": vp, "tgt": vt})
    if mode == "set_tol":
        spred = _try_parse_set(pred_norm)
        stgt = _try_parse_set(tgt_norm)
        if spred is None or stgt is None:
            return resp(False, "set-parse-failed")
        return resp(_eq_set(spred, stgt, tol), "set-compare", {"pred": spred, "tgt": stgt})
    if mode == "matrix_tol":
        mpred = _try_parse_matrix(pred_norm)
        mtgt = _try_parse_matrix(tgt_norm)
        if mpred is None or mtgt is None:
            return resp(False, "matrix-parse-failed")
        return resp(_eq_mat(mpred, mtgt, tol), "matrix-compare")
    if mode == "expr_equiv":
        return resp(_expr_equiv(pred_norm, tgt_norm), "expr-equivalence")

    # auto mode inference from target
    mt = _try_parse_matrix(tgt_norm)
    if mt is not None:
        mpred = _try_parse_matrix(pred_norm)
        if mpred is None:
            return resp(False, "matrix-parse-failed")
        return resp(_eq_mat(mpred, mt, tol), "matrix-compare")

    stgt = _try_parse_set(tgt_norm)
    if stgt is not None:
        spred = _try_parse_set(pred_norm)
        if spred is None:
            return resp(False, "set-parse-failed")
        return resp(_eq_set(spred, stgt, tol), "set-compare")

    vt = _try_parse_tuple(tgt_norm)
    if vt is not None:
        vp = _try_parse_tuple(pred_norm)
        if vp is None:
            return resp(False, "tuple-parse-failed")
        return resp(_eq_vec(vp, vt, tol), "tuple-compare")

    at = _try_parse_number(tgt_norm)
    if at is not None:
        ap = _try_parse_number(pred_norm)
        if ap is None:
            return resp(False, "numeric-parse-failed")
        return resp(_eq_num(ap, at, tol), "numeric-compare")

    if sp is not None and _expr_equiv(pred_norm, tgt_norm):
        return resp(True, "expr-equivalence")
    return resp(pred_norm == tgt_norm, "normalized-exact")


class OmegaMathEnv(QAEnv):
    """Omega Math environment for single-turn QA-style problems.

    Inherits reset/step from QAEnv and overrides evaluation to support
    normalized exact matching and numeric comparison with tolerance.
    """

    def __init__(
        self,
        env_id: str,
        env_type: str,
        question: str,
        answer: str,
        metadata: Dict[str, Any],
        instruction_template: Optional[str] = None,
        logger: Optional[Logger] = None,
        eval_mode: str = "auto",
        eval_tolerance: float = 1e-6,
        expect_boxed: bool = False,
        feedback_lm: Optional[LanguageModel] = None,
        enable_llm_feedback: bool = True,
        output_type: Literal["text", "json"] = "text",
        feedback_type: Literal[
            "final_answer",
            "ground_truth_solution",
            "llm_feedback_from_ground_truth_solution",
        ] = "final_answer",
    ):
        super().__init__(
            env_id=env_id,
            env_type=env_type,
            question=question,
            answer=answer,
            metadata=metadata,
            instruction_template=instruction_template,
            logger=logger,
        )
        self.eval_mode = eval_mode
        self.eval_tolerance = eval_tolerance
        self.expect_boxed = expect_boxed
        self.feedback_lm = feedback_lm
        self.enable_llm_feedback = enable_llm_feedback
        self._output_type: Literal["text", "json"] = output_type
        self.feedback_type: Literal[
            "final_answer",
            "ground_truth_solution",
            "llm_feedback_from_ground_truth_solution",
        ] = feedback_type
        # Extract optional solution/rationale from metadata if present
        # Expected structure: metadata.get("ground_truth_solution") -> {"final_answer": str, "rationale": str}
        self.ground_truth_solution: Optional[Dict[str, Any]] = None
        gts = metadata.get("ground_truth_solution") if isinstance(metadata, dict) else None
        # Accept either dict or JSON-string field
        if isinstance(gts, str):
            try:
                gts = json.loads(gts)
            except Exception:
                gts = None
        if isinstance(gts, dict):
            # normalize keys we care about
            fa = gts.get("final_answer")
            rat = gts.get("rationale")
            self.ground_truth_solution = {
                "final_answer": fa if isinstance(fa, str) else None,
                "rationale": rat if isinstance(rat, str) else None,
            }

    class OmegaAnswerSchema(BaseModel):
        final_answer: str
        rationale: str

    def output_type(self) -> Literal["text", "json"]:
        return self._output_type

    def response_schema(self) -> Optional[Dict[str, Any]]:
        if self._output_type == "json":
            return self.OmegaAnswerSchema.model_json_schema()
        return None

    @staticmethod
    def _feedback_system_prompt() -> str:
        return (
            "You are a helpful math tutor. A student attempted to solve a math problem but got it wrong. "
            "Provide concise feedback (2-3 sentences) that: "
            "1) identifies the specific error or misconception in their approach, "
            "2) gives a hint or insight to guide them toward the correct solution. "
        )

    @staticmethod
    def _feedback_user_prompt(question: str, predicted: str, correct: str, error_type: str) -> str:
        lines: List[str] = []
        lines.append("Problem:")
        lines.append(question)
        lines.append("")
        lines.append("Student's Answer:")
        lines.append(predicted)
        lines.append("")
        lines.append("Correct Answer:")
        lines.append(correct)
        lines.append("")
        lines.append(f"Error Type: {error_type}")
        lines.append("")
        lines.append("Provide feedback to help the student understand their mistake:")
        return "\n".join(lines)

    def _normalize(self, s: str) -> str:
        s = (s or "").strip()
        # Strip LaTeX boxed and inline math markers
        if s.startswith("\\boxed{") and s.endswith("}"):
            s = s[len("\\boxed{"):-1]
        if s.startswith("$") and s.endswith("$"):
            s = s[1:-1]
        return s.strip()

    def _to_float(self, s: str) -> Optional[float]:
        try:
            s_clean = s.replace(",", "").strip()
            return float(s_clean)
        except Exception:
            return None

    def evaluate(self, action: str) -> Dict[str, Any]:
        # Extract canonical answer text based on output type
        action_text: str = action
        if self._output_type == "json":
            try:
                data = json.loads(action)
                parsed = self.OmegaAnswerSchema(**data)
                action_text = parsed.final_answer
            except Exception:
                return {
                    "score": 0,
                    "target": self.answer,
                    "message": "Incorrect due to schema-parse-failed! The correct answer is {0}.".format(self.answer),
                    "extra": {"error": "schema-parse-failed"},
                }

        result = evaluate_answer(
            action_text,
            self.answer,
            mode=self.eval_mode,
            tol=self.eval_tolerance,
            expect_boxed=self.expect_boxed,
        )

        # Build base template message variants
        template_message = result.get("message", "")
        if result.get("score", 0) == 0:
            if self.feedback_type == "ground_truth_solution" and self.ground_truth_solution is not None:
                # Prefer rationale when available; otherwise include final_answer
                rationale = self.ground_truth_solution.get("rationale")
                final_answer = self.ground_truth_solution.get("final_answer") or self.answer
                if isinstance(rationale, str) and rationale.strip():
                    template_message = f"Incorrect. Here's a solution sketch: {rationale.strip()}"
                else:
                    template_message = f"Incorrect! The correct answer is {final_answer}."

            # LLM-generated feedback using ground_truth_solution context
            if (
                self.feedback_type == "llm_feedback_from_ground_truth_solution"
                and self.feedback_lm is not None
                and self.enable_llm_feedback
            ):
                try:
                    system_prompt = self._feedback_system_prompt()
                    error_type = "unknown"
                    msg = result.get("message", "")
                    if "Incorrect due to " in msg and "!" in msg:
                        try:
                            error_type = msg.split("Incorrect due to ", 1)[1].split("!", 1)[0]
                        except Exception:
                            pass
                    # Build a richer user prompt that may include the rationale/solution if present
                    correct_text = self.answer
                    if self.ground_truth_solution is not None:
                        fa = self.ground_truth_solution.get("final_answer")
                        rat = self.ground_truth_solution.get("rationale")
                        # If rationale exists, append it after the correct answer for tutoring context
                        if isinstance(rat, str) and rat.strip():
                            correct_text = f"{fa or self.answer}\n\nSolution Outline:\n{rat.strip()}"
                        elif isinstance(fa, str) and fa.strip():
                            correct_text = fa
                    user_prompt = self._feedback_user_prompt(self.question, action, correct_text, error_type)
                    with jsonlogger.json_log_context(call_type="feedback"):
                        messages = [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ]
                        feedback_text = self.feedback_lm.call(messages)
                    if isinstance(feedback_text, str) and feedback_text.strip():
                        result.setdefault("extra", {})
                        result["extra"]["template_message"] = template_message or result.get("message", "")
                        result["message"] = f"Your final answer is incorrect. Here's a detailed feedback on your solution: {feedback_text.strip()}"
                        return result
                except Exception:
                    # fall back to template message on failure
                    pass

        # If we changed template_message, ensure it's reflected
        if template_message and template_message != result.get("message"):
            result.setdefault("extra", {})
            result["extra"]["template_message"] = result.get("message", "")
            result["message"] = template_message
        return result


class OmegaMathEnvDatasetConfig(EnvDatasetConfig):
    """Config for loading Omega math datasets from Hugging Face.

    Examples for `hf_dataset`:
      - "sunyiyou/math_arithmetic_gcd_7B_test_in"
      - "allenai/omega-explorative"
    If the dataset has subsets/configs, specify `hf_subset`.
    """

    hf_dataset: str
    hf_subset: Optional[str] = None
    split: str = "test"
    input_field: str = "input"
    target_field: str = "target"
    instruction_template: Optional[str] = None
    eval_mode: str = "auto"  # auto|numeric_tol|normalized_exact|tuple_tol|set_tol|matrix_tol|expr_equiv
    eval_tolerance: float = 1e-6
    expect_boxed: bool = False
    max_samples: Optional[int] = None
    shuffle: bool = False
    seed: int = 42
    # Optional grouping controls: limit number of subsets and items per subset (preserve order)
    num_subsets: Optional[int] = None
    samples_per_subset: Optional[int] = None
    # Optional LM to generate feedback for incorrect answers
    feedback_lm_config: Optional[LMConfig] = None
    enable_llm_feedback: bool = True
    # Output type control: "text" or "json"
    output_type: Literal["text", "json"] = "text"
    # Feedback type control
    feedback_type: Literal[
        "final_answer",
        "ground_truth_solution",
        "llm_feedback_from_ground_truth_solution",
    ] = "final_answer"


class OmegaMathEnvDataset(EnvDataset):
    def __init__(self, config: OmegaMathEnvDatasetConfig, logger: Optional[Logger] = None):
        self.config = config
        self.logger = logger or getLogger("omega_dataset")
        # Instantiate feedback LM once (if configured)
        self._feedback_lm: Optional[LanguageModel] = None
        if self.config.feedback_lm_config is not None:
            try:
                self._feedback_lm = get_lm_client(self.config.feedback_lm_config, logger=self.logger)
            except Exception:
                self._feedback_lm = None
        self.dataset = self.load_dataset()

    def _build_env(self, question: str, answer: str, metadata: Dict[str, Any]) -> Environment:
        env = OmegaMathEnv(
            env_id=metadata.get("id", "omega_math"),
            env_type=metadata.get("setting_key", "omega_math"),
            question=question,
            answer=answer,
            metadata=metadata,
            instruction_template=self.config.instruction_template,
            logger=self.logger,
            eval_mode=self.config.eval_mode,
            eval_tolerance=self.config.eval_tolerance,
            expect_boxed=self.config.expect_boxed,
            feedback_lm=self._feedback_lm,
            enable_llm_feedback=self.config.enable_llm_feedback,
            output_type=self.config.output_type,
            feedback_type=self.config.feedback_type,
        )
        return env

    def load_dataset(self) -> List[Environment]:
        ds = load_dataset(self.config.hf_dataset, self.config.hf_subset) if self.config.hf_subset else load_dataset(self.config.hf_dataset)
        if self.config.split not in ds:
            # Some datasets use custom split names; raise early to guide user
            raise ValueError(f"Split '{self.config.split}' not found in dataset. Available splits: {list(ds.keys())}")
        split_ds = ds[self.config.split]
        if self.config.shuffle:
            split_ds = split_ds.shuffle(seed=self.config.seed)
        if self.config.max_samples is not None:
            split_ds = split_ds.select(range(min(self.config.max_samples, len(split_ds))))

        envs: List[Environment] = []
        in_key = self.config.input_field
        tgt_key = self.config.target_field
        # Prepare ordered subset filtering controls (preserve original iteration order)
        allowed_subset_ids: List[str] = []
        per_subset_kept: Dict[str, int] = {}

        def _subset_id(example: Dict[str, Any]) -> str:
            # Prefer common omega key; fall back to other possible group keys or a default
            for k in ("setting_key", "subset", "category", "group", "task"):
                v = example.get(k)
                if isinstance(v, str) and v.strip():
                    return v
            return "default"

        for ex in split_ds:
            # Subset gating (ordered, first-appearance)
            sid = _subset_id(ex)
            if self.config.num_subsets is not None:
                if sid not in allowed_subset_ids:
                    if len(allowed_subset_ids) >= self.config.num_subsets:
                        # Skip all items from new subsets beyond the allowed count
                        continue
                    allowed_subset_ids.append(sid)
            # Per-subset sample gating
            if self.config.samples_per_subset is not None:
                kept = per_subset_kept.get(sid, 0)
                if kept >= self.config.samples_per_subset:
                    continue

            q: Optional[str] = None
            a: Optional[str] = None

            # Preferred: use configured keys when present
            if in_key in ex and tgt_key in ex:
                src_q = ex[in_key]
                src_a = ex[tgt_key]
                # If question is a messages list (allenai schema), extract first user turn
                if isinstance(src_q, list):
                    try:
                        user_msgs = [m for m in src_q if isinstance(m, dict) and m.get("role") == "user"]
                        if user_msgs:
                            q = user_msgs[0].get("content", "")
                    except Exception:
                        q = None
                elif isinstance(src_q, str):
                    q = src_q
                # answer should be string in allenai schema
                if isinstance(src_a, str):
                    a = src_a

            # Fallback: detect allenai schema explicitly
            if q is None or a is None:
                if ("messages" in ex) and ("ground_truth" in ex):
                    msgs = ex["messages"]
                    if isinstance(msgs, list):
                        user_msgs = [m for m in msgs if isinstance(m, dict) and m.get("role") == "user"]
                        if user_msgs:
                            q = user_msgs[0].get("content", "")
                    gt = ex["ground_truth"]
                    if isinstance(gt, str):
                        a = gt

            if q is None or a is None:
                # Skip malformed rows but log once
                self.logger.warning("Could not derive question/answer from example keys=%s", list(ex.keys()))
                continue

            metadata = {k: v for k, v in ex.items() if k not in (in_key, tgt_key)}
            # If the dataset provides ground_truth_solution column, pass it through
            if "ground_truth_solution" in ex and isinstance(ex["ground_truth_solution"], dict):
                metadata["ground_truth_solution"] = ex["ground_truth_solution"]
            envs.append(self._build_env(question=q, answer=a, metadata=metadata))
            # Update per-subset counts only when we actually include an example
            if self.config.samples_per_subset is not None:
                per_subset_kept[sid] = per_subset_kept.get(sid, 0) + 1
        return envs


