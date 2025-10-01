from typing import Dict, Any, Optional, List, Tuple
from logging import Logger, getLogger

from pydantic import BaseModel

from datasets import load_dataset

from src.data.env import EnvDataset, EnvDatasetConfig, Environment
from src.data.envs.qa_env import QAEnv
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
        out = {"correct": correct, "target": target, "message": msg}
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
        question: str,
        answer: str,
        metadata: Dict[str, Any],
        instruction_template: Optional[str] = None,
        logger: Optional[Logger] = None,
        eval_mode: str = "auto",
        eval_tolerance: float = 1e-6,
        expect_boxed: bool = False,
    ):
        super().__init__(
            question=question,
            answer=answer,
            metadata=metadata,
            instruction_template=instruction_template,
            logger=logger,
        )
        self.eval_mode = eval_mode
        self.eval_tolerance = eval_tolerance
        self.expect_boxed = expect_boxed

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
        return evaluate_answer(
            action,
            self.answer,
            mode=self.eval_mode,
            tol=self.eval_tolerance,
            expect_boxed=self.expect_boxed,
        )


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


class OmegaMathEnvDataset(EnvDataset):
    def __init__(self, config: OmegaMathEnvDatasetConfig, logger: Optional[Logger] = None):
        self.config = config
        self.logger = logger or getLogger("omega_dataset")
        self.dataset = self.load_dataset()

    def _build_env(self, question: str, answer: str, metadata: Dict[str, Any]) -> Environment:
        env = OmegaMathEnv(
            question=question,
            answer=answer,
            metadata=metadata,
            instruction_template=self.config.instruction_template,
            logger=self.logger,
            eval_mode=self.config.eval_mode,
            eval_tolerance=self.config.eval_tolerance,
            expect_boxed=self.config.expect_boxed,
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
        for ex in split_ds:
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
            envs.append(self._build_env(question=q, answer=a, metadata=metadata))
        return envs


