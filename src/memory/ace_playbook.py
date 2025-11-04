from __future__ import annotations

from typing import Dict, List, Optional, Union, Any
from pathlib import Path
from pydantic import BaseModel
import json
import time

from src.memory.memory_module import MemoryModule, MemoryModuleConfig
 


class Bullet(BaseModel):
    id: str
    content: str
    helpful: int = 0
    harmful: int = 0
    neutral: int = 0
    metadata: Optional[Dict[str, Any]] = None


class Playbook(BaseModel):
    sections: Dict[str, List[Bullet]] = {}

    def to_text(self) -> str:
        lines: List[str] = []
        for section, bullets in self.sections.items():
            lines.append(f"[Section] {section}")
            for b in bullets:
                counters = f"helpful={b.helpful} harmful={b.harmful} neutral={b.neutral}"
                lines.append(f"[{b.id}] {counters} :: {b.content}")
            lines.append("")
        return "\n".join(lines).strip()

    def stats(self) -> Dict[str, Any]:
        counts_by_section: Dict[str, int] = {s: len(bs) for s, bs in self.sections.items()}
        total = sum(counts_by_section.values())
        return {
            "total_bullets": total,
            "sections": counts_by_section,
        }

    def _next_id(self, prefix: str = "ctx") -> str:
        # Timestamp-based unique id; readable and monotonic enough for single-process
        ts = int(time.time() * 1000)
        return f"{prefix}-{ts}"

    def apply_curator_ops(self, operations: List[Dict[str, Any]]) -> List[Bullet]:
        added: List[Bullet] = []
        for op in operations or []:
            if op.get("type") != "ADD":
                continue
            section = op.get("section") or "general"
            content = op.get("content") or ""
            bullet = Bullet(id=self._next_id(), content=str(content))
            if section not in self.sections:
                self.sections[section] = []
            self.sections[section].append(bullet)
            added.append(bullet)
        return added

    def tag_bullets(self, tags: List[Dict[str, str]]) -> None:
        index: Dict[str, Bullet] = {}
        for _, bullets in self.sections.items():
            for b in bullets:
                index[b.id] = b
        for item in tags or []:
            bid = item.get("id") or item.get("bullet_id") or item.get("bulletId")
            tag = (item.get("tag") or "").lower()
            if bid in index:
                if tag == "helpful":
                    index[bid].helpful += 1
                elif tag == "harmful":
                    index[bid].harmful += 1
                else:
                    index[bid].neutral += 1


class AceMemoryConfig(MemoryModuleConfig):
    _type: str = "ace_playbook"
    max_history_length: Optional[int] = None


class AceMemory(MemoryModule):
    def __init__(self, config: AceMemoryConfig):
        super().__init__(config)
        self.playbook: Playbook = Playbook(sections={})
        self.reflections: List[Dict[str, Any]] = []

    # History-compatible API
    def _update(self, entry: Any):
        # No-op: ACE manages its own transient state in the agent
        return None

    def recall(self) -> List[Any]:
        # No chat-style history is stored in memory for ACE
        return []

    # Convenience helpers
    def last_reflection_text(self) -> str:
        if not self.reflections:
            return ""
        ref = self.reflections[-1]
        # Prefer key_insight + reasoning if present
        ki = ref.get("key_insight")
        reasoning = ref.get("reasoning")
        parts = []
        if isinstance(ki, str) and ki.strip():
            parts.append(f"Key insight: {ki}")
        if isinstance(reasoning, str) and reasoning.strip():
            parts.append(f"Reasoning: {reasoning}")
        return "\n".join(parts) if parts else json.dumps(ref)

    # Snapshot implementations
    def save_snapshot(self, base_dir: Union[str, Path], snapshot_id: Union[int, str]) -> str:
        base = Path(base_dir)
        base.mkdir(parents=True, exist_ok=True)
        mem_dir = base / f"ace_memory_{snapshot_id}"
        mem_dir.mkdir(parents=True, exist_ok=True)

        # Write playbook
        with open(mem_dir / "playbook.json", "w", encoding="utf-8") as f:
            f.write(self.playbook.model_dump_json())

        # Write reflections
        with open(mem_dir / "reflections.jsonl", "w", encoding="utf-8") as f:
            for ref in self.reflections:
                f.write(json.dumps(ref) + "\n")

        return str(mem_dir)

    @classmethod
    def load_snapshot(cls, snapshot_path: Union[str, Path]) -> "AceMemory":
        path = Path(snapshot_path)
        cfg = AceMemoryConfig()
        mem = cls(cfg)

        # Load playbook
        p_path = path / "playbook.json"
        if p_path.exists():
            with open(p_path, "r", encoding="utf-8") as f:
                pb = json.loads(f.read())
            sections: Dict[str, List[Bullet]] = {}
            for section_name, bullets in (pb.get("sections") or {}).items():
                sections[section_name] = [Bullet(**b) for b in bullets]
            mem.playbook = Playbook(sections=sections)

        # Load reflections
        r_path = path / "reflections.jsonl"
        if r_path.exists():
            with open(r_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    mem.reflections.append(json.loads(line))

        return mem


