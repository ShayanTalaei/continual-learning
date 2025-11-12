from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Sequence, Tuple, Union

import yaml

from src.agent.history_agent import HistoryAgentConfig
from src.memory.history_list import HistoryList, HistoryListConfig, Entry
from src.run_config import RunConfig
from src.data.dataset_factory import build_dataset
from src.data.env import Environment, EnvDataset

from .types import ChatMessage


@dataclass
class ExperienceEvent:
    text: str
    entry_index: int
    step: int
    entry: Dict[str, Any]


@dataclass
class ExperienceBundle:
    observation: Optional[ExperienceEvent] = None
    action: Optional[ExperienceEvent] = None
    feedback: Optional[ExperienceEvent] = None


def resolve_system_prompt(system_prompt: str) -> str:
    if system_prompt.startswith("{file:") and system_prompt.endswith("}"):
        path = Path(system_prompt[len("{file:") : -1])
        return path.read_text(encoding="utf-8")
    return system_prompt


def load_run_configuration(config_path: Union[str, Path]) -> Tuple[RunConfig, Dict[str, object]]:
    from src.main import resolve_paths

    path = Path(config_path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    resolved = resolve_paths(raw)
    run_conf = RunConfig(**resolved)
    return run_conf, resolved


def load_history_memory(
    history_config: HistoryListConfig,
    snapshot_path: Optional[Union[str, Path]] = None,
) -> HistoryList:
    if snapshot_path is not None:
        mem = HistoryList.load_snapshot(snapshot_path)
    else:
        mem = HistoryList(history_config)
    if history_config.max_length is not None and len(mem.history_list) > history_config.max_length:
        mem.history_list = mem.history_list[-history_config.max_length :]
    return mem


class HistoryPromptBuilder:
    def __init__(self, config: HistoryAgentConfig):
        self.config = config

    def _collect_experiences(self, entries: Sequence[Entry]) -> List[ExperienceBundle]:
        experiences: List[ExperienceBundle] = []
        current = ExperienceBundle()
        current_step = 0

        for entry_index, entry in enumerate(entries):
            etype = entry.type.lower()
            event = ExperienceEvent(
                text=str(entry.content),
                entry_index=entry_index,
                step=current_step,
                entry=entry.model_dump(),
            )

            if etype == "observation":
                if current.observation or current.action or current.feedback:
                    experiences.append(current)
                    current_step += 1
                    current = ExperienceBundle()
                current.observation = event
            elif etype == "action":
                current.action = event
            elif etype == "feedback":
                current.feedback = event

        if current.observation or current.action or current.feedback:
            experiences.append(current)

        return experiences

    def _format_collapsed_prompt(self, experiences: List[ExperienceBundle], obs: str) -> str:
        blocks: List[str] = []
        for exp in experiences:
            lines: List[str] = []
            observation = exp.observation.text if exp.observation else None
            action = exp.action.text if exp.action else None
            feedback = exp.feedback.text if exp.feedback else None

            if observation:
                lines.append(f"Observation: {observation}")
            if action:
                lines.append(f"Action: {action}")
            if feedback:
                lines.append(f"Feedback: {feedback}")

            if lines:
                blocks.append("\n".join(lines))

        prior = "\n\n".join(blocks)
        parts = [prior] if prior else []
        parts.append(str(obs))
        return "\n\n".join(parts)

    def _format_expanded_prompt(
        self,
        experiences: List[ExperienceBundle],
    ) -> List[Tuple[str, ExperienceEvent, str]]:
        """Return (role, event, tag_suffix) tuples for each message."""
        messages: List[Tuple[str, ExperienceEvent, str]] = []

        for idx, exp in enumerate(experiences):
            if exp.observation:
                messages.append(("user", exp.observation, f"memory_{idx}_observation"))
            if exp.action:
                messages.append(("assistant", exp.action, f"memory_{idx}_action"))
            if exp.feedback:
                messages.append(("user", exp.feedback, f"memory_{idx}_feedback"))
        return messages

    def build_history_messages(
        self,
        obs: str,
        history_entries: Sequence[Entry],
    ) -> List[ChatMessage]:
        recent_entries = (
            history_entries[-self.config.history_k :] if self.config.history_k is not None else history_entries
        )
        experiences = self._collect_experiences(recent_entries)
        if self.config.collapse_messages:
            content = self._format_collapsed_prompt(experiences, obs)
            return [ChatMessage(role="user", content=content, tag="current_observation")]

        messages: List[ChatMessage] = []
        for role, event, tag in self._format_expanded_prompt(experiences):
            metadata = {
                "entry_index": event.entry_index,
                "step": event.step,
                "entry": event.entry,
            }
            messages.append(ChatMessage(role=role, content=event.text, tag=tag, metadata=metadata))
        messages.append(
            ChatMessage(
                role="user",
                content=str(obs),
                tag="current_observation",
                metadata={"is_current_observation": True},
            )
        )
        return messages


def prepare_conversation_messages(
    system_prompt: str,
    history_messages: Sequence[ChatMessage],
) -> List[ChatMessage]:
    messages = [
        ChatMessage(
            role="system",
            content=system_prompt,
            tag="system_prompt",
            metadata={"is_system_prompt": True},
        )
    ]
    messages.extend(history_messages)
    return messages


def build_validation_dataset(
    config: Dict[str, object],
) -> EnvDataset:
    return build_dataset(config, logger=None)


def iter_validation_prompts(
    dataset: EnvDataset,
) -> Generator[Tuple[Environment, str], None, None]:
    for env in dataset.get_dataset():
        obs = env.reset()
        yield env, obs


