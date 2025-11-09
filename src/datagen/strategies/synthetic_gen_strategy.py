from typing import List, Dict, Any, Optional, Literal, Union, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from tqdm import tqdm
import random

from src.datagen.strategies.strategy import StrategyConfig
from src.datagen.strategies.strategy import Strategy
from src.datagen.types import GenerationItem, Message, SyntheticTask
from src.datagen.memory_adapters.memory_adapter import MemoryAdapterConfig
from src.datagen.memory_adapters.factory import build_memory_adapter
from src.lm.lm_factory import get_lm_client
from src.datagen.memory_adapters.history_list_adapter import HistoryAdapter

class SyntheticGenStrategyConfig(StrategyConfig):
    synthetic_gen_system_prompt_path: str
    synthetic_gen_user_prompt_path: str
    task_system_prompt_path: str
    num_generations: int
    num_threads: int = 1
    chat_mode: bool = True
    num_experiences_subsample_per_task_generation: Optional[Union[int, Tuple[int, int]]] = None
    use_task_generation_experiences_for_solving: bool = False
    shuffle_experiences_when_solving: bool = False
    memory_adapter: MemoryAdapterConfig
    solver_memory_adapter: Optional[MemoryAdapterConfig] = None
    student_messages_mode: Literal["none", "teacher_context", "custom_memory"] = "none"
    student_memory_adapter: Optional[MemoryAdapterConfig] = None
    lm_config: Dict[str, Any]
    
    
class SyntheticGenStrategy(Strategy):
    def __init__(self, config: SyntheticGenStrategyConfig, logger=None):
        super().__init__(config, logger)
        self.config: SyntheticGenStrategyConfig = config
        self.resolve_prompts_paths()
        self._question_memory_adapter: Optional[HistoryAdapter] = None
        self._answer_memory_adapter: Optional[HistoryAdapter] = None
        self._student_memory_adapter: Optional[HistoryAdapter] = None
        self._question_memory_adapter_config: MemoryAdapterConfig = config.memory_adapter
        self._answer_memory_adapter_config: MemoryAdapterConfig = (
            config.solver_memory_adapter or config.memory_adapter
        )
        if config.student_messages_mode == "custom_memory":
            if config.student_memory_adapter is None:
                raise ValueError("student_memory_adapter must be provided when student_messages_mode is 'custom_memory'")
            self._student_memory_adapter_config: MemoryAdapterConfig = config.student_memory_adapter
        else:
            self._student_memory_adapter_config = config.student_memory_adapter or config.memory_adapter
        self.lm_client = get_lm_client(config.lm_config)
    
    def _build_history_adapter(self, adapter_config: MemoryAdapterConfig) -> HistoryAdapter:
        history_adapter = build_memory_adapter(adapter_config)
        assert isinstance(history_adapter, HistoryAdapter), "Memory adapter must be a HistoryAdapter"
        return history_adapter

    def _get_question_memory_adapter(self) -> HistoryAdapter:
        if self._question_memory_adapter is None:
            self._question_memory_adapter = self._build_history_adapter(self._question_memory_adapter_config)
        return self._question_memory_adapter

    def _get_answer_memory_adapter(self) -> HistoryAdapter:
        if self._answer_memory_adapter is None:
            self._answer_memory_adapter = self._build_history_adapter(self._answer_memory_adapter_config)
        return self._answer_memory_adapter

    def _get_student_memory_adapter(self) -> HistoryAdapter:
        if self._student_memory_adapter is None:
            self._student_memory_adapter = self._build_history_adapter(self._student_memory_adapter_config)
        return self._student_memory_adapter
    
    def resolve_prompts_paths(self):
        with open(self.config.synthetic_gen_system_prompt_path, "r") as f:
            self.synthetic_gen_system_prompt = f.read()
        with open(self.config.synthetic_gen_user_prompt_path, "r") as f:
            self.synthetic_gen_user_prompt = f.read()
        with open(self.config.task_system_prompt_path, "r") as f:
            self.task_system_prompt = f.read()
    
    def triplets_to_strings(self, triplets: List[Dict[str, Any]]) -> str:
        result = []
        for triplet in triplets:
            question = triplet["Observation"]
            answer = triplet["Action"]
            feedback = triplet["Feedback"]
            result.append(f"Question: {question}\n Generated Answer: {answer}\nFeedback: {feedback}\n\n")
        return "".join(result)
    
    def generate_synthetic_task(self) -> Any:
        experiences = list(self._get_question_memory_adapter().to_triplets())
        
        # breakpoint()
        random.shuffle(experiences)
        if self.config.num_experiences_subsample_per_task_generation is not None:
            if isinstance(self.config.num_experiences_subsample_per_task_generation, int):
                num_exprs = self.config.num_experiences_subsample_per_task_generation
            elif isinstance(self.config.num_experiences_subsample_per_task_generation, tuple):
                min_exprs, max_exprs = self.config.num_experiences_subsample_per_task_generation
                num_exprs = random.randint(min_exprs, max_exprs)
            experiences = experiences[:num_exprs]
        experiences_string = self.triplets_to_strings(experiences)
        messages = [
            {"role": "system", "content": self.synthetic_gen_system_prompt},
            {"role": "user", "content": self.synthetic_gen_user_prompt.format(experiences=experiences_string)}
        ]
        response = self.lm_client.call(messages=messages)
        return response["text"], experiences
    
    def generate_questions(self) -> List[SyntheticTask]:

        def _build_task(idx: int, synthetic_task: str, used_experiences: List[Dict[str, Any]]) -> SyntheticTask:
            return SyntheticTask(
                id=f"synthetic_task_{idx}",
                prompt_text=synthetic_task,
                used_experiences=used_experiences,
                metadata={"generation_index": idx},
            )

        tasks: List[SyntheticTask] = []

        if self.config.num_threads == 1:
            for i in range(self.config.num_generations):
                synthetic_task, used_experiences = self.generate_synthetic_task()
                tasks.append(_build_task(i, synthetic_task, used_experiences))
        else:
            lock = Lock()
            with ThreadPoolExecutor(max_workers=self.config.num_threads) as ex:
                future_to_idx = {}
                for i in range(self.config.num_generations):
                    fut = ex.submit(self.generate_synthetic_task)
                    future_to_idx[fut] = i
                
                with tqdm(total=self.config.num_generations, desc="Generating synthetic tasks", unit="task") as pbar:
                    for fut in as_completed(future_to_idx):
                        idx = future_to_idx[fut]
                        synthetic_task, used_experiences = fut.result()
                        with lock:
                            tasks.append(_build_task(idx, synthetic_task, used_experiences))
                        pbar.update(1)

        tasks.sort(key=lambda t: t.id)
        return tasks

    def _build_messages_from_triplets(self, triplets: List[Dict[str, Any]]) -> List[Message]:
        messages: List[Message] = []
        for triplet in triplets:
            if "Observation" in triplet:
                messages.append(Message(role="user", content=triplet["Observation"]))
            if "Action" in triplet:
                messages.append(Message(role="assistant", content=triplet["Action"]))
            if "Feedback" in triplet:
                messages.append(Message(role="user", content=triplet["Feedback"]))
        return messages

    def _build_collapsed_prefix(self, triplets: List[Dict[str, Any]]) -> str:
        exp_blocks: List[str] = []
        for triplet in triplets:
            lines: List[str] = []
            if "Observation" in triplet:
                lines.append(f"Observation: {triplet['Observation']}")
            if "Action" in triplet:
                lines.append(f"Action: {triplet['Action']}")
            if "Feedback" in triplet:
                lines.append(f"Feedback: {triplet['Feedback']}")
            if lines:
                exp_blocks.append("\n".join(lines))
        user_content = "Here are the previous experiences you've had and their feedback:\n\n"
        user_content += "\n\n".join(exp_blocks)
        user_content += "\n\nHere is the current observation: "
        return user_content

    def build_items_from_tasks(self, tasks: List[SyntheticTask]) -> List[GenerationItem]:
        all_memory_triplets = list(self._get_answer_memory_adapter().to_triplets())
        student_mode = self.config.student_messages_mode
        student_memory_triplets: Optional[List[Dict[str, Any]]] = None
        if student_mode == "custom_memory":
            student_memory_triplets = list(self._get_student_memory_adapter().to_triplets())

        def _create_item(task: SyntheticTask, triplets: List[Dict[str, Any]]) -> GenerationItem:
            if self.config.chat_mode:
                teacher_messages = [Message(role="system", content=self.task_system_prompt)] + self._build_messages_from_triplets(triplets) + [Message(role="user", content=task.prompt_text)]
            else:
                combined_user = self._build_collapsed_prefix(triplets) + task.prompt_text
                teacher_messages = [Message(role="system", content=self.task_system_prompt), Message(role="user", content=combined_user)]

            student_messages: List[Message]
            if student_mode == "teacher_context":
                student_messages = [msg.model_copy() for msg in teacher_messages]
            elif student_mode == "custom_memory":
                if student_memory_triplets is None:
                    raise ValueError("Student memory adapter not initialized.")
                triplets_student = list(student_memory_triplets)
                if self.config.shuffle_experiences_when_solving:
                    random.shuffle(triplets_student)
                if self.config.chat_mode:
                    student_messages = [Message(role="system", content=self.task_system_prompt)] + self._build_messages_from_triplets(triplets_student) + [Message(role="user", content=task.prompt_text)]
                else:
                    combined_student = self._build_collapsed_prefix(triplets_student) + task.prompt_text
                    student_messages = [
                        Message(role="system", content=self.task_system_prompt),
                        Message(role="user", content=combined_student),
                    ]
            else:
                student_messages = [
                    Message(role="system", content=self.task_system_prompt),
                    Message(role="user", content=task.prompt_text),
                ]

            return GenerationItem(
                id=task.id,
                teacher_messages=teacher_messages,
                student_messages=student_messages,
                metadata=task.metadata or {},
            )

        items: List[GenerationItem] = []
        for task in tasks:
            if self.config.use_task_generation_experiences_for_solving and task.used_experiences:
                triplets_to_use = list(task.used_experiences)
            else:
                triplets_to_use = list(all_memory_triplets)

            if self.config.shuffle_experiences_when_solving:
                random.shuffle(triplets_to_use)

            items.append(_create_item(task, triplets_to_use))

        return items

    def generate(self) -> List[GenerationItem]:
        tasks = self.generate_questions()
        return self.build_items_from_tasks(tasks)
