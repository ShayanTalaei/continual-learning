from typing import List, Any, Union, Dict

from src.agent.memory_agent import MemoryAgent, MemoryAgentConfig
from src.memory.history_list import HistoryListConfig, Entry


class HistoryAgentConfig(MemoryAgentConfig):
    memory_config: HistoryListConfig  # type: ignore[assignment]
    history_k: Union[int, None] = None


class HistoryAgent(MemoryAgent):
    def __init__(self, config: HistoryAgentConfig, logger=None):
        super().__init__(config, logger=logger)

    def build_system_prompt(self) -> str:
        history_list_instructions = ("You will be given the previous experiences you've had and their feedback. "
            "You should use this feedback to improve your performance in the subsequent actions.")
        
        return self.system_prompt #+ "\n\n" + history_list_instructions

    def build_user_prompt(self, obs: str, history: List[Any], k: Union[int, None], merge_feedback_and_observation: bool = False) -> List[Dict[str, str]]:
        messages: List[dict] = []
        recent: List[Entry] = history[-k:] if k is not None else history  # type: ignore[assignment]
        
        # messages.append({"role": "user", "content": "Here are the previous experiences you've had and their feedback:"})
        # Add previous experiences as alternating user/assistant messages
        # The following code (detailed user/assistant alternating message construction) is commented out in favor of a single user message:
        # for entry in recent:
        #     if entry.type.lower() == "observation":
        #         messages.append({"role": "user", "content": str(entry.content)})
        #     elif entry.type.lower() == "action":
        #         messages.append({"role": "assistant", "content": str(entry.content)})
        #     elif entry.type.lower() == "feedback":
        #         # Add feedback as a user message
        #         messages.append({"role": "user", "content": f"{entry.content}"}) #Feedback: 
        # messages.append({"role": "user", "content": f"{obs}"}) #Here is the current observation: 

        # Instead, create a single user message with all previous experiences and feedbacks:
        combined_content_lines = []
        for entry in recent:
            if entry.type.lower() == "observation":
                combined_content_lines.append(f"{entry.content}".split("?")[1].split('.')[0])
            # elif entry.type.lower() == "action":
            #     combined_content_lines.append(f"Action: {entry.content}")
            elif entry.type.lower() == "feedback":
                combined_content_lines.append(f"{entry.content}\n\n")
        combined_content_lines.append(f"Current question: {obs.split('.')[0] + '.'}")
        combined_content = "\n".join(combined_content_lines)
        messages.append({"role": "user", "content": combined_content})

                # Add feedback as a user message
                messages.append({"role": "user", "content": f"{entry.content}"}) #Feedback: 
        # if len(recent) == 0:
        #     messages.append({"role": "user", "content": "No previous experiences."})
        
        # Add current observation as the final user message
        messages.append({"role": "user", "content": f"{obs}"}) #Here is the current observation: 

        if merge_feedback_and_observation:
            merged_messages = []
            curr_idx = 0
            while curr_idx < len(messages):
                if curr_idx < len(messages)-1 and messages[curr_idx]["role"] == "user" and messages[curr_idx+1]["role"] == "user":
                    merged_messages.append({"role": "user", "content": f"{messages[curr_idx]['content']}\n\n{messages[curr_idx+1]['content']}"})
                    curr_idx += 2
                else:
                    merged_messages.append(messages[curr_idx])
                    curr_idx += 1
            return merged_messages
        
        return messages

    # def end_episode(self) -> None:
    #     self._trajectory = []
    #     ## Warning: This is a patch to test the positive signal of the feedback.
    #     if len(self.memory.recall()) > 2 and self.training and self.memory.recall()[-1].content.startswith("Feedback: You incorrectly"):
    #         # breakpoint()
    #         feedback_event = self.memory.history_list.pop()
    #         action_event = self.memory.history_list.pop()
    #         import re
    #         # Extract incorrect and correct answers from feedback message
    #         pattern = r"Feedback: You incorrectly answered (.*?)! But, the correct anwer for this question is (.*?)\."
    #         match = re.search(pattern, feedback_event.content)
    #         if match:
    #             incorrect_answer = match.group(1)
    #             correct_answer = match.group(2)
    #             self.logger.info(f"Extracted answers - Incorrect: {incorrect_answer}, Correct: {correct_answer}")
    #             # Override the incorrect answer with correct answer using \boxed
    #             corrected_action = action_event.content.replace(f"\\boxed{{{incorrect_answer}}}", f"\\boxed{{{correct_answer}}}")
                
    #             # Push back corrected action and positive feedback
    #             self.memory.history_list.append(Entry(type="Action", content=corrected_action))
    #             self.memory.history_list.append(Entry(type="Feedback", content="Feedback: Correct!"))
    #             self.logger.info("Replaced incorrect answer with correct answer and updated feedback")
        

    def create_observation_event(self, obs: str) -> Any:
        return Entry(type="Observation", content=obs)

    def create_action_event(self, action: str) -> Any:
        return Entry(type="Action", content=action)

    def create_feedback_event(self, feedback: dict) -> Any:
        return Entry(type="Feedback", content=f"->  \\boxed{{{feedback.get('target', '')}}}.")
        # return Entry(type="Feedback", content=feedback.get("message", ""))


