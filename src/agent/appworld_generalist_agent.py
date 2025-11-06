from typing import List, Any, Dict, Optional, Union
import re
import json
import ast

from src.agent.memory_agent import MemoryAgent, MemoryAgentConfig
from src.memory.history_list import HistoryListConfig, Entry
from src.agent.variable_manager import VariablesManager


class AppWorldPlannerConfig(MemoryAgentConfig):
    """Configuration for the AppWorld Planner component."""
    memory_config: HistoryListConfig  # type: ignore[assignment]
    history_k: int | None = 50

    # Planning-specific configuration
    planning_enabled: bool = True
    replan_on_failure: bool = True
    max_planning_steps: int = 10

    # Agent behavior
    system_prompt: str | None = None
    verbose: bool = True


class AppWorldPlanner(MemoryAgent):
    """AppWorld Planner: High-level task decomposition and strategy formulation.

    This implements the Planner component of the CUGA architecture:
    - Decomposes complex tasks into executable sub-tasks
    - Provides strategic guidance for long-horizon planning
    - Coordinates with specialized sub-execution agents
    """

    def __init__(self, config: AppWorldPlannerConfig, logger=None):
        super().__init__(config, logger=logger)
        self._planning_step_counter: int = 0
        self._current_plan: List[str] = []
        self._last_failure_reason: str = ""
        self._initial_observation: str | None = None
        self.logger.info("AppWorldPlanner init: planning=%s, replan_on_failure=%s",
                        self.config.planning_enabled, self.config.replan_on_failure)

    def reset(self) -> None:
        """Reset planner state between episodes."""
        self.memory.reset()
        self._trajectory = []
        self._last_action = None
        self._planning_step_counter = 0
        self._current_plan.clear()
        self._last_failure_reason = ""
        self._initial_observation = None

    def end_episode(self) -> None:
        """Reset planner state when an episode finishes."""
        self.reset()

    def _set_system_prompt(self) -> None:
        """Use the class-defined system prompt instead of requiring config to provide one."""
        self.system_prompt = self.build_system_prompt()

    def build_system_prompt(self) -> str:
        """Build system prompt for the planner component."""
        cfg_prompt = getattr(self.config, "system_prompt", None)
        if cfg_prompt:
            return cfg_prompt

        return (
            "You are the PLANNER in a multi-agent CUGA system for completing AppWorld tasks.\n\n"
            "=== YOUR ROLE ===\n"
            "You provide HIGH-LEVEL STRATEGIC GUIDANCE but DO NOT write code or execute APIs.\n"
            "Your output format is ALWAYS: think: <your strategic guidance>\n\n"
            "=== YOUR RESPONSIBILITIES ===\n"
            "1. TASK DECOMPOSITION: Break complex tasks into logical sub-goals\n"
            "   Example: 'think: To count playlists: 1) login to Spotify, 2) paginate through playlist API, 3) count results'\n\n"
            "2. API DISCOVERY GUIDANCE: Suggest which apps/APIs might be needed\n"
            "   Example: 'think: Need to find Spotify credentials - check supervisor app for passwords'\n\n"
            "3. ERROR RECOVERY: When you see execution failures, provide corrective strategy\n"
            "   Example: 'think: Login failed - verify credentials and retry with correct password variable'\n\n"
            "4. PROGRESS MONITORING: Acknowledge successful steps briefly\n"
            "   Example: 'think: Good progress - credentials obtained, now proceed with API calls'\n\n"
            "=== WHAT YOU DON'T DO ===\n"
            "- DO NOT write Python code or ```python blocks\n"
            "- DO NOT make specific API calls or suggest exact parameter values\n"
            "- DO NOT handle implementation details (that's the executor's job)\n\n"
            "=== COMMUNICATION STYLE ===\n"
            "Always prefix with 'think:' and provide strategic direction, not implementation.\n"
            "Keep guidance concise but actionable. Focus on WHAT to do next, not HOW to implement it."
        )

    def build_user_prompt(self, obs: str, history: List[Any], k: int | None) -> List[Dict[str, str]]:
        """Build user prompt with task context and planning history.
        
        Uses alternating user/assistant format like history_agent:
        - Observations → user messages
        - Actions → assistant messages
        - Feedback → user messages
        """
        messages: List[Dict[str, str]] = []

        if self._initial_observation:
            messages.append({"role": "user", "content": self._initial_observation})

        # Add history as alternating messages
        recent: List[Entry] = history[-k:] if k is not None else history  # type: ignore[assignment]
        
        for entry in recent:
            if entry.type == "Observation":
                messages.append({"role": "user", "content": str(entry.content)})
            elif entry.type == "Action":
                messages.append({"role": "assistant", "content": str(entry.content)})
            elif entry.type == "Feedback":
                if isinstance(entry.content, dict):
                    msg = entry.content.get("message", "")
                    messages.append({"role": "user", "content": f"Feedback: {msg}"})
                else:
                    messages.append({"role": "user", "content": f"Feedback: {str(entry.content)}"})
        
        # Add current observation as final user message
        messages.append({"role": "user", "content": obs})

        return messages

    def set_initial_task(self, task_description: str) -> None:
        self._initial_observation = task_description

    def create_observation_event(self, obs: str) -> Any:
        """Create observation entry for memory."""
        return Entry(type="Observation", content=obs)

    def create_action_event(self, action: str) -> Any:
        """Create planning decision entry for memory."""
        return Entry(type="Action", content=action)

    def create_feedback_event(self, feedback: dict) -> Any:
        """Create feedback entry for memory."""
        return Entry(type="Feedback", content=feedback)

    def _should_replan(self, obs: str, feedback: dict) -> bool:
        """Determine if we need to replan based on current situation."""
        if not self.config.planning_enabled:
            return False

        # Always plan at the start
        if self._planning_step_counter == 0:
            return True

        # Replan after failures if enabled
        if self.config.replan_on_failure and self._last_failure_reason:
            return True

        # Limit total planning steps
        if self._planning_step_counter >= self.config.max_planning_steps:
            return False

        return False

    def act(self, obs: str) -> str:
        """Generate planning guidance for the current task state."""
        # Check if we need to replan
        if not self._should_replan(obs, {}):  # We don't have feedback here, so pass empty dict
            return "think: Continue"  # Signal to continue with current plan

        self._planning_step_counter += 1

        # Generate planning guidance
        history = self.memory.recall()
        system_prompt = self.build_system_prompt()
        user_messages = self.build_user_prompt(obs, history, self.config.history_k)

        from src.utils import logger as jsonlogger
        with jsonlogger.json_log_context(call_type="planning"):
            messages = [{"role": "system", "content": system_prompt}] + user_messages
            resp = self.lm.call(messages)

        planning_guidance = (resp.get("text") or "").strip()
        if not planning_guidance:
            planning_guidance = "Analyze the current task and break it down into executable steps using available APIs."

        # Normalize to a single think: prefix
        if planning_guidance.lower().startswith("think:"):
            planning_guidance = planning_guidance.split(":", 1)[1].strip()
        plan_action = f"think: {planning_guidance}".strip()

        # Clear failure flag after producing a new plan
        self._last_failure_reason = ""

        # Store planning decision
        self._last_action = plan_action
        action_event = self.create_action_event(self._last_action)
        if action_event is not None:
            self.memory.update(action_event)
            self._trajectory.append(action_event)

        self.logger.info("Planning decision: %s",
                        planning_guidance[:100] + "..." if len(planning_guidance) > 100 else planning_guidance)
        return plan_action

    def observe(self, obs: str | None, feedback: dict, done: bool) -> None:
        """Observe execution feedback. Store observations for the planner to reason about."""
        # Store feedback via parent
        super().observe(obs, feedback, done)
        
        # Update failure tracking heuristically based on observation / feedback
        failure_detected = False
        if obs:
            lower_obs = obs.lower()
            failure_indicators = [
                "execution failed",
                "traceback",
                "error",
                "exception",
                "failed to",
                "could not",
                "invalid",
                "permission denied",
            ]
            if any(indicator in lower_obs for indicator in failure_indicators):
                failure_detected = True
                self._last_failure_reason = obs
            elif lower_obs.strip() in {"ok.", "ok"}:
                self._last_failure_reason = ""

        if not failure_detected:
            score = float(feedback.get("score", 0.0) or 0.0)
            won = bool(feedback.get("won", False))
            if score < 1.0 and not won:
                message = feedback.get("message")
                if message and message != "OK.":
                    self._last_failure_reason = str(message)
            else:
                self._last_failure_reason = ""
        
        if done:
            won = bool(feedback.get("won", False))
            if won:
                self.logger.info("Planner: Episode completed successfully")
            else:
                self.logger.info("Planner: Episode ended without completion")
        
        self.logger.debug("Planner observed: done=%s, obs=%s", done, obs[:50] if obs else "None")


class AppWorldAPISubAgentConfig(MemoryAgentConfig):
    """Configuration for the AppWorld API Sub-task Agent."""
    memory_config: HistoryListConfig  # type: ignore[assignment]
    history_k: int | None = 50

    # API sub-agent specific configuration
    guardrails_enabled: bool = False
    enforce_doc_lookup: bool = True
    retry_on_api_error: int = 1
    max_api_calls_per_step: int = 1

    # Agent behavior
    system_prompt: str | None = None
    verbose: bool = True


class AppWorldAPISubAgent(MemoryAgent):
    """AppWorld API Sub-task Agent: Specialized execution agent for API operations.

    This implements the Sub-execution Agent component of the CUGA architecture:
    - Handles individual API calls with proper documentation lookup
    - Executes specific sub-tasks assigned by the planner
    - Maintains variable state and handles pagination
    - Applies safety guardrails and error recovery
    """

    def __init__(self, config: AppWorldAPISubAgentConfig, logger=None):
        super().__init__(config, logger=logger)
        self._step_counter: int = 0
        self._doc_lookups_done: set = set()  # Track which APIs we've looked up docs for
        self._last_failed: bool = False
        self.logger.info("AppWorldAPISubAgent init: guardrails=%s, doc_enforcement=%s",
                        self.config.guardrails_enabled, self.config.enforce_doc_lookup)

    def _set_system_prompt(self) -> None:
        """Ensure executor uses internal system prompt template."""
        self.system_prompt = self.build_system_prompt()

    def reset(self) -> None:
        """Reset executor state between episodes."""
        self.memory.reset()
        self._trajectory = []
        self._last_action = None
        self._step_counter = 0
        self._doc_lookups_done.clear()
        self._last_failed = False

    def end_episode(self) -> None:
        """Reset executor state when an episode finishes."""
        self.reset()

    def build_system_prompt(self) -> str:
        """Build system prompt for the API sub-agent."""
        cfg_prompt = getattr(self.config, "system_prompt", None)
        if cfg_prompt:
            return cfg_prompt

        return (
            "You are the API EXECUTOR in a multi-agent CUGA system for completing AppWorld tasks.\n\n"
            "=== YOUR ROLE ===\n"
            "You IMPLEMENT tasks by writing and executing Python code that calls AppWorld APIs.\n"
            "The Planner provides strategic guidance (think: messages), and YOU execute the actual API calls.\n\n"
            "=== YOUR RESPONSIBILITIES ===\n"
            "1. API DOCUMENTATION LOOKUP: ALWAYS check docs before calling APIs\n"
            "   - Use apis.api_docs.show_app_descriptions() to list available apps\n"
            "   - Use apis.api_docs.show_api_descriptions(app_name='...') to list app's APIs\n"
            "   - Use apis.api_docs.show_api_doc(app_name='...', api_name='...') before each API call\n\n"
            "2. CODE EXECUTION: Write Python code in ```python blocks\n"
            "   - One logical operation per step (e.g., one API call or small group of related operations)\n"
            "   - Store results in variables for reuse (e.g., access_token, playlists)\n"
            "   - Use print() to show important values\n\n"
            "3. STATE MANAGEMENT: Maintain variables across steps\n"
            "   - Variables persist: if you set spotify_token in step 1, it's available in step 2\n"
            "   - Handle pagination: loop through all pages, don't stop at first page\n\n"
            "4. TASK COMPLETION: Call apis.supervisor.complete_task() when done\n"
            "   - With answer if task requires one: complete_task(answer=result)\n"
            "   - Without answer otherwise: complete_task()\n\n"
            "=== SAFETY RULES ===\n"
            "- ONLY use provided apis.* modules, print(), and Python stdlib (datetime, etc.)\n"
            "- NO OS operations (os, sys, subprocess)\n"
            "- NO filesystem operations (open, write to disk)\n"
            "- NO external libraries (requests, spotipy, etc.)\n"
            "- Always validate API responses before using them\n\n"
            "=== OUTPUT FORMAT ===\n"
            "Your response should contain ONE Python code block:\n"
            "```python\n"
            "# your code here\n"
            "```\n\n"
            "=== EXECUTION WORKFLOW ===\n"
            "1. Read the planner's think: guidance (if present)\n"
            "2. Look up API documentation\n"
            "3. Write focused code for the current step\n"
            "4. Store results in variables for next steps\n"
            "5. Move methodically toward task completion"
        )

    def build_user_prompt(self, obs: str, history: List[Any], k: int | None) -> List[Dict[str, str]]:
        """Build user prompt with execution context and planner guidance.
        
        Uses alternating user/assistant format like history_agent:
        - Observations → user messages
        - Actions → assistant messages
        - Feedback → user messages
        """
        messages: List[Dict[str, str]] = []

        # Add history as alternating messages
        if len(history) > k:
            recent = [history[0]] + history[-k:] if k is not None else history    
        else:
            recent = history
        
        for entry in recent:
            if entry.type == "Observation":
                messages.append({"role": "user", "content": str(entry.content)})
            elif entry.type == "Action":
                messages.append({"role": "assistant", "content": str(entry.content)})
            elif entry.type == "Feedback":
                if isinstance(entry.content, dict):
                    msg = entry.content.get("message", "")
                    messages.append({"role": "user", "content": f"Feedback: {msg}"})
                else:
                    messages.append({"role": "user", "content": f"Feedback: {str(entry.content)}"})
        
        # Add current observation as final user message
        messages.append({"role": "user", "content": obs})

        return messages

    def create_observation_event(self, obs: str) -> Any:
        """Create observation entry for memory."""
        return Entry(type="Observation", content=obs)

    def create_action_event(self, action: str) -> Any:
        """Create action entry for memory."""
        return Entry(type="Action", content=action)

    def create_feedback_event(self, feedback: dict) -> Any:
        """Create feedback entry for memory."""
        return Entry(type="Feedback", content=feedback)

    def _wrap_single_python_block(self, text: str) -> str:
        """Ensure action is wrapped in a single Python code block."""
        text = text.strip()

        # If already properly formatted, return as-is
        if re.search(r"```\s*python.*?\n.*?\n```", text, re.IGNORECASE | re.DOTALL):
            return text

        # Extract code if it's embedded in other text
        code_match = re.search(r"(.*?```python.*?\n.*?```.*)", text, re.IGNORECASE | re.DOTALL)
        if code_match:
            return code_match.group(1).strip()

        # Wrap plain code in a block
        if text and not text.startswith("```"):
            return f"```python\n{text}\n```"

        return text

    def _violates_guardrails(self, code: str) -> Optional[str]:
        """Check if generated code violates safety guardrails."""
        # Disallow dangerous imports or OS access
        dangerous_imports = ["os", "sys", "subprocess", "shutil", "pathlib", "requests", "urllib"]
        for imp in dangerous_imports:
            if re.search(rf"\bimport\s+{re.escape(imp)}\b", code) or re.search(rf"\bfrom\s+{re.escape(imp)}\b", code):
                return f"Disallowed import '{imp}'; only minimal stdlib is permitted."

        # Disallow filesystem operations
        fs_ops = ["open(", "os.", "subprocess.", "shutil.", "pathlib."]
        for op in fs_ops:
            if op in code:
                return f"Disallowed operation '{op}'; no filesystem access allowed."

        # Must use apis.* for app interactions (unless it's just print or variable assignment)
        api_calls = re.findall(r"apis\.\w+\.\w+\(", code)
        if not api_calls and "print(" not in code and "=" not in code and "for " not in code:
            return "No API usage or meaningful action detected; each step should progress via apis.* or print()."

        # Check for API doc lookup requirement
        if self.config.enforce_doc_lookup:
            api_matches = re.findall(r"apis\.(\w+)\.(\w+)\(", code)
            for app_name, api_name in api_matches:
                api_key = f"{app_name}.{api_name}"
                if api_key not in self._doc_lookups_done:
                    return f"API '{api_key}' used without prior documentation lookup. Always check docs first."

        # Limit API calls per step
        if len(api_calls) > self.config.max_api_calls_per_step:
            return f"Too many API calls ({len(api_calls)}) in one step. Limit to {self.config.max_api_calls_per_step}."

        return None

    def _repair_action(self, original_action: str, violation: str) -> str:
        """Attempt to repair a guardrail-violating action."""
        if "documentation lookup" in violation.lower():
            # Insert doc lookup before the problematic API call
            api_match = re.search(r"apis\.(\w+)\.(\w+)\(", original_action)
            if api_match:
                app_name, api_name = api_match.groups()
                repair_code = (
                    f"print(apis.api_docs.show_api_doc(app_name='{app_name}', api_name='{api_name}'))\n"
                    "# Now proceed with the API call after reviewing the documentation"
                )
                return f"```python\n{repair_code}\n```"

        # Default repair: doc lookup scaffolding
        repair_code = (
            "print(apis.api_docs.show_app_descriptions())\n"
            "# Choose an app and API, then print its documentation before calling it"
        )
        return f"```python\n{repair_code}\n```"

    def act(self, obs: str) -> str:
        """Execute the next API operation based on planner guidance."""
        self._step_counter += 1

        # Generate action via LLM
        history = self.memory.recall()
        system_prompt = self.build_system_prompt()
        user_messages = self.build_user_prompt(obs, history, self.config.history_k)

        from src.utils import logger as jsonlogger
        with jsonlogger.json_log_context(call_type="api_execution"):
            messages = [{"role": "system", "content": system_prompt}] + user_messages
            resp = self.lm.call(messages)

        action_text = (resp.get("text") or "").strip()
        if not action_text:
            action_text = "print('No clear API operation determined; consulting planner...')"
            self.logger.warning("No action returned from LM, using fallback")

        # Apply guardrails if enabled
        if self.config.guardrails_enabled:
            # Extract code for validation
            code_match = re.search(r"```(?:python)?\s*\n(.*?)\n```", action_text, re.IGNORECASE | re.DOTALL)
            code = code_match.group(1).strip() if code_match else action_text

            violation = self._violates_guardrails(code)
            if violation:
                self.logger.warning("Guardrail violation: %s", violation)
                # Attempt repair
                if self.config.retry_on_api_error > 0:
                    action_text = self._repair_action(action_text, violation)
                    self.logger.info("Applied guardrail repair")
                else:
                    # Fallback to safe action
                    action_text = "```python\nprint(apis.api_docs.show_app_descriptions())\n```"
                    self.logger.warning("Guardrail violation with no repair; using safe fallback")

        # Ensure proper formatting
        action_text = self._wrap_single_python_block(action_text)

        self._last_action = action_text
        action_event = self.create_action_event(self._last_action)
        if action_event is not None:
            self.memory.update(action_event)
            self._trajectory.append(action_event)

        self.logger.info("API execution: %s",
                        action_text[:50] + "..." if len(action_text) > 50 else action_text)
        return action_text

    def observe(self, obs: str | None, feedback: dict, done: bool) -> None:
        """Observe API execution feedback and update state."""
        super().observe(obs, feedback, done)

        # Update doc lookups tracking
        if obs and "api_doc" in obs.lower():
            # Extract API names from doc lookup outputs
            api_matches = re.findall(r"'(\w+)'\s*:\s*([^,]+)", obs)
            for api_name, _ in api_matches:
                if "login" in api_name.lower() or "show_" in api_name.lower():
                    self._doc_lookups_done.add(f"supervisor.{api_name}")
                # Could extend to track other apps...

        # Update failure tracking
        score = float(feedback.get("score", 0.0) or 0.0)
        won = bool(feedback.get("won", False))
        self._last_failed = (score < 1.0 and not won)

        self.logger.debug("API Agent observed: score=%.2f, won=%s, failed=%s", score, won, self._last_failed)


class AppWorldAPICodePlannerConfig(MemoryAgentConfig):
    """Configuration for the API Code Planner component."""
    memory_config: HistoryListConfig  # type: ignore[assignment]
    history_k: int | None = 50
    system_prompt: str | None = None
    verbose: bool = True


class AppWorldAPICodePlanner(MemoryAgent):
    """API Code Planner: Generates natural language execution plans for the Code Agent.
    
    Takes high-level planner guidance and filtered APIs, produces step-by-step
    natural language plans that guide code generation.
    """

    def __init__(self, config: AppWorldAPICodePlannerConfig, logger=None):
        super().__init__(config, logger=logger)
        self._initial_task: str = ""

    def _set_system_prompt(self) -> None:
        """Use the class-defined system prompt."""
        self.system_prompt = self.build_system_prompt()

    def build_system_prompt(self) -> str:
        """Build system prompt for the code planner."""
        cfg_prompt = getattr(self.config, "system_prompt", None)
        if cfg_prompt:
            return cfg_prompt

        return (
            "You are the API CODE PLANNER in a multi-agent CUGA system for AppWorld tasks.\n\n"
            "=== YOUR ROLE ===\n"
            "You translate HIGH-LEVEL strategic guidance into DETAILED, step-by-step natural language plans "
            "that guide a Code Agent to write Python code.\n\n"
            "=== INPUT ===\n"
            "1. **Strategic Guidance**: From the Planner (e.g., 'Login to Amazon, get cart, order benches')\n"
            "2. **Filtered APIs**: Relevant API schemas (10-20 APIs, not hundreds)\n"
            "3. **Variable History**: Previously stored variables and their summaries\n\n"
            "=== YOUR RESPONSIBILITIES ===\n"
            "1. PLAN GENERATION: Break strategy into concrete, implementable steps\n"
            "   Example:\n"
            "   - Step 1: Get Amazon password from supervisor.show_account_passwords()\n"
            "   - Step 2: Login using amazon.login with email and password\n"
            "   - Step 3: Get cart contents using amazon.show_cart with access_token\n"
            "   - Step 4: VERIFY cart contains only weightlifting benches\n"
            "   - Step 5: If verified, call amazon.place_order\n\n"
            "2. API MAPPING: Match each step to specific API(s) from filtered list\n"
            "   - Reference APIs as: app_name.api_name\n"
            "   - Explain what parameters are needed and where they come from\n\n"
            "3. DATA FLOW: Explain how outputs become inputs\n"
            "   Example: 'Use the access_token from Step 2 login response in Step 3 show_cart call'\n\n"
            "4. VERIFICATION STEPS: Include checks before destructive operations\n"
            "   - Before placing orders: Verify cart contents match task requirements\n"
            "   - Before deletions: Confirm target items\n"
            "   - Before sending: Review recipients and content\n\n"
            "5. ERROR HANDLING: Note where to check for errors\n"
            "   Example: 'Check if login response has access_token; if not, handle error'\n\n"
            "6. PAGINATION: Identify when to loop through pages\n"
            "   Example: 'Loop through all pages of playlists using page_index'\n\n"
            "=== OUTPUT FORMAT ===\n"
            "Your output should be a JSON list of step descriptions:\n"
            "```json\n"
            "[\n"
            '  "Step 1: ...",\n'
            '  "Step 2: ...",\n'
            '  "Step 3: ..."\n'
            "]\n"
            "```\n\n"
            "=== CRITICAL: AVOID COLLATERAL DAMAGE ===\n"
            "For e-commerce tasks:\n"
            "- ALWAYS verify cart contents before placing orders\n"
            "- Filter items to match task requirements EXACTLY\n"
            "- Never assume cart only has target items\n\n"
            "For deletion tasks:\n"
            "- ALWAYS list items before deleting\n"
            "- Verify targets match task requirements\n"
            "- Confirm before executing deletions\n\n"
            "Your plans should be detailed enough that a Code Agent can implement them "
            "without additional strategic decisions."
        )

    def build_user_prompt(self, obs: str, history: List[Any], k: int | None) -> List[Dict[str, str]]:
        """Build user prompt with planning context."""
        messages: List[Dict[str, str]] = []

        if self._initial_task:
            messages.append({"role": "user", "content": f"Initial Task:\n{self._initial_task}"})
        
        # Add history
        recent: List[Entry] = history[-k:] if k is not None else history  # type: ignore[assignment]
        for entry in recent:
            if entry.type == "Observation":
                messages.append({"role": "user", "content": str(entry.content)})
            elif entry.type == "Action":
                messages.append({"role": "assistant", "content": str(entry.content)})
        
        # Add current observation
        messages.append({"role": "user", "content": obs})
        return messages

    def create_observation_event(self, obs: str) -> Any:
        """Create observation entry for memory."""
        return Entry(type="Observation", content=obs)

    def create_action_event(self, action: str) -> Any:
        return Entry(type="Action", content=action)

    def create_feedback_event(self, feedback: dict) -> Any:
        return Entry(type="Feedback", content=feedback)

    def act(self, obs: str) -> str:
        """Generate a natural language execution plan."""
        history = self.memory.recall()
        system_prompt = self.build_system_prompt()
        user_messages = self.build_user_prompt(obs, history, self.config.history_k)

        from src.utils import logger as jsonlogger
        with jsonlogger.json_log_context(call_type="api_code_planning"):
            messages = [{"role": "system", "content": system_prompt}] + user_messages
            resp = self.lm.call(messages)

        plan = (resp.get("text") or "").strip()
        if not plan:
            plan = '["Get credentials", "Login", "Execute task", "Complete"]'

        if plan.lower().startswith("think:"):
            formatted_plan = plan
        else:
            formatted_plan = f"think: {plan}"

        self._last_action = formatted_plan
        action_event = self.create_action_event(self._last_action)
        if action_event is not None:
            self.memory.update(action_event)
            self._trajectory.append(action_event)

        self.logger.info("Generated execution plan")
        return formatted_plan

    def set_initial_task(self, task_description: str) -> None:
        self._initial_task = task_description

    def reset(self) -> None:
        super().reset()
        self._initial_task = ""
        self.logger.debug("APICodePlanner reset")

    def end_episode(self) -> None:
        self.reset()


class AppWorldCodeAgentConfig(MemoryAgentConfig):
    """Configuration for the Code Agent (executor)."""
    memory_config: HistoryListConfig  # type: ignore[assignment]
    history_k: int | None = 50
    enable_reflection: bool = True
    system_prompt: str | None = None
    verbose: bool = True


class AppWorldCodeAgent(MemoryAgent):
    """Code Agent: Executes code based on natural language plans.
    
    Generates and executes Python code to implement the plan provided by
    APICodePlanner, with variable management and reflection.
    """

    def __init__(self, config: AppWorldCodeAgentConfig, logger=None):
        super().__init__(config, logger=logger)
        self.var_manager = VariablesManager()
        self._last_execution_output: str = ""
        self._initial_task: str = ""
        self.logger.info("AppWorldCodeAgent initialized with variable manager")

    def _set_system_prompt(self) -> None:
        """Use the class-defined system prompt."""
        self.system_prompt = self.build_system_prompt()

    def build_system_prompt(self) -> str:
        """Build system prompt for the code agent."""
        cfg_prompt = getattr(self.config, "system_prompt", None)
        if cfg_prompt:
            return cfg_prompt

        return (
            "You are the CODE AGENT in a multi-agent CUGA system for AppWorld tasks.\n\n"
            "=== YOUR ROLE ===\n"
            "You IMPLEMENT the detailed execution plan by writing Python code that calls AppWorld APIs.\n\n"
            "=== INPUT ===\n"
            "1. **Execution Plan**: Step-by-step natural language instructions\n"
            "2. **Variable History**: Previously stored variables (use these, don't recreate)\n"
            "3. **Filtered APIs**: Relevant API documentation\n\n"
            "=== YOUR RESPONSIBILITIES ===\n"
            "1. CODE GENERATION: Write Python code in ```python blocks\n"
            "   - Implement each step from the plan sequentially\n"
            "   - One focused code block per environment step\n"
            "   - Use descriptive variable names (not 'result', 'data', etc.)\n\n"
            "2. API DOCUMENTATION: Look up API specs before calling\n"
            "   - apis.api_docs.show_api_doc(app_name='...', api_name='...')\n"
            "   - Check required parameters and response structure\n\n"
            "3. VARIABLE MANAGEMENT: Store results for reuse\n"
            "   - Variables persist across steps\n"
            "   - Use meaningful names: spotify_token, cart_contents, etc.\n"
            "   - Print important values for debugging\n\n"
            "4. VERIFICATION: Include checks as specified in plan\n"
            "   - Print what you're checking\n"
            "   - Compare actual vs expected\n"
            "   - Only proceed if verification passes\n\n"
            "5. ERROR HANDLING: Check API responses\n"
            "   - Validate responses have expected fields\n"
            "   - Print errors clearly\n"
            "   - Don't proceed if critical errors occur\n\n"
            "6. PAGINATION: Loop through all pages when needed\n"
            "   - Use page_index parameter\n"
            "   - Collect all results before processing\n"
            "   - Print progress (e.g., 'Page 2: 15 items, total 30')\n\n"
            "7. TASK COMPLETION: Call supervisor.complete_task when done\n"
            "   - With answer if required: complete_task(answer=result)\n"
            "   - Without answer otherwise: complete_task()\n\n"
            "=== OUTPUT FORMAT ===\n"
            "```python\n"
            "# Your code here\n"
            "```\n\n"
            "=== SAFETY RULES ===\n"
            "- ONLY use apis.* modules and Python stdlib\n"
            "- NO os, sys, subprocess, requests, etc.\n"
            "- NO filesystem operations\n"
            "- Always validate before destructive operations\n\n"
            "Follow the plan carefully and implement it step by step."
        )

    def build_user_prompt(self, obs: str, history: List[Any], k: int | None) -> List[Dict[str, str]]:
        """Build user prompt with execution context and variable history."""
        messages: List[Dict[str, str]] = []

        if self._initial_task:
            messages.append({"role": "user", "content": f"Initial Task:\n{self._initial_task}"})

        # Add variable history summary
        var_summary = self.var_manager.get_variables_summary(last_n=5)
        if var_summary != "# No variables stored":
            messages.append({
                "role": "user",
                "content": f"Variable History:\n{var_summary}"
            })
        
        # Add recent history
        if len(history) > k:
            recent = [history[0]] + history[-k:] if k is not None else history
        else:
            recent = history
        
        for entry in recent:
            if entry.type == "Observation":
                messages.append({"role": "user", "content": str(entry.content)})
            elif entry.type == "Action":
                messages.append({"role": "assistant", "content": str(entry.content)})
        
        # Add current observation
        messages.append({"role": "user", "content": obs})
        return messages

    def create_observation_event(self, obs: str) -> Any:
        """Create observation entry for memory."""
        return Entry(type="Observation", content=obs)

    def create_action_event(self, action: str) -> Any:
        return Entry(type="Action", content=action)

    def create_feedback_event(self, feedback: dict) -> Any:
        return Entry(type="Feedback", content=feedback)

    def _strip_code_fences(self, code: str) -> str:
        if not code:
            return ""
        stripped = code.strip()
        if stripped.lower().startswith("```python"):
            stripped = stripped[len("```python"):].strip()
        if stripped.endswith("```"):
            stripped = stripped[:-3].strip()
        return stripped

    def _extract_last_assignment(self, code: str) -> Optional[tuple[str, str, str]]:
        cleaned = self._strip_code_fences(code)
        matches = re.findall(r"(\w+)\s*=\s*apis\.(\w+)\.(\w+)", cleaned)
        if not matches:
            return None
        return matches[-1]

    def _parse_structured_output(self, obs: str) -> Optional[Any]:
        if not obs:
            return None
        # Try full text first
        candidate = obs.strip()
        for text in [candidate] + [line.strip() for line in reversed(obs.splitlines()) if line.strip()]:
            if not text or text.lower() in {"ok", "ok."}:
                continue
            if text[0] not in "[{" and not text[0].isdigit() and text[0] not in "\"'":
                continue
            try:
                return json.loads(text)
            except Exception:
                try:
                    return ast.literal_eval(text)
                except Exception:
                    continue
        return None

    def act(self, obs: str) -> str:
        """Generate and return code based on the execution plan."""
        history = self.memory.recall()
        system_prompt = self.build_system_prompt()
        user_messages = self.build_user_prompt(obs, history, self.config.history_k)

        from src.utils import logger as jsonlogger
        with jsonlogger.json_log_context(call_type="code_execution"):
            messages = [{"role": "system", "content": system_prompt}] + user_messages
            resp = self.lm.call(messages)

        code = (resp.get("text") or "").strip()
        if not code:
            code = "print('No code generated')"

        # Ensure code is wrapped in ```python blocks
        if not re.search(r"```\s*python", code, re.IGNORECASE):
            code = f"```python\n{code}\n```"

        self._last_action = code
        action_event = self.create_action_event(self._last_action)
        if action_event is not None:
            self.memory.update(action_event)
            self._trajectory.append(action_event)

        self.logger.info("Generated code: %s", code[:100] + "..." if len(code) > 100 else code)
        return code

    def set_initial_task(self, task_description: str) -> None:
        self._initial_task = task_description

    def observe(self, obs: str | None, feedback: dict, done: bool) -> None:
        """Observe execution output and store in variable manager if substantial."""
        super().observe(obs, feedback, done)
        
        # Store execution output for potential variable extraction
        if obs and len(obs) > 100:
            self._last_execution_output = obs
            # Try to extract structured output for variable storage
            # This will be used by reflection to create variable summaries

        if obs:
            structured = self._parse_structured_output(obs)
            if structured is not None:
                assignment = self._extract_last_assignment(self._last_action or "")
                var_name = None
                description = "Execution output"
                if assignment:
                    var_name, app_name, api_name = assignment
                    description = f"Result returned by apis.{app_name}.{api_name}"
                stored = self.var_manager.add_variable(structured, name=var_name, description=description)
                self.logger.debug("Stored variable via VariablesManager: %s", stored)

    def generate_reflection(self, task_description: str, execution_output: str) -> str:
        """Generate reflection on the execution to guide next steps."""
        if not self.config.enable_reflection:
            return ""

        reflection_prompt = (
            "Analyze the recent code execution and provide strategic guidance.\n\n"
            "=== YOUR ROLE ===\n"
            "You are a strategic analysis agent reviewing code execution results.\n\n"
            "=== RESPONSIBILITIES ===\n"
            "1. ANALYSIS: Did the code execute successfully? Any errors?\n"
            "2. PROGRESS: What was accomplished? What variables were created?\n"
            "3. VERIFICATION: For e-commerce tasks, verify cart contents before purchase\n"
            "4. NEXT STEPS: What should happen next? Any corrections needed?\n"
            "5. SKEPTICISM: Consider edge cases and potential issues\n\n"
            "=== CRITICAL: COLLATERAL DAMAGE PREVENTION ===\n"
            "For Amazon/e-commerce tasks:\n"
            "- If placing an order, verify cart contains ONLY items mentioned in task\n"
            "- If cart has extra items, recommend filtering before ordering\n\n"
            "For deletion tasks:\n"
            "- Verify items to delete match task requirements exactly\n\n"
            "Provide a concise strategic summary (2-3 sentences) for next planning step."
        )

        # Build reflection query
        var_summary = self.var_manager.get_variables_summary(last_n=3)
        reflection_query = (
            f"Task: {task_description}\n\n"
            f"Recent Execution Output:\n{execution_output[:500]}\n\n"
            f"Variables Created:\n{var_summary}\n\n"
            "Provide strategic guidance:"
        )

        from src.utils import logger as jsonlogger
        with jsonlogger.json_log_context(call_type="reflection"):
            messages = [
                {"role": "system", "content": reflection_prompt},
                {"role": "user", "content": reflection_query}
            ]
            resp = self.lm.call(messages)

        reflection = (resp.get("text") or "").strip()
        if not reflection:
            reflection = "Execution completed. Continue with next step."

        self.logger.info("Generated reflection: %s", reflection)
        return reflection

    def reset(self) -> None:
        super().reset()
        self._initial_task = ""
        # Note: We don't reset var_manager here - variables persist across the episode
        self.logger.debug("CodeAgent reset")

    def end_episode(self) -> None:
        self.reset()
        # Reset variables at episode end
        self.var_manager.reset()
        self.logger.debug("CodeAgent end_episode - variables cleared")


class AppWorldCUGAConfig(MemoryAgentConfig):
    """Configuration for the complete CUGA system with planner, code planner, and code agent."""
    memory_config: HistoryListConfig  # type: ignore[assignment]
    history_k: int | None = 50

    # Agent configurations
    planner_config: AppWorldPlannerConfig
    code_planner_config: AppWorldAPICodePlannerConfig
    code_agent_config: AppWorldCodeAgentConfig

    # Coordination configuration
    coordination_enabled: bool = True
    max_coordination_steps: int = 50
    enable_reflection_feedback: bool = True

    # Agent behavior
    verbose: bool = True


class AppWorldCUGA(MemoryAgent):
    """Complete CUGA system coordinating planner, code planner, and code agent.

    This implements the full CUGA architecture where:
    - Planner provides high-level strategic guidance
    - Code Planner generates detailed execution plans
    - Code Agent executes code with variable management
    - Reflection provides feedback after execution
    - Coordinator manages the interaction between components
    """

    def __init__(self, config: AppWorldCUGAConfig, logger=None):
        super().__init__(config, logger=logger)

        # Initialize sub-components
        self.planner = AppWorldPlanner(config.planner_config, logger=self.logger.getChild("planner") if self.logger else None)
        self.code_planner = AppWorldAPICodePlanner(config.code_planner_config, logger=self.logger.getChild("code_planner") if self.logger else None)
        self.code_agent = AppWorldCodeAgent(config.code_agent_config, logger=self.logger.getChild("code_agent") if self.logger else None)

        # Coordination state
        self._coordination_step: int = 0
        self._current_phase: str = "planning"  # "planning", "code_planning", or "execution"
        self._last_planner_output: str = ""
        self._last_code_plan: str = ""
        self._initial_task: str = ""

        self.logger.info("AppWorldCUGA init: coordination=%s, max_steps=%d, reflection=%s",
                        self.config.coordination_enabled, self.config.max_coordination_steps,
                        self.config.enable_reflection_feedback)

    def _set_system_prompt(self) -> None:
        """Coordinator uses its internal system prompt template."""
        self.system_prompt = self.build_system_prompt()

    def build_system_prompt(self) -> str:
        """System prompt for the coordinating agent."""
        cfg_prompt = getattr(self.config, "system_prompt", None)
        if cfg_prompt:
            return cfg_prompt

        return (
            "You are the COORDINATOR in the CUGA (Computer Using Generalist Agent) system for AppWorld.\n\n"
            "=== YOUR ROLE ===\n"
            "You orchestrate collaboration between three specialist agents:\n"
            "1. PLANNER: Provides high-level strategic guidance\n"
            "2. CODE PLANNER: Generates detailed execution plans\n"
            "3. CODE AGENT: Executes code with reflection-based feedback\n\n"
            "=== YOUR RESPONSIBILITIES ===\n"
            "1. PHASE MANAGEMENT: Coordinate multi-phase execution\n"
            "   - Planning: Strategic guidance from Planner\n"
            "   - Code Planning: Detailed steps from Code Planner\n"
            "   - Execution: Code implementation with reflection feedback\n\n"
            "2. FEEDBACK INTEGRATION: Reflection provides strategic feedback\n"
            "   - After execution, reflection analyzes results\n"
            "   - Feedback guides next planning iteration\n"
            "   - Catches collateral damage issues (e.g., ordering wrong items)\n\n"
            "3. CONTEXT MANAGEMENT: Ensure agents see relevant information\n"
            "   - Variable summaries instead of full outputs\n"
            "   - Reflection feedback instead of raw observations\n"
            "   - Skip uninformative 'OK.' messages\n\n"
            "=== COORDINATION WORKFLOW ===\n"
            "1. Planning: High-level strategy\n"
            "2. Code Planning: Detailed execution steps\n"
            "3. Execution: Code runs, produces output\n"
            "4. Reflection: Analyzes output, provides feedback\n"
            "5. Return to Planning with feedback for next iteration\n\n"
            "Your role is to maintain smooth flow and integrate reflection feedback."
        )

    def build_user_prompt(self, obs: str, history: List[Any], k: int | None) -> List[Dict[str, str]]:
        """Build user prompt with coordination context.
        
        Uses alternating user/assistant format like history_agent:
        - Observations → user messages
        - Actions → assistant messages
        - Feedback → user messages
        """
        messages: List[Dict[str, str]] = []

        # Add history as alternating messages
        recent: List[Entry] = history[-k:] if k is not None else history  # type: ignore[assignment]
        
        for entry in recent:
            if entry.type == "Observation":
                messages.append({"role": "user", "content": str(entry.content)})
            elif entry.type == "Action":
                messages.append({"role": "assistant", "content": str(entry.content)})
            elif entry.type == "Feedback":
                if isinstance(entry.content, dict):
                    msg = entry.content.get("message", "")
                    messages.append({"role": "user", "content": f"Feedback: {msg}"})
                else:
                    messages.append({"role": "user", "content": f"Feedback: {str(entry.content)}"})
        
        # Add current observation as final user message
        messages.append({"role": "user", "content": obs})

        return messages

    def reset(self) -> None:
        """Reset coordinator and sub-agent state between episodes."""
        self.memory.reset()
        self._trajectory = []
        self._last_action = None
        self._coordination_step = 0
        self._current_phase = "planning"
        self._last_planner_output = ""
        self._last_code_plan = ""
        self._initial_task = ""
        self.planner.reset()
        self.code_planner.reset()
        self.code_agent.reset()

    def end_episode(self) -> None:
        """Reset state when an episode finishes."""
        self.planner.end_episode()
        self.code_planner.end_episode()
        self.code_agent.end_episode()
        self.reset()

    def create_observation_event(self, obs: str) -> Any:
        """Create observation entry for memory."""
        return Entry(type="Observation", content=obs)

    def create_action_event(self, action: str) -> Any:
        """Create coordinated action entry for memory."""
        return Entry(type="Action", content=action)

    def create_feedback_event(self, feedback: dict) -> Any:
        """Create feedback entry for memory."""
        return Entry(type="Feedback", content=feedback)

    def _get_current_agent(self):
        """Get the currently active sub-agent based on coordination phase."""
        if self._current_phase == "planning":
            return self.planner
        elif self._current_phase == "code_planning":
            return self.code_planner
        elif self._current_phase == "execution":
            return self.code_agent
        else:
            return self.code_agent  # Default to execution

    def act(self, obs: str) -> str:
        """Coordinate between planner, code planner, and code agent."""
        # Store initial task for reflection
        if self._coordination_step == 0:
            self._initial_task = obs
            self.logger.debug("Initial task stored for coordination: %s", obs[:80])
            self.planner.set_initial_task(obs)
            self.code_planner.set_initial_task(obs)
            self.code_agent.set_initial_task(obs)

        # Determine current phase based on coordination cycle
        if not self.config.coordination_enabled:
            self._current_phase = "execution"
        else:
            phase_cycle = ["planning", "code_planning", "execution"]
            phase_index = self._coordination_step % len(phase_cycle)
            self._current_phase = phase_cycle[phase_index]

        # Select the active agent for this phase
        active_agent = self._get_current_agent()
 
        # Build context-aware observation for the active agent based on current phase
        agent_obs = obs
        if self._current_phase == "code_planning":
            plan_context = self._last_planner_output or "(Planner must provide strategy first)"
            agent_obs = (
                f"Latest observation: {obs}\n"
                "Use the planner's most recent think: guidance in the assistant message above to craft the API plan."
            )
        elif self._current_phase == "execution":
            plan_context = self._last_planner_output or "(Planner has not provided strategy yet)"
            code_plan_context = self._last_code_plan or "(Code planner has not provided steps yet)"
            agent_obs = (
                f"Latest observation: {obs}\n"
                "Follow the assistance above: read the planner's think: guidance and the JSON step list from the code planner, then execute safely."
            )

        # Generate action from the active agent
        # Note: active_agent.act() will auto-store agent_obs in its own memory
        action = active_agent.act(agent_obs)
 
        # Store context for coordination
        if self._current_phase == "planning" and "think:" in action:
            self._last_planner_output = action.replace("think:", "").strip()
        elif self._current_phase == "code_planning":
            self._last_code_plan = action

        # Log coordination
        self.logger.info(
            "CUGA Coordination: phase=%s, coordination_step=%d",
            self._current_phase,
            self._coordination_step,
        )
 
        # Cross-share: broadcast the produced action so inactive agents stay aligned
        for agent in [self.planner, self.code_planner, self.code_agent]:
            if agent != active_agent:
                action_event = agent.create_action_event(action)
                if action_event is not None:
                    agent.memory.update(action_event)
                    agent._trajectory.append(action_event)

        # Store in coordinator's memory with phase label
        self._last_action = action
        action_event = self.create_action_event(f"[{self._current_phase.upper()}] {action}")
        if action_event is not None:
            self.memory.update(action_event)
            self._trajectory.append(action_event)

        # Increment coordination step after completing this phase
        self._coordination_step += 1

        return action

    def observe(self, obs: str | None, feedback: dict, done: bool) -> None:
        """Observe feedback and coordinate between sub-agents.
        
        After code execution, generates reflection and passes it as feedback
        instead of raw observations.
        """
        super().observe(obs, feedback, done)

        # Forward feedback to all sub-agents
        self.planner.observe(obs, feedback, done)
        self.code_planner.observe(obs, feedback, done)
        self.code_agent.observe(obs, feedback, done)
        
        # Skip uninformative "OK." observations from think: actions
        should_store_obs = obs is not None and obs.strip().lower() not in ["ok.", "ok"]
        
        # After code execution, generate reflection instead of storing raw observation
        reflection_generated = False
        if should_store_obs and self._current_phase == "execution" and self.config.enable_reflection_feedback:
            # Generate reflection on the execution output
            reflection = self.code_agent.generate_reflection(self._initial_task, obs or "")
            
            if reflection:
                # Store reflection as feedback observation instead of raw output
                reflection_obs = f"[REFLECTION] {reflection}"
                
                for agent in [self, self.planner, self.code_planner, self.code_agent]:
                    refl_event = agent.create_observation_event(reflection_obs)
                    if refl_event is not None:
                        agent.memory.update(refl_event)
                        agent._trajectory.append(refl_event)
                
                self.logger.info("Generated and stored reflection feedback")
                reflection_generated = True
        
        # For non-execution phases or when reflection is disabled, store observations normally
        if should_store_obs and not reflection_generated:
            for agent in [self, self.planner, self.code_planner, self.code_agent]:
                obs_event = agent.create_observation_event(obs)
                if obs_event is not None:
                    agent.memory.update(obs_event)
                    agent._trajectory.append(obs_event)
            
            self.logger.debug("Forwarded obs to all agents: %s", obs[:50] if obs else "")

        # Update coordination state based on feedback
        score = float(feedback.get("score", 0.0) or 0.0)
        won = bool(feedback.get("won", False))

        # If execution failed, may need to replan
        if score < 1.0 and not won and self._current_phase == "execution":
            self.logger.info("Execution failed (score=%.2f), will replan on next cycle", score)

        self.logger.debug("CUGA observed: phase=%s, score=%.2f, won=%s, done=%s",
                         self._current_phase, score, won, done)