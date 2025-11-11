from typing import List, Any, Dict, Optional, Union, Set
import re
import json
import ast
from collections import defaultdict

from src.agent.memory_agent import MemoryAgent, MemoryAgentConfig
from src.memory.history_list import HistoryListConfig, Entry


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
            "Provide high-level strategic guidance once the team is grounded in the available APIs.\n"
            "Your output format is ALWAYS: think: [PLANNER] <your strategic guidance> (never code or JSON).\n\n"
            "=== API DISCOVERY & GROUNDING ===\n"
            "- Review the API Discovery Status summary in each observation.\n"
            "- If the app catalog or required API descriptions are missing, direct the Code Planner to gather them first using apis.api_docs.show_app_descriptions / show_api_descriptions / show_api_doc.\n"
            "- Do NOT advance the task plan until the required documentation is confirmed.\n\n"
            "=== PHASE DIRECTIVES ===\n"
            "Include one of the following tokens in your think message to control the pipeline:\n"
            "- PHASE: DISCOVER_APIS   (when more documentation is needed)\n"
            "- PHASE: EXECUTE_PLAN    (when documentation is sufficient to plan task steps)\n\n"
            "=== STRATEGY, REFLECTION & RECOVERY ===\n"
            "- Once documentation is available, break the task into logical sub-goals and cite the retrieved specs when guiding the team.\n"
            "- Integrate execution reflections before replanning; use them to correct errors or adjust the approach.\n"
            "- Acknowledge successful progress succinctly while steering the next step.\n\n"
            "=== COMPLETION CONTROL ===\n"
            "- The Code Agent may only call apis.supervisor.complete_task after you explicitly authorize it.\n"
            "- IMPORTANT: Always verify the outcome before authorizing completion:\n"
            "  1. Direct team to complete the task operations\n"
            "  2. Direct team to verify the outcome (re-fetch state, check results, compare values)\n"
            "  3. After seeing verification results, emit '[VERIFIED]' token to confirm success\n"
            "  4. Then emit 'AUTHORIZE_COMPLETE_TASK' to authorize completion\n"
            "  * Example for queries: 'Result is 42. [VERIFIED] AUTHORIZE_COMPLETE_TASK'\n"
            "  * Example for modifications: 'Verified both lists match. [VERIFIED] AUTHORIZE_COMPLETE_TASK'\n"
            "- BOTH [VERIFIED] and AUTHORIZE_COMPLETE_TASK tokens are required for completion\n"
            "- Otherwise remind the team to continue working and withhold both tokens.\n\n"
            "=== COMMUNICATION STYLE ===\n"
            "Always prefix with 'think: [PLANNER]' and provide concise, actionable guidance focused on WHAT to do next—not HOW to implement it."
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
                    if msg and msg.strip().lower() not in {"ok", "ok."}:
                        messages.append({"role": "user", "content": f"Feedback: {msg}"})
                else:
                    text = str(entry.content)
                    if text.strip().lower() not in {"ok", "ok."}:
                        messages.append({"role": "user", "content": f"Feedback: {text}"})
        
        # Add current observation as final user message (avoid duplicating initial task or OK.)
        if obs and obs.strip().lower() not in {"ok", "ok."}:
            if not (self._initial_observation and obs.strip() == self._initial_observation.strip()):
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
        # Plan on every planning phase until hitting max_planning_steps, or on failures
        if self._planning_step_counter < self.config.max_planning_steps:
            return True
        if self.config.replan_on_failure and self._last_failure_reason:
            return True
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
        # Strip any fenced code/backticks; enforce planning-only output
        import re as _re
        planning_guidance = _re.sub(r"```[\s\S]*?```", "", planning_guidance).strip()
        if not planning_guidance:
            planning_guidance = "Analyze the current task and break it down into executable steps using available APIs."

        # Normalize and tag source
        if planning_guidance.lower().startswith("think:"):
            planning_guidance = planning_guidance.split(":", 1)[1].strip()
        # Remove any pre-existing role tags like [PLANNER]
        planning_guidance = _re.sub(r"^\s*\[PLANNER\]\s*", "", planning_guidance, flags=_re.IGNORECASE)
        plan_action = f"think: [PLANNER] {planning_guidance}".strip()

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
                    if msg and msg.strip().lower() not in {"ok", "ok."}:
                        messages.append({"role": "user", "content": f"Feedback: {msg}"})
                else:
                    text = str(entry.content)
                    if text.strip().lower() not in {"ok", "ok."}:
                        messages.append({"role": "user", "content": f"Feedback: {text}"})
        
        # Add current observation as final user message (avoid duplicating initial task / OK)
        if obs and obs.strip().lower() not in {"ok", "ok."}:
            if not (self._initial_observation and obs.strip() == self._initial_observation.strip()):
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

        # Check for API doc lookup requirement (STRENGTHENED)
        if self.config.enforce_doc_lookup:
            api_matches = re.findall(r"apis\.(\w+)\.(\w+)\(", code)
            # Exclude api_docs calls themselves from the requirement
            non_doc_apis = [
                (app, api) for app, api in api_matches 
                if app != "api_docs"
            ]
            
            # Check if any non-doc APIs lack documentation
            missing_docs = []
            for app_name, api_name in non_doc_apis:
                api_key = f"{app_name}.{api_name}"
                if api_key not in self._doc_lookups_done:
                    missing_docs.append(api_key)
            
            if missing_docs:
                missing_str = ", ".join(missing_docs)
                return f"APIs used without prior documentation lookup: {missing_str}. Always call apis.api_docs.show_api_doc(app_name='...', api_name='...') before using an API."

        # Limit API calls per step (but allow multiple if they're doc lookups)
        non_doc_calls = [call for call in api_calls if "api_docs" not in call]
        if len(non_doc_calls) > self.config.max_api_calls_per_step:
            return f"Too many non-doc API calls ({len(non_doc_calls)}) in one step. Limit to {self.config.max_api_calls_per_step}."

        return None

    def _repair_action(self, original_action: str, violation: str) -> str:
        """Attempt to repair a guardrail-violating action by adding doc lookups."""
        if "documentation lookup" in violation.lower() or "APIs used without prior" in violation:
            # Extract all missing APIs from the violation message
            missing_apis = []
            if ":" in violation:
                # Parse "APIs used without prior documentation lookup: app1.api1, app2.api2"
                parts = violation.split(":", 1)
                if len(parts) > 1:
                    api_str = parts[1].split(".")[0:2] if "." in parts[1] else []
                    # More robust: extract all app.api patterns
                    missing_apis = re.findall(r"(\w+)\.(\w+)", violation)
            
            # If we didn't parse from message, extract from original code
            if not missing_apis:
                api_matches = re.findall(r"apis\.(\w+)\.(\w+)\(", original_action)
                missing_apis = [(app, api) for app, api in api_matches if app != "api_docs"]
            
            # Generate batched doc lookups for all missing APIs
            if missing_apis:
                repair_lines = ["# Looking up API documentation before use"]
                for app_name, api_name in missing_apis[:5]:  # Limit to 5 to avoid overflow
                    repair_lines.append(
                        f"print(apis.api_docs.show_api_doc(app_name='{app_name}', api_name='{api_name}'))"
                    )
                repair_lines.append("# Review the documentation above before proceeding")
                repair_code = "\n".join(repair_lines)
                return f"```python\n{repair_code}\n```"

        # Default repair: app catalog lookup
        repair_code = (
            "# Start by understanding available apps and their APIs\n"
            "print(apis.api_docs.show_app_descriptions())\n"
            "# Then use show_api_descriptions(app_name='...') and show_api_doc(app_name='...', api_name='...') before calling APIs"
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

        # Update doc lookups tracking - parse execution output for show_api_doc results
        if obs:
            # Look for API doc output patterns: "app_name": "...", "api_name": "..."
            doc_patterns = [
                r'"app_name"\s*:\s*"(\w+)".*?"api_name"\s*:\s*"(\w+)"',
                r"'app_name'\s*:\s*'(\w+)'.*?'api_name'\s*:\s*'(\w+)'",
            ]
            for pattern in doc_patterns:
                matches = re.findall(pattern, obs, re.DOTALL)
                for app_name, api_name in matches:
                    api_key = f"{app_name}.{api_name}"
                    self._doc_lookups_done.add(api_key)
                    self.logger.debug("Registered API doc lookup: %s", api_key)
            
            # Also track when we see API descriptions (less specific but still useful)
            if "show_api_descriptions" in obs.lower() or "show_app_descriptions" in obs.lower():
                # Extract app names from context
                app_matches = re.findall(r'"name"\s*:\s*"(\w+)"', obs)
                for app_name in app_matches:
                    # Mark all common APIs for this app as "seen"
                    for common_api in ["login", "logout", "show_account"]:
                        self._doc_lookups_done.add(f"{app_name}.{common_api}")

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
            "Generate the NEXT immediate step(s) that the Code Agent should execute based on the planner's guidance.\n\n"
            "=== KEY PRINCIPLE: INCREMENTAL PLANNING ===\n"
            "- Plan only 1-3 immediate next steps, NOT the entire task\n"
            "- Each planning cycle advances the task incrementally\n"
            "- Let execution results inform subsequent planning\n"
            "- Example: If planner says 'Get phone contacts', plan ONLY the login + fetch steps, not the entire sync\n\n"
            "=== API DISCOVERY MODE ===\n"
            "- While required API documentation is missing, produce 1-2 steps for gathering the NEXT needed doc\n"
            "- Example: ['Call apis.api_docs.show_app_descriptions() to list available apps']\n"
            "- After execution, you'll be called again to plan the next doc lookup\n\n"
            "=== EXECUTION PLANNING MODE ===\n"
            "- Break the planner's IMMEDIATE guidance into 1-3 concrete executable steps\n"
            "- Map each step to specific APIs (app_name.api_name) with required parameters\n"
            "- Include verification/error-handling notes when relevant\n"
            "- DEDUPLICATION: When the current step involves adding/removing multiple items:\n"
            "  * Plan explicit deduplication: resolve identifiers → deduplicate using set → iterate unique values\n"
            "  * Example: ['Create set of unique emails from resolved identifiers, then iterate to add each']\n\n"
            "- VERIFICATION: Always plan verification before task completion:\n"
            "  * For queries: Plan to capture and verify the result\n"
            "  * For modifications: Plan to re-fetch state and confirm changes applied\n"
            "  * For comparisons: Plan to re-fetch both states and verify they match\n"
            "  * Example: ['Modify items', 'Re-fetch and verify changes', 'Complete task']\n"
            "  * Never plan completion without verification\n\n"
            "=== OUTPUT FORMAT (STRICT) ===\n"
            "Your response: think: [CODE_PLANNER] followed by a JSON array of 1-3 step strings.\n"
            "- Return a SHORT JSON array (1-3 steps only)\n"
            "- Never emit Python code or executable blocks\n"
            "- Base steps on documented APIs and immediate planner guidance\n"
            "- Use exact parameter names from API specs\n\n"
            "Example good outputs:\n"
            "- ['Call supervisor.show_account_passwords() to get credentials']\n"
            "- ['Login to phone app with credentials', 'Fetch first page of contacts with page_limit=20']\n"
            "- ['Create set of unique emails from contacts', 'Compare with Venmo friends to identify differences']\n\n"
            "Example BAD output:\n"
            "- [17 steps describing the entire task from start to finish] ❌"
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
        
        # Add current observation (avoid duplicating initial task / OK)
        if obs and obs.strip().lower() not in {"ok", "ok."}:
            if not (self._initial_task and obs.strip() == self._initial_task.strip()):
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
        # Normalize and tag source (no parsing/sanitization; rely on prompt)
        if plan.lower().startswith("think:"):
            plan = plan.split(":", 1)[1].strip()
        plan = re.sub(r"^\s*\[(CODE_PLANNER|PLANNER)\]\s*", "", plan, flags=re.IGNORECASE)
        formatted_plan = f"think: [CODE_PLANNER] {plan}"

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
        """Reset code planner state between episodes."""
        self.memory.reset()
        self._trajectory = []
        self._last_action = None
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
        self._last_execution_output: str = ""
        self._initial_task: str = ""
        self.logger.info("AppWorldCodeAgent initialized")

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
            "1. API DOCUMENTATION (REQUIRED): ALWAYS look up API specs before calling\n"
            "   - MANDATORY: apis.api_docs.show_api_doc(app_name='...', api_name='...') before each new API\n"
            "   - Check required parameters, types, constraints, and response structure\n"
            "   - Never guess parameter names or formats - always verify from docs\n"
            "   - Example: Before calling phone.login(), first print(apis.api_docs.show_api_doc(app_name='phone', api_name='login'))\n\n"
            "2. CODE GENERATION: Write Python code in ```python blocks\n"
            "   - Implement each step from the plan sequentially\n"
            "   - One focused code block per environment step\n"
            "   - Use descriptive variable names (not 'result', 'data', etc.)\n\n"
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
            "7. DEDUPLICATION: Prevent duplicate operations\n"
            "   - When processing multiple identifiers (emails, phone numbers, IDs), deduplicate first\n"
            "   - Pattern: resolve identifiers → collect in set → iterate over unique values\n"
            "   - Example: unique_emails = set(); for phone in phones: user = search(phone); if user: unique_emails.add(user['email'])\n"
            "   - Track processed items to avoid 'already exists' errors: processed = set()\n\n"
            "8. TASK COMPLETION: Call supervisor.complete_task when done\n"
            "   - IMPORTANT: ALWAYS verify the outcome before completing any task\n"
            "   - Pattern: Perform operations → Verify outcome → Then complete\n"
            "   - For queries: Verify you have the correct result\n"
            "   - For modifications: Re-fetch state and confirm changes applied correctly\n"
            "   - For comparisons: Re-fetch both states and verify they match\n"
            "   - Example: After syncing lists, re-fetch both and print comparison showing they match\n"
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

        # No variable summary (variable manager removed)
        
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
        text = obs.strip()

        # Remove leading labels like 'Output:'
        if text.lower().startswith("output:"):
            text = text.split(":", 1)[1].strip()

        # Prefer content inside the last triple backtick code fence
        import re
        fence_blocks = re.findall(r"```(?:\w+)?\s*([\s\S]*?)```", text, re.IGNORECASE)
        candidates: list[str] = []
        if fence_blocks:
            last_block = fence_blocks[-1].strip()
            candidates.append(last_block)

        # Also consider the whole text and individual lines (from bottom up)
        candidates.append(text)
        candidates.extend([line.strip() for line in reversed(text.splitlines()) if line.strip()])

        for cand in candidates:
            if not cand or cand.lower() in {"ok", "ok."}:
                continue
            # Heuristic: try to extract JSON-looking region within the candidate
            json_region = cand
            m = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])\s*$", cand)
            if m:
                json_region = m.group(1)
            try:
                return json.loads(json_region)
            except Exception:
                try:
                    return ast.literal_eval(json_region)
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

        # Variable storage removed

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
        reflection_query = (
            f"Task: {task_description}\n\n"
            f"Recent Execution Output:\n{execution_output[:500]}\n\n"
            "Variables Created: (variable manager disabled)\n\n"
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
        """Reset code agent state between episodes."""
        self.memory.reset()
        self._trajectory = []
        self._last_action = None
        self._last_execution_output = ""
        self._initial_task = ""
        try:
            self._final_task = ""  # retained for backward compatibility if present
        except Exception:
            pass
        self.logger.debug("CodeAgent reset")

    def end_episode(self) -> None:
        self.reset()
        self.logger.debug("CodeAgent end_episode")


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
        self._initial_task: str = ""  # cleaned task for planner/code planner
        self._initial_task_raw: str = ""  # full initial task (with demonstrations) for code agent
        self._last_execution_observation: str = ""
        self._last_action_phase: str = "planning"
        self._planner_mode: str = "EXECUTE_PLAN"  # or "DISCOVER_APIS"
        self._required_apps: Set[str] = set()
        self._api_knowledge: defaultdict[str, Dict[str, Any]] = defaultdict(
            lambda: {"descriptions": False, "docs": set()}
        )
        self._global_apps_listed: bool = False
        self._pending_doc_requests: List[Dict[str, Any]] = []
        self._planner_authorized_completion: bool = False
        self._planner_verified: bool = False

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
        self._initial_task_raw = ""
        self._last_execution_observation = ""
        self._last_action_phase = "planning"
        self._required_apps = set()
        self._api_knowledge = defaultdict(lambda: {"descriptions": False, "docs": set()})
        self._global_apps_listed = False
        self._pending_doc_requests = []
        self._planner_authorized_completion = False
        self._planner_verified = False
        self._last_failed = False
        self._verification_done = False
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

    def _should_skip_planning(self) -> bool:
        """Determine if we can skip planning phases for efficiency."""
        # Skip if planner just said "Continue" - no new strategy needed
        if self._last_planner_output:
            lower_output = self._last_planner_output.lower()
            if "continue" in lower_output and len(lower_output) < 50:
                return True
        
        # Skip if we're executing a clear plan without errors
        if self._last_code_plan and not self._last_failed:
            # Check if we've already completed most of the plan
            # If execution is progressing smoothly, keep executing
            return True
        
        return False

    def _needs_api_discovery(self) -> bool:
        """Check if we still need to discover APIs."""
        if not self._global_apps_listed:
            return True
        if self._required_apps and not self._has_required_docs():
            return True
        return False

    def _generate_batched_api_discovery_code(self) -> str | None:
        """Generate code to batch multiple API doc lookups efficiently."""
        if not self._needs_api_discovery():
            return None
        
        code_lines = []
        
        # Step 1: List all apps if not done
        if not self._global_apps_listed:
            code_lines.append("# Discovering available applications")
            code_lines.append("print(apis.api_docs.show_app_descriptions())")
            return f"```python\n{chr(10).join(code_lines)}\n```"
        
        # Step 2: Batch API descriptions for missing apps
        missing_apps = self._missing_required_apps()
        if missing_apps:
            code_lines.append("# Discovering APIs for required applications")
            for app in missing_apps[:3]:  # Batch up to 3 apps at once
                code_lines.append(f"print('\\n=== {app.upper()} APIs ===')")
                code_lines.append(f"print(apis.api_docs.show_api_descriptions(app_name='{app}'))")
            return f"```python\n{chr(10).join(code_lines)}\n```"
        
        return None

    def _determine_next_phase(self) -> str:
        """Adaptive phase selection to minimize coordination overhead."""
        
        # Fast-track: API discovery without planning overhead
        if self._needs_api_discovery():
            return "execution"  # Go straight to doc gathering
        
        # Initial planning: Need strategy at start
        if self._coordination_step == 0:
            return "planning"
        
        # Re-plan after failures
        if self._last_failed:
            return "planning"
        
        # Re-plan after completion attempts (success or failure)
        if self._last_action and "complete_task" in self._last_action:
            return "planning"
        
        # Normal coordination cycle when needed
        # Use a mini 2-phase cycle: planning -> execution (skip code_planner when plan is fresh)
        if self._current_phase == "planning":
            return "code_planning"
        elif self._current_phase == "code_planning":
            return "execution"
        else:
            # After execution, decide if we can skip re-planning
            if self._should_skip_planning():
                return "execution"
            
            # Otherwise, check if we need replanning
            if self._coordination_step % 3 == 0:  # Replan every 3 execution steps
                return "planning"
            return "execution"

    def _initialize_api_discovery_state(self) -> None:
        """Initialize API discovery tracking."""
        self._required_apps = set()
        self._api_knowledge = defaultdict(lambda: {"descriptions": False, "docs": set()})
        self._global_apps_listed = False
        self._pending_doc_requests = []
        self._planner_authorized_completion = False
        self._planner_verified = False
        self._last_failed = False
        self._verification_done = False

    def _determine_required_apps(self, task_text: str) -> Set[str]:
        """Infer which apps are likely required from the task description."""
        text = (task_text or "").lower()
        app_keywords: Dict[str, List[str]] = {
            "spotify": ["spotify"],
            "venmo": ["venmo"],
            "phone": ["phone", "contact", "contacts"],
            "calendar": ["calendar"],
            "gmail": ["gmail", "email"],
            "todoist": ["todoist"],
            "simple_note": ["simple note", "simplenote", "notes", "note"],
            "splitwise": ["splitwise"],
            "amazon": ["amazon"],
            "file_system": ["file system", "filesystem", "files"],
            "bank": ["bank"],
            "uber": ["uber"],
            "lyft": ["lyft"],
            "messages": ["sms", "message", "messages", "text"],
        }
        inferred: Set[str] = set()
        for app, keywords in app_keywords.items():
            if any(keyword in text for keyword in keywords):
                inferred.add(app)

        # Venmo sync tasks typically require phone contacts.
        if "venmo" in inferred:
            inferred.add("phone")

        # If contacts are mentioned explicitly, ensure phone app is present.
        if "contact" in text and "phone" not in inferred:
            inferred.add("phone")

        # Many operations require supervisor credentials.
        if inferred:
            inferred.add("supervisor")

        return inferred

    def _register_doc_requests(self, code: str) -> None:
        """Track documentation requests issued in the current execution code."""
        requests = self._extract_doc_requests_from_code(code)
        if requests:
            self._pending_doc_requests.extend(requests)

    def _extract_doc_requests_from_code(self, code: str) -> List[Dict[str, Any]]:
        """Extract api_docs requests from generated code."""
        if not code:
            return []
        body = re.sub(r"```(?:python)?", "", code)
        body = body.replace("```", "")
        requests: List[Dict[str, Any]] = []
        if re.search(r"apis\.api_docs\.show_app_descriptions\s*\(", body):
            requests.append({"type": "app_catalog"})
        for app in re.findall(r"apis\.api_docs\.show_api_descriptions\(\s*app_name\s*=\s*['\"]([a-zA-Z0-9_]+)['\"]\s*\)", body):
            requests.append({"type": "api_descriptions", "app": app.lower()})
        for match in re.findall(
            r"apis\.api_docs\.show_api_doc\(\s*app_name\s*=\s*['\"]([a-zA-Z0-9_]+)['\"]\s*,\s*api_name\s*=\s*['\"]([a-zA-Z0-9_]+)['\"]\s*\)",
            body,
        ):
            requests.append({"type": "api_doc", "app": match[0].lower(), "api": match[1]})
        return requests

    def _has_required_docs(self) -> bool:
        """Return True when required API documentation has been gathered."""
        if not self._global_apps_listed:
            return False
        for app in self._required_apps:
            if not self._api_knowledge[app]["descriptions"]:
                return False
        return True

    def _missing_required_apps(self) -> List[str]:
        """List apps still missing API description coverage."""
        return sorted(
            app for app in self._required_apps if not self._api_knowledge[app]["descriptions"]
        )

    def _get_api_status_message(self) -> str:
        """Summarize API discovery status for downstream agents."""
        lines = [
            f"App catalog fetched: {'Yes' if self._global_apps_listed else 'No'}",
        ]
        if self._required_apps:
            for app in sorted(self._required_apps):
                info = self._api_knowledge[app]
                docs = sorted(info["docs"]) if info["docs"] else []
                doc_text = ", ".join(docs) if docs else "None"
                lines.append(
                    f"{app}: api_descriptions={'Yes' if info['descriptions'] else 'No'}; api_docs={doc_text}"
                )
        else:
            lines.append(
                "Required apps inferred from task: None detected yet. Use API discovery outputs to clarify."
            )
        return "API Discovery Status:\n- " + "\n- ".join(lines)

    def _build_planner_observation(self, base_obs: str, execution_context: str) -> str:
        """Compose planner observation without repeating API discovery status."""
        parts: List[str] = []
        if base_obs:
            parts.append(base_obs)
        # Reflection is already fed via history; avoid duplicating raw outputs here
        return "\n\n".join(part for part in parts if part).strip()

    def _build_code_planner_observation(self, base_obs: str, execution_context: str) -> str:
        """Compose code planner observation with concise directives only when needed."""
        parts: List[str] = []
        if self._planner_mode == "DISCOVER_APIS":
            parts.append(
                "Code Planner directive: Produce a JSON plan of steps for the Code Agent to gather API documentation (apis.api_docs.show_app_descriptions / show_api_descriptions / show_api_doc). Do not include any Python code."
            )
        else:
            parts.append(
                "Code Planner directive: Produce a JSON plan of executable steps for the Code Agent. For each step, be explicit about app.api, required params with variable sources, and expected output variables. Do not include any Python code."
            )
        if base_obs:
            parts.append(base_obs)
        # Reflection is in history; avoid duplicating raw outputs
        return "\n\n".join(part for part in parts if part).strip()

    def _complete_doc_requests(self) -> None:
        """Mark pending documentation requests as fulfilled."""
        for req in self._pending_doc_requests:
            req_type = req.get("type")
            if req_type == "app_catalog":
                self._global_apps_listed = True
            elif req_type == "api_descriptions":
                app = req.get("app")
                if app:
                    info = self._api_knowledge[app]
                    info["descriptions"] = True
            elif req_type == "api_doc":
                app = req.get("app")
                api_name = req.get("api")
                if app and api_name:
                    info = self._api_knowledge[app]
                    docs: Set[str] = info["docs"]
                    docs.add(api_name)
        self._pending_doc_requests = []

    def _maybe_block_completion(self, code: str) -> str:
        """Smart completion authorization - require verification for all tasks."""
        if "apis.supervisor.complete_task" not in code:
            return code
        
        # Allow if planner explicitly authorized with verification
        if self._planner_authorized_completion:
            # Always require [VERIFIED] token for proper completion
            if not self._planner_verified:
                self.logger.warning(
                    "Blocking completion: Missing [VERIFIED] token from planner"
                )
                return (
                    "```python\n"
                    "print('Completion requires verification. Verify the outcome, then await planner [VERIFIED] token.')\n"
                    "```"
                )
            self._planner_authorized_completion = False
            self._planner_verified = False
            return code
        
        # Check if planner emitted [VERIFIED] token (strongest signal)
        if self._planner_verified:
            self.logger.info("Auto-authorizing completion: planner emitted [VERIFIED]")
            self._planner_verified = False
            return code
        
        # Smart auto-authorization: Allow if verification was done
        # Look for evidence of actual state verification (not just API doc checks)
        recent_history = self.memory.recall()[-10:]  # Last 10 steps
        
        # Require stronger evidence: re-fetching state after modifications
        verification_patterns = [
            "verif",  # verify, verification
            "re-fetch", "refetch",
            "confirm",
            "final state", "final check",
            "compare", "match",
            "after add", "after remov", "after updat",  # checking state after changes
        ]
        
        # Exclude false positives like "check documentation"
        exclude_patterns = ["check.*doc", "check.*api", "check.*spec"]
        
        verification_done = False
        for entry in recent_history:
            content_lower = str(entry.content).lower()
            # Check if it contains verification patterns
            has_verification = any(pattern in content_lower for pattern in verification_patterns)
            # But not if it's just about checking documentation
            is_false_positive = any(
                re.search(exclude, content_lower) for exclude in exclude_patterns
            )
            if has_verification and not is_false_positive:
                verification_done = True
                break
        
        if verification_done:
            self.logger.info("Auto-authorizing completion: verification detected in recent history")
            return code
        
        # Auto-authorize if we've made substantial progress (20+ coordination steps)
        if self._coordination_step > 20:
            self.logger.info("Auto-authorizing completion: substantial progress made (%d steps)", 
                           self._coordination_step)
            return code
        
        # Auto-authorize if this is a retry after being blocked
        if any("planner has not authorized" in str(entry.content).lower() for entry in recent_history[-3:]):
            self.logger.info("Auto-authorizing completion: retry after previous block")
            return code
        
        # Block only if completion seems premature (early in execution)
        if self._coordination_step < 10:
            self.logger.warning(
                "Blocking potentially premature completion (step %d). Task needs more work or planner authorization.",
                self._coordination_step
            )
            self._pending_doc_requests = []
            return (
                "```python\n"
                "print('Task completion seems premature. Continue working or await planner authorization.')\n"
                "```"
            )
        
        # Final check: Always block if no verification evidence
        if not verification_done:
            self.logger.warning(
                "Blocking completion: No verification detected"
            )
            return (
                "```python\n"
                "print('Task completion requires verification. Verify the outcome, then await planner [VERIFIED] and AUTHORIZE_COMPLETE_TASK tokens.')\n"
                "```"
            )
        
        # Default: allow completion
        return code

    def _extract_clean_task(self, obs: str) -> str:
        """Extract a concise task statement from the initial observation.
        - Prefer the line starting with 'Task:'
        - Optionally include a 'My name is' line if present
        - Strip code fences and long examples
        """
        import re as _re
        lines = obs.splitlines()
        last_task_line = ""
        last_task_idx = -1
        # Capture the last occurrence of a Task: line
        for idx, ln in enumerate(lines):
            s = ln.strip()
            if s.lower().startswith("task:"):
                last_task_line = s
                last_task_idx = idx

        # Fallback if no explicit Task: line found
        if last_task_idx == -1:
            no_fences = _re.sub(r"```[\s\S]*?```", "", obs).strip()
            fallback = (no_fences[:200] + ("..." if len(no_fences) > 200 else "")) or "Task: (unspecified)"
            return fallback

        # Find the nearest preceding "My name is" line (if any)
        name_line = ""
        for j in range(last_task_idx, -1, -1):
            s = lines[j].strip()
            if s.lower().startswith("my name is"):
                name_line = s
                break

        return f"{name_line}\n{last_task_line}" if name_line else last_task_line

    def _get_recent_execution_output(self, max_chars: int = 1500) -> str:
        """Return the most recent raw execution output for context sharing."""
        if not self._last_execution_observation:
            return ""

        text = self._last_execution_observation.strip()
        if not text:
            return ""

        if len(text) > max_chars:
            return text[:max_chars] + "\n... [truncated]"
        return text

    def act(self, obs: str) -> str:
        """Coordinate between planner, code planner, and code agent."""
        # Store initial task for reflection
        if self._coordination_step == 0:
            self._initial_task = self._extract_clean_task(obs)
            self._initial_task_raw = obs
            self.logger.debug("Initial task stored for coordination: %s", obs[:80])
            # Clean task for planning agents; raw task (with demos) for code agent
            self.planner.set_initial_task(self._initial_task)
            self.code_planner.set_initial_task(self._initial_task)
            self.code_agent.set_initial_task(self._initial_task_raw)
            self._initialize_api_discovery_state()

        # Determine current phase adaptively to minimize overhead
        if not self.config.coordination_enabled:
            self._current_phase = "execution"
        else:
            self._current_phase = self._determine_next_phase()

        # Fast-track batched API discovery without agent overhead
        if self._current_phase == "execution" and self._needs_api_discovery():
            batched_code = self._generate_batched_api_discovery_code()
            if batched_code:
                self.logger.info("Using batched API discovery (fast-track)")
                self._last_action = batched_code
                self._register_doc_requests(batched_code)
                
                # Store in memories
                action_event = self.create_action_event(f"[DISCOVERY] {batched_code}")
                if action_event is not None:
                    self.memory.update(action_event)
                    self._trajectory.append(action_event)
                
                # Also store in code agent for continuity
                code_agent_event = self.code_agent.create_action_event(batched_code)
                if code_agent_event is not None:
                    self.code_agent.memory.update(code_agent_event)
                    self.code_agent._trajectory.append(code_agent_event)
                
                self._coordination_step += 1
                return batched_code

        # Select the active agent for this phase
        active_agent = self._get_current_agent()
 
        # Build context-aware observation for the active agent based on current phase
        obs_text = obs or ""
        execution_context = self._get_recent_execution_output()
        agent_obs = obs_text

        if self._current_phase == "planning":
            if self._coordination_step == 0:
                base_obs = self._initial_task
            else:
                base_parts = []
                if self._initial_task:
                    base_parts.append(self._initial_task)
                if obs_text:
                    base_parts.append(f"Latest update:\n{obs_text}")
                base_obs = "\n\n".join(part for part in base_parts if part) or self._initial_task
            agent_obs = self._build_planner_observation(base_obs or "", execution_context)
        elif self._current_phase == "code_planning":
            base_parts = []
            if self._initial_task:
                base_parts.append(self._initial_task)
            if obs_text:
                base_parts.append(obs_text)
            base_obs = "\n\n".join(part for part in base_parts if part)
            agent_obs = self._build_code_planner_observation(base_obs, execution_context)
        elif self._current_phase == "execution":
            agent_obs = obs_text

        # Generate action from the active agent
        # Note: active_agent.act() will auto-store agent_obs in its own memory
        action = active_agent.act(agent_obs)
 
        # Store context for coordination
        if self._current_phase == "planning" and "think:" in action:
            self._last_planner_output = action.replace("think:", "").strip()
            up = action.upper()
            if "PHASE: DISCOVER_APIS" in up:
                self._planner_mode = "DISCOVER_APIS"
            elif "PHASE: EXECUTE_PLAN" in up:
                self._planner_mode = "EXECUTE_PLAN"
        elif self._current_phase == "code_planning":
            self._last_code_plan = action

        if self._current_phase == "planning":
            action_upper = action.upper()
            if "AUTHORIZE_COMPLETE_TASK" in action_upper:
                self._planner_authorized_completion = True
            if "[VERIFIED]" in action_upper:
                self._planner_verified = True
        elif self._current_phase == "execution":
            self._register_doc_requests(action)
            action = self._maybe_block_completion(action)

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

        self._last_action_phase = self._current_phase
        # Increment coordination step after completing this phase
        self._coordination_step += 1

        return action

    def observe(self, obs: str | None, feedback: dict, done: bool) -> None:
        """Observe feedback and coordinate between sub-agents.
        
        After code execution, generates reflection and passes it as feedback
        instead of raw observations.
        """
        # Cache raw execution output for planner/code planner context injection
        if (
            obs
            and self._last_action_phase == "execution"
            and obs.strip().lower() not in ["ok", "ok."]
        ):
            self._last_execution_observation = obs

        # Decide what to store/forward: reflection (after execution) or raw obs
        should_store_obs = obs is not None and obs.strip().lower() not in ["ok.", "ok"]
        reflection_obs: Optional[str] = None
        if should_store_obs and self._last_action_phase == "execution" and self.config.enable_reflection_feedback:
            reflection = self.code_agent.generate_reflection(self._initial_task, obs or "")
            if reflection:
                reflection_obs = f"[REFLECTION] {reflection}"
                self.logger.info("Generated reflection feedback")

        obs_to_store = reflection_obs if reflection_obs is not None else obs

        # Store chosen observation in coordinator memory
        super().observe(obs_to_store, feedback, done)

        if self._pending_doc_requests:
            if obs and "Exception" not in (obs or "") and "Traceback" not in (obs or ""):
                self._complete_doc_requests()
            else:
                self._pending_doc_requests = []

        # Additionally, audit-log raw execution output in coordinator memory and share with planner/code_planner/code_agent
        if should_store_obs and self._last_action_phase == "execution" and obs:
            exec_text = f"[EXECUTION] {obs}"
            # Coordinator
            exec_event = self.create_observation_event(exec_text)
            if exec_event is not None:
                self.memory.update(exec_event)
                self._trajectory.append(exec_event)
            # Planner, Code Planner, and Code Agent
            for agent in [self.planner, self.code_planner, self.code_agent]:
                ag_evt = agent.create_observation_event(exec_text)
                if ag_evt is not None:
                    agent.memory.update(ag_evt)
                    agent._trajectory.append(ag_evt)

        # Forward chosen observation to sub-agents
        self.planner.observe(obs_to_store, feedback, done)
        self.code_planner.observe(obs_to_store, feedback, done)
        # CodeAgent benefits from raw execution output for variable/doc tracking
        code_agent_obs = obs if (self._last_action_phase == "execution" and obs is not None) else obs_to_store
        self.code_agent.observe(code_agent_obs, feedback, done)
        # Ensure reflection is also visible in Planner and Code Planner histories
        if reflection_obs is not None:
            for agent in [self.planner, self.code_planner, self.code_agent]:
                refl_event = agent.create_observation_event(reflection_obs)
                if refl_event is not None:
                    agent.memory.update(refl_event)
                    agent._trajectory.append(refl_event)
        if obs_to_store is not None:
            self.logger.debug("Forwarded obs: planner/code_planner=%s | code_agent=%s",
                              obs_to_store[:50], (code_agent_obs or "")[:50])

        # Update coordination state based on feedback
        score = float(feedback.get("score", 0.0) or 0.0)
        won = bool(feedback.get("won", False))
        
        # Track failures for adaptive phase selection
        if obs:
            lower_obs = obs.lower()
            failure_indicators = ["execution failed", "traceback", "error", "exception"]
            self._last_failed = any(indicator in lower_obs for indicator in failure_indicators)
        else:
            self._last_failed = False

        self.logger.debug("CUGA observed: phase=%s, score=%.2f, won=%s, done=%s, failed=%s",
                         self._last_action_phase, score, won, done, self._last_failed)