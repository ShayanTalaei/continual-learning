## Language Model

- `LanguageModel` base with `call(system,user)`.
- `GeminiClient` via `lm_factory.get_lm_client` when model contains "gemini".
- Config: `model`, `temperature`, `max_output_tokens` (plus optional thinking_budget in Gemini).


