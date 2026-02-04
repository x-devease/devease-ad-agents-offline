"""
LLM Client for Diagnoser Agents.

Integrates with Anthropic Claude API for real agent interactions.
"""

import os
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """Response from LLM."""
    content: str
    model: str
    input_tokens: int
    output_tokens: int
    stop_reason: str


class LLMClient:
    """
    Client for interacting with Anthropic Claude API.

    Supports multiple models, configurable parameters, and threshold injection.
    """

    DEFAULT_MODEL = "claude-3-5-sonnet-20241022"
    MAX_TOKENS = 8192
    TEMPERATURE = 0.7

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DEFAULT_MODEL,
        max_tokens: int = MAX_TOKENS,
        temperature: float = TEMPERATURE,
        enable_threshold_injection: bool = True,
    ):
        """
        Initialize LLM client.

        Args:
            api_key: Anthropic API key (defaults to ANTHROPIC_API_KEY env var)
            model: Model to use
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature (0-1)
            enable_threshold_injection: Whether to inject threshold snapshot into prompts
        """
        if not ANTHROPIC_AVAILABLE:
            raise ImportError(
                "anthropic package is required. Install with: pip install anthropic"
            )

        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY environment variable must be set"
            )

        self.client = anthropic.Anthropic(api_key=self.api_key)
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.enable_threshold_injection = enable_threshold_injection

        # Load threshold snapshot if enabled
        self.threshold_snapshot = None
        if enable_threshold_injection:
            self.threshold_snapshot = self._load_threshold_snapshot()

        logger.info(f"LLM Client initialized with model: {model}")

    def _load_threshold_snapshot(self) -> Optional[Dict[str, Any]]:
        """
        Load threshold snapshot from JSON file.

        Returns:
            Threshold snapshot dict or None if file not found
        """
        snapshot_path = Path(__file__).parent / "prompts" / "threshold_snapshot.json"

        if not snapshot_path.exists():
            logger.warning(f"Threshold snapshot not found at {snapshot_path}")
            return None

        try:
            with open(snapshot_path, 'r') as f:
                snapshot = json.load(f)

            logger.info(f"Loaded threshold snapshot (commit: {snapshot.get('git_commit', 'unknown')[:8]})")
            return snapshot
        except Exception as e:
            logger.error(f"Failed to load threshold snapshot: {e}")
            return None

    def inject_thresholds(self, prompt: str) -> str:
        """
        Inject threshold snapshot into prompt template.

        Replaces {THRESHOLD_SNAPSHOT} placeholder with actual threshold values.

        Args:
            prompt: Prompt template string

        Returns:
            Prompt with injected thresholds
        """
        if not self.threshold_snapshot:
            logger.debug("No threshold snapshot available for injection")
            return prompt

        # Format thresholds for injection
        thresholds_str = json.dumps(
            self.threshold_snapshot["detectors"],
            indent=2,
            ensure_ascii=False,
        )

        # Replace placeholder
        injected_prompt = prompt.replace("{THRESHOLD_SNAPSHOT}", thresholds_str)

        # Also replace individual detector thresholds
        for detector_name, detector_data in self.threshold_snapshot["detectors"].items():
            thresholds = detector_data["thresholds"]
            for key, value in thresholds.items():
                placeholder = f"{{THRESHOLD:{detector_name}.{key}}}"
                injected_prompt = injected_prompt.replace(placeholder, str(value))

        return injected_prompt

    def send_message(
        self,
        system_prompt: str,
        user_message: str,
        response_format: Optional[Dict[str, str]] = None,
        timeout: int = 60,
    ) -> LLMResponse:
        """
        Send message to Claude.

        Args:
            system_prompt: System prompt with role instructions
            user_message: User message with task/context
            response_format: Optional format specification (e.g., {"type": "json_object"})
            timeout: Request timeout in seconds

        Returns:
            LLMResponse with content and metadata
        """
        logger.debug(f"Sending message to {self.model}")
        logger.debug(f"System prompt length: {len(system_prompt)}")
        logger.debug(f"User message length: {len(user_message)}")

        messages = [{"role": "user", "content": user_message}]

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                system=system_prompt,
                messages=messages,
            )

            content = response.content[0].text
            input_tokens = response.usage.input_tokens
            output_tokens = response.usage.output_tokens
            stop_reason = response.stop_reason

            logger.info(f"LLM response: {input_tokens} input tokens, {output_tokens} output tokens")
            logger.debug(f"Stop reason: {stop_reason}")

            return LLMResponse(
                content=content,
                model=self.model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                stop_reason=stop_reason,
            )

        except anthropic.APIError as e:
            logger.error(f"Anthropic API error: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error calling LLM: {e}")
            raise

    def send_message_with_tools(
        self,
        system_prompt: str,
        user_message: str,
        tools: List[Dict[str, Any]],
        tool_choice: Optional[Any] = None,
        timeout: int = 120,
    ) -> Dict[str, Any]:
        """
        Send message with tool use.

        Args:
            system_prompt: System prompt
            user_message: User message
            tools: List of tool definitions
            tool_choice: Tool choice constraint
            timeout: Request timeout

        Returns:
            Dictionary with response content and tool_use calls
        """
        logger.debug(f"Sending message with tools to {self.model}")

        messages = [{"role": "user", "content": user_message}]

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                system=system_prompt,
                messages=messages,
                tools=tools,
                tool_choice=tool_choice,
            )

            result = {
                "content": [],
                "tool_use_blocks": [],
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            }

            for block in response.content:
                if block.type == "text":
                    result["content"].append(block.text)
                elif block.type == "tool_use":
                    result["tool_use_blocks"].append({
                        "id": block.id,
                        "name": block.name,
                        "input": block.input,
                    })

            logger.info(f"LLM tool response: {result['input_tokens']} input, {result['output_tokens']} output")
            logger.debug(f"Tool use blocks: {len(result['tool_use_blocks'])}")

            return result

        except anthropic.APIError as e:
            logger.error(f"Anthropic API error: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error calling LLM: {e}")
            raise


def load_prompt(prompt_file: str) -> str:
    """
    Load prompt from file.

    Args:
        prompt_file: Path to prompt file

    Returns:
        Prompt content as string
    """
    prompt_path = (
        Path(__file__).parent / "prompts" / prompt_file
    )

    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")

    with open(prompt_path, 'r', encoding='utf-8') as f:
        return f.read()


# Convenience functions for each agent type
def call_pm_agent(
    llm_client: LLMClient,
    context: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Call PM Agent to analyze metrics and generate experiment spec.

    Args:
        llm_client: LLM client instance
        context: Context including current_metrics, target_metrics, memory_context

    Returns:
        Experiment spec dictionary
    """
    system_prompt = load_prompt("pm_system_prompt.txt")

    # Inject threshold snapshot
    system_prompt = llm_client.inject_thresholds(system_prompt)

    user_message = f"""# Task: Generate Experiment Spec

## Current Metrics
{context.get('current_metrics', {})}

## Target Metrics
{context.get('target_metrics', {})}

## Memory Context (Similar Past Experiments)
{context.get('memory_context', {})}

## Detector
{context.get('detector', 'Unknown')}

Please analyze the metrics and generate an experiment spec following the format in the system prompt.
"""

    response = llm_client.send_message(system_prompt, user_message)

    # Parse JSON response
    import json
    try:
        # Extract JSON from response
        content = response.content.strip()
        if "```json" in content:
            json_str = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            json_str = content.split("```")[1].split("```")[0].strip()
        else:
            json_str = content

        spec = json.loads(json_str)

        # Validate that spec uses correct thresholds from snapshot
        if llm_client.threshold_snapshot:
            _validate_spec_thresholds(spec, llm_client.threshold_snapshot, context.get('detector'))

        return spec
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse PM Agent response as JSON: {e}")
        logger.debug(f"Response content: {response.content[:500]}")
        raise ValueError("PM Agent response is not valid JSON") from e


def _validate_spec_thresholds(spec: Dict[str, Any], snapshot: Dict[str, Any], detector: Optional[str]):
    """
    Validate that experiment spec uses thresholds from snapshot.

    Args:
        spec: Experiment spec to validate
        snapshot: Threshold snapshot
        detector: Detector name

    Raises:
        ValueError: If spec uses incorrect threshold values
    """
    if not detector:
        return

    changes = spec.get("changes", [])
    detector_name = detector.replace("Detector", "") + "Detector"

    if detector_name not in snapshot.get("detectors", {}):
        logger.warning(f"Unknown detector in spec: {detector_name}")
        return

    snapshot_thresholds = snapshot["detectors"][detector_name]["thresholds"]

    for change in changes:
        param = change.get("parameter")
        to_value = change.get("to")

        if param in snapshot_thresholds:
            snapshot_value = snapshot_thresholds[param]
            # Allow some tolerance for floating point comparison
            if isinstance(snapshot_value, float) and isinstance(to_value, float):
                if abs(snapshot_value - to_value) < 0.01:
                    continue  # Values match
            elif snapshot_value == to_value:
                continue  # Values match exactly

            logger.warning(
                f"Spec threshold mismatch for {param}: "
                f"spec={to_value}, snapshot={snapshot_value}"
            )


def call_coder_agent(
    llm_client: LLMClient,
    spec: Dict[str, Any],
    file_contents: Dict[str, str],
) -> Dict[str, Any]:
    """
    Call Coder Agent to implement experiment spec.

    Args:
        llm_client: LLM client instance
        spec: Experiment spec
        file_contents: Dictionary of file_path -> content

    Returns:
        Implementation result with files_changed
    """
    system_prompt = load_prompt("coder_system_prompt.txt")

    # Inject threshold snapshot
    system_prompt = llm_client.inject_thresholds(system_prompt)

    user_message = f"""# Task: Implement Experiment Spec

## Experiment Spec
{json.dumps(spec, indent=2, ensure_ascii=False)}

## Available Files
{json.dumps(list(file_contents.keys()), indent=2)}

Please implement the spec following the guidelines in the system prompt.
"""

    # For code editing, we'll need tool use
    tools = [
        {
            "name": "read_file",
            "description": "Read the contents of a file",
            "input_schema": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the file to read",
                    },
                },
                "required": ["file_path"],
            },
        },
        {
            "name": "edit_file",
            "description": "Edit a file by replacing old_string with new_string",
            "input_schema": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the file to edit",
                    },
                    "old_string": {
                        "type": "string",
                        "description": "Exact string to replace",
                    },
                    "new_string": {
                        "type": "string",
                        "description": "Replacement string",
                    },
                },
                "required": ["file_path", "old_string", "new_string"],
            },
        },
    ]

    response = llm_client.send_message_with_tools(
        system_prompt,
        user_message,
        tools,
    )

    # Process tool use blocks
    files_changed = []
    for tool_block in response.get("tool_use_blocks", []):
        if tool_block["name"] == "edit_file":
            files_changed.append({
                "path": tool_block["input"]["file_path"],
                "change_type": "code_edit",
                "action": "edit",
            })

    return {
        "status": "success",
        "files_changed": files_changed,
        "raw_response": response,
    }


def call_reviewer_agent(
    llm_client: LLMClient,
    implementation: Dict[str, Any],
    spec: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Call Reviewer Agent to review implementation.

    Args:
        llm_client: LLM client instance
        implementation: Implementation result
        spec: Original experiment spec

    Returns:
        Review result with approval decision
    """
    system_prompt = load_prompt("reviewer_system_prompt.txt")

    # Inject threshold snapshot
    system_prompt = llm_client.inject_thresholds(system_prompt)

    user_message = f"""# Task: Review Implementation

## Experiment Spec
{json.dumps(spec, indent=2, ensure_ascii=False)}

## Implementation
{json.dumps(implementation, indent=2, ensure_ascii=False)}

Please review the implementation and provide your assessment following the format in the system prompt.
"""

    response = llm_client.send_message(system_prompt, user_message)

    try:
        content = response.content.strip()
        if "```json" in content:
            json_str = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            json_str = content.split("```")[1].split("```")[0].strip()
        else:
            json_str = content

        return json.loads(json_str)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse Reviewer Agent response as JSON: {e}")
        raise ValueError("Reviewer Agent response is not valid JSON") from e
