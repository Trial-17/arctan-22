from typing import Any, Dict

# This dictionary will be shared between the API and the agent
# to track JS tool calls and their results.
JS_TOOL_CALLS: Dict[str, Dict[str, Any]] = {} 