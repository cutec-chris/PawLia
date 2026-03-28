"""Tool base class and registry."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class Tool(ABC):
    """Base class for all tools available to SkillRunnerAgents."""

    name: str = ""
    description: str = ""

    @abstractmethod
    def parameters(self) -> Dict[str, Any]:
        """Return JSON Schema properties for this tool's parameters."""

    def required_parameters(self) -> List[str]:
        """Return required parameter names for the default object schema."""
        return []

    def input_schema(self) -> Dict[str, Any]:
        """Return the full JSON schema for tool arguments."""
        return {
            "type": "object",
            "properties": self.parameters(),
            "required": self.required_parameters(),
            "additionalProperties": False,
        }

    def normalize_args(self, args: Any) -> Dict[str, Any]:
        """Best-effort normalization for malformed model tool arguments."""
        if args is None:
            return {}
        if isinstance(args, dict):
            return dict(args)
        if isinstance(args, str):
            properties = list(self.input_schema().get("properties", {}))
            if len(properties) == 1:
                return {properties[0]: args}
        return {}

    def validate_args(self, args: Any) -> Optional[str]:
        """Return an error string when args don't match the declared schema."""
        return _validate_schema(self.normalize_args(args), self.input_schema())

    @abstractmethod
    def execute(self, args: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Any:
        """Execute the tool with the given arguments."""

    def as_openai_spec(self) -> Dict[str, Any]:
        """Generate OpenAI function-calling tool specification."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.input_schema(),
            },
        }


def _validate_schema(value: Any, schema: Dict[str, Any]) -> Optional[str]:
    """Validate a small JSON Schema subset used by PawLia tools."""
    if schema.get("type") == "object":
        if not isinstance(value, dict):
            return "Arguments must be a JSON object."
        if schema.get("additionalProperties") is False:
            properties = schema.get("properties", {})
            extras = [key for key in value if key not in properties]
            if extras:
                return f"Unexpected argument(s): {', '.join(sorted(extras))}."
        one_of = schema.get("oneOf")
        if one_of:
            for branch in one_of:
                merged = {
                    **schema,
                    **branch,
                    "properties": {
                        **schema.get("properties", {}),
                        **branch.get("properties", {}),
                    },
                }
                merged.pop("oneOf", None)
                if _validate_schema(value, merged) is None:
                    break
            else:
                return "Arguments do not match any allowed parameter shape."
        required = schema.get("required", [])
        missing = [key for key in required if key not in value or _is_empty(value[key])]
        if missing:
            return f"Missing required argument(s): {', '.join(missing)}."
        for key, prop_schema in schema.get("properties", {}).items():
            if key not in value:
                continue
            error = _validate_value(value[key], prop_schema, key)
            if error:
                return error
        return None

    return _validate_value(value, schema, "value")


def _validate_value(value: Any, schema: Dict[str, Any], field_name: str) -> Optional[str]:
    expected_type = schema.get("type")
    if expected_type == "string":
        if not isinstance(value, str):
            return f"Argument '{field_name}' must be a string."
        if schema.get("minLength") and len(value.strip()) < schema["minLength"]:
            return f"Argument '{field_name}' must not be empty."
    elif expected_type == "integer":
        if not isinstance(value, int) or isinstance(value, bool):
            return f"Argument '{field_name}' must be an integer."
    elif expected_type == "number":
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            return f"Argument '{field_name}' must be a number."
    elif expected_type == "boolean":
        if not isinstance(value, bool):
            return f"Argument '{field_name}' must be a boolean."

    if "enum" in schema and value not in schema["enum"]:
        allowed = ", ".join(map(str, schema["enum"]))
        return f"Argument '{field_name}' must be one of: {allowed}."

    return None


def _is_empty(value: Any) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())


class ToolRegistry:
    """Manages tool registration, spec generation, and execution."""

    def __init__(self):
        self.tools: Dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self.tools[tool.name] = tool

    def get_specs(self) -> List[Dict[str, Any]]:
        return [t.as_openai_spec() for t in self.tools.values()]

    def execute(self, name: str, args: Any,
                context: Optional[Dict[str, Any]] = None) -> Any:
        original_name = name
        name = self._resolve(name)
        tool = self.tools.get(name)
        if not tool:
            return f"Error: Tool '{original_name}' not found."
        normalized_args = tool.normalize_args(args)
        error = tool.validate_args(normalized_args)
        if error:
            return f"Error: Invalid arguments for tool '{name}': {error}"
        try:
            return tool.execute(normalized_args, context)
        except Exception as e:
            return f"Error: {e}"

    def _resolve(self, name: str) -> str:
        """Fuzzy-match tool name (ignore dashes/underscores/case)."""
        norm = name.replace("_", "").replace("-", "").lower()
        for registered in self.tools:
            if registered.replace("_", "").replace("-", "").lower() == norm:
                return registered
        return name

    def names(self) -> List[str]:
        return list(self.tools.keys())
