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
                "parameters": {
                    "type": "object",
                    "properties": self.parameters(),
                },
            },
        }


class ToolRegistry:
    """Manages tool registration, spec generation, and execution."""

    def __init__(self):
        self.tools: Dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self.tools[tool.name] = tool

    def get_specs(self) -> List[Dict[str, Any]]:
        return [t.as_openai_spec() for t in self.tools.values()]

    def execute(self, name: str, args: Dict[str, Any],
                context: Optional[Dict[str, Any]] = None) -> Any:
        name = self._resolve(name)
        tool = self.tools.get(name)
        if not tool:
            return f"Error: Tool '{name}' not found."
        try:
            return tool.execute(args, context)
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
