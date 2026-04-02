"""Tests for tool-choice error handling and retry logic."""

import pytest

from pawlia.agents.base import BaseAgent
from pawlia.skills.executor import _is_tool_choice_error, _extract_tool_name


class TestToolChoiceErrorHandling:
    """Tests for tool_use_failed retry logic in BaseAgent._invoke."""

    def _make_error(self, tool_name: str = "files") -> Exception:
        """Create an exception that mimics the real API error format."""
        failed_gen = '{"name": "' + tool_name + '", "arguments": {"query": "test"}}'
        error_msg = (
            "Error code: 400 - {'error': {'message': 'Tool choice is none, "
            "but model called a tool', 'type': 'invalid_request_error', "
            "'code': 'tool_use_failed', "
            "'failed_generation': '" + failed_gen + "'}}"
        )
        return Exception(error_msg)

    def test_detects_tool_use_failed(self):
        error_str = str(self._make_error("files"))
        assert "tool_use_failed" in error_str

    def test_detects_tool_choice_none(self):
        error_str = str(self._make_error("bash"))
        assert "Tool choice is none" in error_str
        assert "called a tool" in error_str

    def test_extract_tool_name(self):
        exc = self._make_error("searxng")
        name = BaseAgent._extract_failed_tool_call(str(exc))
        assert name == "searxng"

    def test_extract_tool_name_no_match_returns_none(self):
        result = BaseAgent._extract_failed_tool_call("some other error")
        assert result is None


class TestWorkflowExecutorToolChoiceHelpers:
    """Tests for _is_tool_choice_error and _extract_tool_name helpers."""

    def test_is_tool_choice_error_with_tool_use_failed(self):
        assert _is_tool_choice_error(Exception("tool_use_failed in generation")) is True

    def test_is_tool_choice_error_with_tool_choice_none(self):
        exc = Exception("Tool choice is none, but model called a tool")
        assert _is_tool_choice_error(exc) is True

    def test_is_tool_choice_error_false_for_other_errors(self):
        assert _is_tool_choice_error(Exception("Connection refused")) is False

    def test_extract_tool_name(self):
        error_str = '{"name": "files", "arguments": {"query": "delete bootstrap.md"}}'
        assert _extract_tool_name(error_str) == "files"

    def test_extract_tool_name_empty(self):
        assert _extract_tool_name("no tool here") == ""