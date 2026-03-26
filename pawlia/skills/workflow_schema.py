"""Pydantic models for compiled skill workflows (workflow.yaml)."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


class VerifySpec(BaseModel):
    """Programmatic verification rules for a building block."""

    exit_code: int = 0
    output_contains: list[str] = []
    output_not_contains: list[str] = []
    output_regex: Optional[str] = None


class BuildingBlock(BaseModel):
    """An available action the LLM can use during execution."""

    id: str
    command: str            # bash template with {param} placeholders
    description: str        # helps LLM pick the right block
    status_desc: str = ""   # short status template with {param} placeholders, e.g. "Öffne {url}"
    verify: Optional[VerifySpec] = None
    on_error: Optional[str] = None  # block id to run on failure


class GoalCheck(BaseModel):
    """End-of-execution goal verification."""

    prompt: str
    max_retries: int = 2


class Workflow(BaseModel):
    """A single workflow (one procedure a skill can perform)."""

    id: str
    trigger: str
    building_blocks: list[BuildingBlock]
    max_steps: int = 15
    goal_check: Optional[GoalCheck] = None


class CompiledWorkflow(BaseModel):
    """Top-level model for workflow.yaml."""

    skill: str
    version: str
    compiled_at: str
    compiled_by: str
    workflows: list[Workflow]
