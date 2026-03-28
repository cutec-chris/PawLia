"""ScheduleReminder tool - schedules proactive reminders."""

import json
import logging
import os
import re
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

from pawlia.tools.base import Tool


class ReminderTool(Tool):
    name = "schedule_reminder"
    description = (
        "Schedule a proactive reminder or wake-up call. Supports ISO8601 "
        "datetimes and relative times like '30m', '2h', '1d'."
    )

    def parameters(self) -> Dict[str, Any]:
        return {
            "action": {
                "type": "string",
                "enum": ["add", "list", "delete"],
                "description": "Action: 'add', 'list', or 'delete'.",
            },
            "fire_at": {
                "type": "string",
                "description": "When to fire. ISO8601 or relative ('30m', '2h'). Required for add.",
            },
            "message": {
                "type": "string",
                "description": "Message to deliver when firing. Required for add.",
            },
            "label": {
                "type": "string",
                "description": "Short label for the reminder.",
            },
            "recurrence": {
                "type": "string",
                "enum": ["none", "daily", "weekly", "monthly"],
                "description": "How often the reminder repeats.",
            },
            "reminder_id": {
                "type": "string",
                "description": "Reminder ID for delete action.",
            },
        }

    def input_schema(self) -> Dict[str, Any]:
        properties = self.parameters()
        return {
            "type": "object",
            "properties": properties,
            "additionalProperties": False,
            "oneOf": [
                {
                    "required": ["action"],
                    "properties": {
                        "action": {**properties["action"], "enum": ["list"]},
                    },
                },
                {
                    "required": ["action", "reminder_id"],
                    "properties": {
                        "action": {**properties["action"], "enum": ["delete"]},
                    },
                },
                {
                    "required": ["fire_at", "message"],
                    "properties": {
                        "action": {**properties["action"], "enum": ["add"]},
                    },
                },
            ],
        }

    def normalize_args(self, args: Any) -> Dict[str, Any]:
        normalized = super().normalize_args(args)
        alias_map = {
            "id": "reminder_id",
            "when": "fire_at",
            "text": "message",
        }
        for src, dst in alias_map.items():
            if dst not in normalized and src in normalized:
                normalized[dst] = normalized[src]
        return normalized

    def execute(self, args: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Any:
        action = args.get("action", "add")
        ctx = context or {}
        user_id = ctx.get("user_id", "")
        session_dir = ctx.get("session_dir", "")

        if not user_id or not session_dir:
            return {"success": False, "error": "No session context available."}

        path = self._reminders_path(user_id, session_dir)
        reminders = self._load(path)

        if action == "list":
            pending = [r for r in reminders if not r.get("fired")]
            return {"success": True, "reminders": pending, "total": len(pending)}

        if action == "delete":
            rid = args.get("reminder_id", "").strip()
            if not rid:
                return {"success": False, "error": "reminder_id required for delete."}
            before = len(reminders)
            reminders = [r for r in reminders if r.get("id") != rid]
            if len(reminders) == before:
                return {"success": False, "error": "Reminder not found."}
            self._save(path, reminders)
            return {"success": True, "message": "Reminder deleted."}

        # action == "add"
        fire_at_str = args.get("fire_at", "").strip()
        message = args.get("message", "").strip()
        label = args.get("label", "Reminder").strip()

        if not fire_at_str:
            return {"success": False, "error": "fire_at is required."}
        if not message:
            return {"success": False, "error": "message is required."}

        try:
            fire_at = self._parse_fire_at(fire_at_str)
        except Exception as e:
            return {"success": False, "error": f"Invalid fire_at format: {e}"}

        recurrence = args.get("recurrence", "none").strip().lower()
        if recurrence not in ("none", "daily", "weekly", "monthly"):
            recurrence = "none"

        reminder = {
            "id": str(uuid.uuid4()),
            "user_id": user_id,
            "fire_at": fire_at.isoformat(),
            "message": message,
            "label": label,
            "recurrence": recurrence,
            "fired": False,
            "created_at": datetime.now().isoformat(),
        }
        reminders.append(reminder)
        self._save(path, reminders)

        return {
            "success": True,
            "message": f"Reminder '{label}' scheduled for {fire_at.strftime('%d.%m.%Y %H:%M')}",
            "reminder_id": reminder["id"],
        }

    @staticmethod
    def _parse_fire_at(fire_at: str) -> datetime:
        fire_at = fire_at.strip()
        m = re.match(r"^(\d+)\s*(m|min|h|d)$", fire_at.lower())
        if m:
            amount = int(m.group(1))
            unit = m.group(2)
            now = datetime.now()
            if unit in ("m", "min"):
                return now + timedelta(minutes=amount)
            elif unit == "h":
                return now + timedelta(hours=amount)
            elif unit == "d":
                return now + timedelta(days=amount)
        return datetime.fromisoformat(fire_at)

    @staticmethod
    def _reminders_path(user_id: str, session_dir: str) -> str:
        user_dir = os.path.join(session_dir, user_id)
        os.makedirs(user_dir, exist_ok=True)
        return os.path.join(user_dir, "reminders.json")

    @staticmethod
    def _load(path: str) -> list:
        if not os.path.exists(path):
            return []
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logging.getLogger("pawlia.tools.reminder").error("Failed to load %s: %s", path, e)
            return []

    @staticmethod
    def _save(path: str, data: list) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
