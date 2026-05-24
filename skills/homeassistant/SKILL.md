---
name: homeassistant
description: >-
  Interact with Home Assistant via its REST API. Use when the user wants to
  query device states, control smart home devices (lights, switches, climate,
  covers, media players, scenes, automations), check sensor values (temperature,
  humidity, energy, etc.), or run Home Assistant services. Triggers on phrases
  like "turn on the lights", "what's the temperature", "is the door locked",
  "show me all sensors", "run automation", "set thermostat", "close the blinds".
license: MIT
metadata:
  author: Christian Ulrich
  version: "1.0"
  trust: internal
  requires_config:
    - url
requires_credentials:
  - ha_token
---

# Home Assistant

## Instructions

1. Parse the user's intent: querying states, calling a service, or listing entities.
2. Run the appropriate script command:
   - **List entity states** (or filter by domain/entity):
     ```
     python <scripts_dir>/ha.py states [--entity <entity_id>] [--domain <domain>]
     ```
   - **Call a service** (turn on/off, set temperature, run automation, etc.):
     ```
     python <scripts_dir>/ha.py call --domain <domain> --service <service> [--entity <entity_id>] [--data '<json>']
     ```
   - **Get history** for an entity:
     ```
     python <scripts_dir>/ha.py history --entity <entity_id> [--hours <n>]
     ```
   The script reads the Home Assistant URL from `skill-config.homeassistant.url` and the long-lived token from `CRED_HA_TOKEN`. Do not pass URL/token as CLI args unless the user explicitly overrides them.
3. Parse the JSON output (`success`, plus result fields or `error`).
4. Format the output clearly for the user using natural language.

## Output format

For state queries, return a clean list:
```
🔌 **Living Room Light**: on (brightness: 80%)
🌡️  **Living Room Temperature**: 22.4°C
🔒 **Front Door**: locked
```

For service calls, confirm the action:
```
Turned on Living Room Light.
Set thermostat to 21°C.
```

The script returns JSON:
```json
{"success": true, "result": [...]}
```

On error:
```json
{"success": false, "error": "error message"}
```

## Error handling

| Error | Recovery action |
|-------|-----------------|
| Connection refused | Check that Home Assistant is reachable at the configured URL |
| Unauthorized (401) | The HA long-lived token is invalid or expired — regenerate it |
| Entity not found | Suggest checking the entity ID with `states --domain <domain>` |
| Service not found | Check available services with `ha.py services --domain <domain>` |
