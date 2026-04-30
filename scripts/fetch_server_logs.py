#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path


DEFAULT_ENV_FILE = ".env"


def load_dotenv(env_path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not env_path.exists():
        raise FileNotFoundError(f".env file not found: {env_path}")

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue

        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        values[key] = value
    return values


def build_ssh_command(config: dict[str, str], remote_command: str) -> list[str]:
    target = config.get("SERVER_SSH_TARGET") or config.get("SSH_TARGET")
    if not target:
        raise ValueError("Missing SERVER_SSH_TARGET in .env")

    command = ["ssh"]
    port = config.get("SERVER_SSH_PORT") or config.get("SSH_PORT")
    if port:
        command.extend(["-p", port])

    extra_options = shlex.split(config.get("SERVER_SSH_OPTS", ""))
    command.extend(extra_options)
    command.extend([target, remote_command])
    return command


def build_remote_command(config: dict[str, str], service: str | None, since: str | None, tail: int | None) -> str:
    compose_path = config.get("SERVER_COMPOSE_PATH") or config.get("COMPOSE_PATH")
    if not compose_path:
        raise ValueError("Missing SERVER_COMPOSE_PATH in .env")

    parts = [
        "cd",
        shlex.quote(compose_path),
        "&&",
        "podman-compose",
        "logs",
        "--no-color",
    ]

    if since:
        parts.extend(["--since", shlex.quote(since)])
    if tail is not None:
        parts.extend(["--tail", str(tail)])
    if service:
        parts.append(shlex.quote(service))

    return " ".join(parts)


def default_output_path(log_dir: Path, service: str | None) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    name = f"server-logs-{service or 'all'}-{stamp}.log"
    return log_dir / name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download podman-compose logs from a server via SSH."
    )
    parser.add_argument(
        "--env-file",
        default=DEFAULT_ENV_FILE,
        help="Path to the .env file with SSH and compose settings.",
    )
    parser.add_argument(
        "--service",
        help="Optional compose service name to fetch logs for.",
    )
    parser.add_argument(
        "--since",
        help="Optional podman-compose --since value, e.g. 2h or 2026-04-22T10:00:00.",
    )
    parser.add_argument(
        "--tail",
        type=int,
        help="Optional number of trailing log lines to fetch.",
    )
    parser.add_argument(
        "--output",
        help="Optional output file path. Defaults to log/server-logs-<timestamp>.log",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = Path(__file__).resolve().parent.parent
    env_path = Path(args.env_file)
    if not env_path.is_absolute():
        env_path = project_root / env_path

    config = load_dotenv(env_path)
    remote_command = build_remote_command(config, args.service, args.since, args.tail)
    ssh_command = build_ssh_command(config, remote_command)

    output_path = Path(args.output) if args.output else default_output_path(project_root / "log", args.service)
    if not output_path.is_absolute():
        output_path = project_root / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Fetching logs via SSH from {config.get('SERVER_SSH_TARGET') or config.get('SSH_TARGET')}...")
    print(f"Remote command: {remote_command}")
    print(f"Writing output to {output_path}")

    completed = subprocess.run(
        ssh_command,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    output_path.write_text(completed.stdout, encoding="utf-8")

    if completed.returncode != 0:
        stderr = completed.stderr.strip()
        if stderr:
            print(stderr, file=sys.stderr)
        print(f"SSH command failed with exit code {completed.returncode}", file=sys.stderr)
        return completed.returncode

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
