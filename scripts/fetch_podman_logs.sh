#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd "${script_dir}/.." && pwd)"
env_file="${project_root}/.env"

if [[ ! -f "$env_file" ]]; then
    echo "Fehlende ${env_file}. Lege sie z. B. aus .env.sample an." >&2
    exit 1
fi

set -a
# shellcheck disable=SC1090
source "$env_file"
set +a

ssh_host="${PODMAN_LOGS_SSH_HOST:-}"
container="${PODMAN_LOGS_CONTAINER:-}"
output_dir="${PODMAN_LOGS_OUTPUT_DIR:-./downloads/podman-logs}"
since="${PODMAN_LOGS_SINCE:-}"

if [[ -z "$ssh_host" || -z "$container" ]]; then
    echo "Bitte PODMAN_LOGS_SSH_HOST und PODMAN_LOGS_CONTAINER in .env setzen." >&2
    exit 1
fi

since_args=()
if [[ -n "$since" ]]; then
    since_args=(--since "$since")
fi

timestamp="$(date +%Y%m%d-%H%M%S)"
safe_host="${ssh_host//@/_}"
safe_host="${safe_host//:/_}"
safe_container="${container//\//_}"

mkdir -p "$output_dir"

target_file="${output_dir}/${safe_host}-${safe_container}-${timestamp}.log"

echo "Lade Logs von ${container} auf ${ssh_host} ..."
remote_command=(
    podman logs
    "${since_args[@]}"
    "$container"
)
ssh "$ssh_host" "${remote_command[@]}" >"$target_file" 2>&1

echo "Gespeichert in: $target_file"
