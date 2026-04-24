#!/usr/bin/env sh

set -eu

script_dir="$(cd "$(dirname "$0")" && pwd)"
project_root="$(cd "${script_dir}/.." && pwd)"
env_file="${project_root}/.env"

if [ ! -f "$env_file" ]; then
    echo "Fehlende ${env_file}. Lege sie z. B. aus .env.sample an." >&2
    exit 1
fi

set -a
# shellcheck disable=SC1090
. "$env_file"
set +a

ssh_host="${PODMAN_LOGS_SSH_HOST:-}"
container="${PODMAN_LOGS_CONTAINER:-}"
output_dir="${PODMAN_LOGS_OUTPUT_DIR:-./downloads/podman-logs}"
since="${PODMAN_LOGS_SINCE:-}"
debug_audio_dir="${PODMAN_LOGS_DEBUG_AUDIO_DIR:-/app/log/debug_audio}"
fetch_debug_audio="${PODMAN_LOGS_FETCH_DEBUG_AUDIO:-1}"

if [ -z "$ssh_host" ] || [ -z "$container" ]; then
    echo "Bitte PODMAN_LOGS_SSH_HOST und PODMAN_LOGS_CONTAINER in .env setzen." >&2
    exit 1
fi

since_args=""
if [ -n "$since" ]; then
    since_args="--since $since"
fi

timestamp="$(date +%Y%m%d-%H%M%S)"
safe_host="$(printf '%s' "$ssh_host" | tr '@:' '__')"
safe_container="$(printf '%s' "$container" | tr '/' '_')"

mkdir -p "$output_dir"

fetch_dir="${output_dir}/${safe_host}-${safe_container}-${timestamp}"
target_file="${fetch_dir}/container.log"
audio_target_dir="${fetch_dir}/debug_audio"

mkdir -p "$fetch_dir"

echo "Lade Logs von ${container} auf ${ssh_host} ..."
if [ -n "$since_args" ]; then
    ssh "$ssh_host" podman logs --since "$since" "$container" >"$target_file" 2>&1
else
    ssh "$ssh_host" podman logs "$container" >"$target_file" 2>&1
fi

echo "Logs gespeichert in: $target_file"

if [ "$fetch_debug_audio" = "1" ]; then
    mkdir -p "$audio_target_dir"
    echo "Lade Debug-Audio aus ${debug_audio_dir} von ${container} auf ${ssh_host} ..."
    remote_audio_command="podman exec '$container' sh -lc 'if [ -d \"$debug_audio_dir\" ]; then tar -C \"$debug_audio_dir\" -cf - .; fi'"
    if ssh "$ssh_host" "$remote_audio_command" \
        | tar -C "$audio_target_dir" -xf - 2>/dev/null
    then
        if find "$audio_target_dir" -type f | grep -q .; then
            echo "Debug-Audio gespeichert in: $audio_target_dir"
        else
            rmdir "$audio_target_dir"
            echo "Kein Debug-Audio gefunden."
        fi
    else
        rmdir "$audio_target_dir" 2>/dev/null || true
        echo "Debug-Audio konnte nicht geladen werden." >&2
    fi
fi
