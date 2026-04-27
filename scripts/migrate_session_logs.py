#!/usr/bin/env python3

import argparse
import os

from pawlia.memory import MemoryManager, SESSION_FORMAT_VERSION


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Migrate PawLia session logs into daily files with embedded thread sections.",
    )
    parser.add_argument("session_dir", help="Root session directory")
    parser.add_argument("--user", help="Only migrate one user/session")
    args = parser.parse_args()

    mm = MemoryManager(args.session_dir)
    user_ids = [args.user] if args.user else sorted(
        name for name in os.listdir(args.session_dir)
        if os.path.isdir(os.path.join(args.session_dir, name))
    )

    migrated_sessions = 0
    migrated_files = 0
    for user_id in user_ids:
        count = mm.migrate_session(user_id)
        if count:
            migrated_sessions += 1
            migrated_files += count
            print(f"{user_id}: migrated {count} legacy thread log(s)")
        else:
            version = mm._read_session_version(user_id)
            print(f"{user_id}: up to date (version {version or SESSION_FORMAT_VERSION})")

    print(
        f"done: {migrated_sessions} session(s) changed, {migrated_files} legacy thread log(s) migrated"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
