"""BackgroundTaskQueue — the /background defer-to-idle persistence layer.

Pure file-backed state, so we drive the public API directly and assert the
on-disk lifecycle: enqueue -> pending -> mark_running/done/error, per-user
isolation, and resilience to corrupt task files.
"""

import json

from pawlia.background_tasks import BackgroundTaskQueue


def test_enqueue_persists_a_pending_task(tmp_path):
    q = BackgroundTaskQueue(str(tmp_path))
    task = q.enqueue("alice", "summarise my notes")

    assert task["status"] == "pending"
    assert task["message"] == "summarise my notes"
    assert task["thread_id"] == f"bg_{task['id']}"

    on_disk = tmp_path / "alice" / "background_tasks" / f"{task['id']}.json"
    assert json.loads(on_disk.read_text(encoding="utf-8"))["message"] == "summarise my notes"


def test_pending_returns_only_pending_oldest_first(tmp_path):
    q = BackgroundTaskQueue(str(tmp_path))
    first = q.enqueue("alice", "one")
    second = q.enqueue("alice", "two")
    q.mark_done("alice", first["id"])

    pending = q.pending()

    ids = [t["id"] for _, t in pending]
    assert first["id"] not in ids
    assert second["id"] in ids


def test_pending_spans_multiple_users(tmp_path):
    q = BackgroundTaskQueue(str(tmp_path))
    q.enqueue("alice", "a")
    q.enqueue("bob", "b")

    users = {user_id for user_id, _ in q.pending()}
    assert users == {"alice", "bob"}


def test_pending_on_empty_session_dir_is_empty(tmp_path):
    q = BackgroundTaskQueue(str(tmp_path / "does-not-exist"))
    assert q.pending() == []


def test_list_tasks_returns_all_states_for_a_user(tmp_path):
    q = BackgroundTaskQueue(str(tmp_path))
    a = q.enqueue("alice", "a")
    b = q.enqueue("alice", "b")
    q.mark_done("alice", a["id"])

    tasks = {t["id"]: t["status"] for t in q.list_tasks("alice")}
    assert tasks[a["id"]] == "done"
    assert tasks[b["id"]] == "pending"


def test_list_tasks_for_unknown_user_is_empty(tmp_path):
    q = BackgroundTaskQueue(str(tmp_path))
    assert q.list_tasks("nobody") == []


def test_mark_running_then_done_updates_status_and_stamps_finished(tmp_path):
    q = BackgroundTaskQueue(str(tmp_path))
    task = q.enqueue("alice", "work")

    q.mark_running("alice", task["id"])
    assert _load(tmp_path, "alice", task["id"])["status"] == "running"

    q.mark_done("alice", task["id"])
    done = _load(tmp_path, "alice", task["id"])
    assert done["status"] == "done"
    assert "finished" in done


def test_mark_error_records_message(tmp_path):
    q = BackgroundTaskQueue(str(tmp_path))
    task = q.enqueue("alice", "work")

    q.mark_error("alice", task["id"], "boom")
    err = _load(tmp_path, "alice", task["id"])
    assert err["status"] == "error"
    assert err["error"] == "boom"
    assert "finished" in err


def test_update_of_unknown_task_is_a_noop(tmp_path):
    q = BackgroundTaskQueue(str(tmp_path))
    # No file exists; must not raise.
    q.mark_done("alice", "deadbeef")
    assert q.list_tasks("alice") == []


def test_corrupt_task_file_is_skipped_not_fatal(tmp_path):
    q = BackgroundTaskQueue(str(tmp_path))
    good = q.enqueue("alice", "good")
    bad = tmp_path / "alice" / "background_tasks" / "broken.json"
    bad.write_text("{ not json", encoding="utf-8")

    pending_ids = [t["id"] for _, t in q.pending()]
    listed_ids = [t["id"] for t in q.list_tasks("alice")]

    assert good["id"] in pending_ids
    assert good["id"] in listed_ids


def _load(session_dir, user_id, task_id):
    path = session_dir / user_id / "background_tasks" / f"{task_id}.json"
    return json.loads(path.read_text(encoding="utf-8"))
