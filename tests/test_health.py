from types import SimpleNamespace

from pawlia.interfaces.web import build_health_payload


class DummyTask:
    def __init__(self, done=False):
        self._done = done

    def done(self):
        return self._done


def test_health_reports_configured_matrix_interface_ok():
    app = SimpleNamespace(
        scheduler=SimpleNamespace(_task=DummyTask()),
        config={"interfaces": {"matrix": {}, "web": {}}},
        interface_health={"matrix": "running", "web": "running"},
    )

    status, payload = build_health_payload(app)

    assert status == 200
    assert payload["status"] == "ok"
    assert payload["checks"]["scheduler"] == "ok"
    assert payload["checks"]["interface:matrix"] == "running"


def test_health_fails_when_matrix_supervisor_crashed():
    app = SimpleNamespace(
        scheduler=SimpleNamespace(_task=DummyTask()),
        config={"interfaces": {"matrix": {}, "web": {}}},
        interface_health={"matrix": "crashed", "web": "running"},
    )

    status, payload = build_health_payload(app)

    assert status == 503
    assert payload["status"] == "unhealthy"
    assert payload["checks"]["interface:matrix"] == "crashed"


def test_health_fails_when_configured_interface_has_no_status():
    app = SimpleNamespace(
        scheduler=SimpleNamespace(_task=DummyTask()),
        config={"interfaces": {"matrix": {}, "web": {}}},
        interface_health={},
    )

    status, payload = build_health_payload(app)

    assert status == 503
    assert payload["checks"]["interface:matrix"] == "unknown"


def test_health_fails_when_scheduler_task_stopped():
    app = SimpleNamespace(
        scheduler=SimpleNamespace(_task=DummyTask(done=True)),
        config={"interfaces": {"web": {}}},
        interface_health={"web": "running"},
    )

    status, payload = build_health_payload(app)

    assert status == 503
    assert payload["checks"]["scheduler"] == "stopped"
