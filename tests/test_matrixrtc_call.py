"""Tests for the MatrixRTC / Element Call (LiveKit) signalling helpers.

The LiveKit media path needs the (glibc-only) livekit SDK and a real SFU, so it
is verified on the deployed container.  These tests cover the wire-protocol
logic that must be byte-correct for the bot to land in the right SFU room:
membership events, the lk-jwt token request, focus discovery, and PCM framing.
"""

import asyncio
import types

import numpy as np
import pytest

from pawlia.interfaces import matrixrtc_call as mr


# --- membership state events -------------------------------------------------

def test_membership_state_key_has_leading_underscore():
    assert mr.make_membership_state_key("@bot:hs.de", "DEV1") == "_@bot:hs.de_DEV1"


def test_membership_content_join_shape():
    c = mr.make_membership_content("DEV1", "https://rtc.hs.de", "!room:hs.de")
    assert c["application"] == "m.call"
    assert c["call_id"] == ""
    assert c["scope"] == "m.room"
    assert c["device_id"] == "DEV1"
    assert c["focus_active"] == {"type": "livekit", "focus_selection": "oldest_membership"}
    assert c["foci_preferred"] == [{
        "type": "livekit",
        "livekit_service_url": "https://rtc.hs.de",
        "livekit_alias": "!room:hs.de",
    }]


def test_empty_membership_is_leave():
    assert mr.empty_membership_content() == {}


def test_is_active_membership():
    assert mr.is_active_membership(mr.make_membership_content("D", "u", "!r")) is True
    assert mr.is_active_membership({}) is False
    assert mr.is_active_membership(None) is False


def test_focus_url_from_member_content():
    content = mr.make_membership_content("D", "https://rtc.hs.de", "!r:hs.de")
    assert mr.focus_url_from_member_content(content) == "https://rtc.hs.de"
    assert mr.focus_url_from_member_content({}) is None
    assert mr.focus_url_from_member_content({"foci_preferred": [{"type": "other"}]}) is None


# --- lk-jwt token request ----------------------------------------------------

def test_build_token_request_get_token_shape():
    openid = {
        "access_token": "TOK",
        "token_type": "Bearer",
        "matrix_server_name": "hs.de",
        "expires_in": 3600,
    }
    req = mr.build_token_request(openid, "!room:hs.de", "@bot:hs.de", "DEV1", slot_id="")
    assert req["room_id"] == "!room:hs.de"
    assert req["slot_id"] == ""
    assert req["openid_token"] == openid
    assert req["member"] == {
        "id": "@bot:hs.de",
        "claimed_user_id": "@bot:hs.de",
        "claimed_device_id": "DEV1",
    }


def test_build_token_request_defaults_token_type_and_expiry():
    req = mr.build_token_request(
        {"access_token": "TOK", "matrix_server_name": "hs.de"},
        "!r:hs.de", "@bot:hs.de", "DEV1")
    assert req["openid_token"]["token_type"] == "Bearer"
    assert req["openid_token"]["expires_in"] == 3600


# --- PCM framing -------------------------------------------------------------

def test_frame_to_float32_passthrough_48k_mono():
    src = np.array([0, 16384, -16384, 32767], dtype=np.int16)
    out = mr._frame_to_float32_mono_48k(src, 48000, 1)
    assert out.dtype == np.float32
    assert len(out) == 4
    assert abs(out[1] - 0.5) < 0.01
    assert -1.0 <= out.min() and out.max() <= 1.0


def test_frame_to_float32_downmixes_stereo():
    # interleaved L/R: two frames, both channels equal → mono identical
    src = np.array([100, 100, 200, 200], dtype=np.int16)
    out = mr._frame_to_float32_mono_48k(src, 48000, 2)
    assert len(out) == 2


def test_frame_to_float32_resamples_to_48k():
    src = np.zeros(240, dtype=np.int16)  # 240 @ 24k → 480 @ 48k
    out = mr._frame_to_float32_mono_48k(src, 24000, 1)
    assert len(out) == 480


# --- fetch_livekit_credentials (mocked httpx) --------------------------------

class _FakeResp:
    def __init__(self, status_code, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload or {}
        self.text = text

    def json(self):
        return self._payload


class _FakeHTTP:
    """Minimal async httpx.AsyncClient stand-in driven by a route table."""

    def __init__(self, routes):
        self._routes = routes  # path -> _FakeResp
        self.calls = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def post(self, url, json=None):
        path = "/" + url.split("/", 3)[-1] if "://" in url else url
        # store last path segment for assertions
        for p, resp in self._routes.items():
            if url.endswith(p):
                self.calls.append((p, json))
                return resp
        return _FakeResp(404, text="no route")


@pytest.mark.asyncio
async def test_fetch_credentials_uses_get_token(monkeypatch):
    fake = _FakeHTTP({"/get_token": _FakeResp(200, {"url": "wss://sfu", "jwt": "J"})})
    monkeypatch.setattr(mr, "_HTTPX_AVAILABLE", True)
    monkeypatch.setattr(mr, "httpx", types.SimpleNamespace(AsyncClient=lambda **kw: fake))

    creds = await mr.fetch_livekit_credentials(
        "https://rtc.hs.de", {"access_token": "T", "matrix_server_name": "hs.de"},
        "!room:hs.de", "@bot:hs.de", "DEV1")
    assert creds == {"url": "wss://sfu", "jwt": "J"}
    assert fake.calls[0][0] == "/get_token"


@pytest.mark.asyncio
async def test_fetch_credentials_falls_back_to_legacy(monkeypatch):
    fake = _FakeHTTP({
        "/get_token": _FakeResp(404),
        "/sfu/get": _FakeResp(200, {"url": "wss://sfu", "jwt": "J2"}),
    })
    monkeypatch.setattr(mr, "_HTTPX_AVAILABLE", True)
    monkeypatch.setattr(mr, "httpx", types.SimpleNamespace(AsyncClient=lambda **kw: fake))

    creds = await mr.fetch_livekit_credentials(
        "https://rtc.hs.de/", {"access_token": "T", "matrix_server_name": "hs.de"},
        "!room:hs.de", "@bot:hs.de", "DEV1")
    assert creds == {"url": "wss://sfu", "jwt": "J2"}
    # legacy body uses "room" + top-level device_id
    legacy_call = [c for c in fake.calls if c[0] == "/sfu/get"][0]
    assert legacy_call[1]["room"] == "!room:hs.de"
    assert legacy_call[1]["device_id"] == "DEV1"


@pytest.mark.asyncio
async def test_fetch_credentials_returns_none_on_all_failures(monkeypatch):
    fake = _FakeHTTP({"/get_token": _FakeResp(500, text="boom"), "/sfu/get": _FakeResp(403)})
    monkeypatch.setattr(mr, "_HTTPX_AVAILABLE", True)
    monkeypatch.setattr(mr, "httpx", types.SimpleNamespace(AsyncClient=lambda **kw: fake))
    creds = await mr.fetch_livekit_credentials(
        "https://rtc.hs.de", {"access_token": "T", "matrix_server_name": "hs.de"},
        "!r:hs.de", "@bot:hs.de", "DEV1")
    assert creds is None


# --- manager detection + membership posting ----------------------------------

def _make_manager(monkeypatch, enabled=True):
    monkeypatch.setattr(mr, "_LIVEKIT_AVAILABLE", True)
    client = types.SimpleNamespace(
        user_id="@bot:hs.de", device_id="BOTDEV", homeserver="https://hs.de")
    mgr = mr.MatrixRTCManager(
        client=client,
        app=types.SimpleNamespace(),
        cfg={"matrixrtc": {"enabled": enabled, "focus_url": "https://rtc.hs.de"}},
        send_text_cb=lambda *a, **k: None,
        send_thread_reply_cb=lambda *a, **k: None,
        get_agent_cb=lambda *a, **k: None,
    )
    return mgr, client


@pytest.mark.asyncio
async def test_manager_ignores_non_rtc_events(monkeypatch):
    mgr, _ = _make_manager(monkeypatch)
    calls = []
    monkeypatch.setattr(mgr, "_join", lambda *a, **k: calls.append(a))
    event = types.SimpleNamespace(type="m.reaction", sender="@a:hs.de", content={})
    await mgr.on_member_event(types.SimpleNamespace(room_id="!r:hs.de"), event)
    assert calls == []


@pytest.mark.asyncio
async def test_manager_ignores_own_membership(monkeypatch):
    mgr, _ = _make_manager(monkeypatch)
    joined = []

    async def _fake_join(*a, **k):
        joined.append(a)

    monkeypatch.setattr(mgr, "_join", _fake_join)
    event = types.SimpleNamespace(
        type=mr.RTC_MEMBER_EVENT_TYPE, sender="@bot:hs.de",
        content=mr.make_membership_content("D", "https://rtc.hs.de", "!r:hs.de"))
    await mgr.on_member_event(types.SimpleNamespace(room_id="!r:hs.de"), event)
    assert joined == []


@pytest.mark.asyncio
async def test_manager_joins_on_remote_membership(monkeypatch):
    mgr, _ = _make_manager(monkeypatch)
    joined = {}

    async def _fake_join(room, sender, focus_url, slot_id):
        joined.update(room_id=room.room_id, sender=sender, focus=focus_url, slot=slot_id)

    monkeypatch.setattr(mgr, "_join", _fake_join)
    event = types.SimpleNamespace(
        type=mr.RTC_MEMBER_EVENT_TYPE, sender="@alice:hs.de",
        content=mr.make_membership_content("ALICEDEV", "https://rtc.hs.de", "!r:hs.de"))
    await mgr.on_member_event(types.SimpleNamespace(room_id="!r:hs.de"), event)
    assert joined["sender"] == "@alice:hs.de"
    assert joined["focus"] == "https://rtc.hs.de"  # override wins


@pytest.mark.asyncio
async def test_manager_disabled_does_nothing(monkeypatch):
    mgr, _ = _make_manager(monkeypatch, enabled=False)
    joined = []
    monkeypatch.setattr(mgr, "_join", lambda *a, **k: joined.append(a))
    event = types.SimpleNamespace(
        type=mr.RTC_MEMBER_EVENT_TYPE, sender="@alice:hs.de",
        content=mr.make_membership_content("D", "https://rtc.hs.de", "!r:hs.de"))
    await mgr.on_member_event(types.SimpleNamespace(room_id="!r:hs.de"), event)
    assert joined == []


@pytest.mark.asyncio
async def test_manager_posts_and_redacts_membership(monkeypatch):
    mgr, client = _make_manager(monkeypatch)
    puts = []

    async def _put_state(room_id, etype, content, state_key=""):
        puts.append((room_id, etype, content, state_key))

    client.room_put_state = _put_state
    await mgr._post_membership("!r:hs.de", "https://rtc.hs.de", "")
    await mgr._redact_membership("!r:hs.de")

    assert puts[0][1] == mr.RTC_MEMBER_EVENT_TYPE
    assert puts[0][3] == "_@bot:hs.de_BOTDEV"
    assert mr.is_active_membership(puts[0][2]) is True
    assert puts[1][2] == {}  # redact = empty content
