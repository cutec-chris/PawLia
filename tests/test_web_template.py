from pathlib import Path


def test_web_chat_persists_thread_id_between_messages():
    template = Path("pawlia/interfaces/templates/index.html").read_text(encoding="utf-8")

    assert "let activeThreadId = null;" in template
    assert "if (activeThreadId) requestBody.thread_id = activeThreadId;" in template
    assert "if (data.thread_id) setActiveThread(data.thread_id);" in template