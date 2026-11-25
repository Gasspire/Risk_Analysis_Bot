import json
import os
import time
from models import SessionStateData, BotStateEnum
from storage.session_repository import SessionRepository
from storage.chat_repository import ChatRepository


def test_session_repository_save_load_and_missing(tmp_path, sample_threat_factory):
    filepath = tmp_path / "session.json"
    repo = SessionRepository(filepath=str(filepath))

    default_state = repo.load(default_context="Custom Test Scenario")
    assert default_state.general_context == "Custom Test Scenario"
    assert default_state.bot_state == BotStateEnum.IDLE

    state = SessionStateData(
        bot_state=BotStateEnum.DISCUSSING_THREAT,
        general_context="Scenario B",
        assessment_results=[
            sample_threat_factory(
                threat_id="1",
                threat="Threat One",
                affected_assets="Asset One",
                security_measures="Control One",
                likelihood=3,
                impact=5,
                risk_factor=15,
            )
        ],
    )
    repo.save(state)
    loaded = repo.load()
    assert loaded.bot_state == BotStateEnum.DISCUSSING_THREAT
    assert loaded.general_context == "Scenario B"
    assert len(loaded.assessment_results) == 1
    assert loaded.assessment_results[0].risk_factor == 15


def test_session_repository_corrupted_json(tmp_path):
    filepath = tmp_path / "bad_session.json"
    filepath.write_text("{not valid json", encoding="utf-8")
    repo = SessionRepository(filepath=str(filepath))

    loaded = repo.load(default_context="Fallback Scenario")
    assert loaded.general_context == "Fallback Scenario"
    assert loaded.assessment_results == []


def test_chat_repository_crud_and_sorting(tmp_path):
    chats_dir = tmp_path / "chats"
    repo = ChatRepository(chats_dir=str(chats_dir))

    assert repo.list_chats() == []
    assert repo.load_chat("nonexistent") == []
    assert repo.delete_chat("nonexistent") is False

    c1 = repo.create_chat("Scenario: Alpha?")
    assert c1 == "Scenario Alpha"
    c2 = repo.create_chat("Scenario Alpha")
    assert c2 == "Scenario Alpha_1"
    c3 = repo.create_chat()
    assert c3.startswith("Chat_")

    repo.save_chat(c1, [{"role": "user", "content": "hi"}])
    time.sleep(0.02)
    repo.save_chat(c2, [{"role": "assistant", "content": "hello"}])

    listed = repo.list_chats()
    assert listed[0] == c2
    assert c1 in listed
    assert repo.load_chat(c1) == [{"role": "user", "content": "hi"}]

    dict_path = os.path.join(str(chats_dir), "dict_chat.json")
    with open(dict_path, "w", encoding="utf-8") as f:
        json.dump({"messages": [{"role": "user", "content": "from dict"}]}, f)
    assert repo.load_chat("dict_chat") == [{"role": "user", "content": "from dict"}]

    bad_path = os.path.join(str(chats_dir), "broken.json")
    with open(bad_path, "w", encoding="utf-8") as f:
        f.write("{broken")
    assert repo.load_chat("broken") == []

    ok, msg = repo.rename_chat(c3, "///")
    assert ok is True
    assert msg.startswith("chat_")

    ok, msg = repo.rename_chat("missing_chat", "NewName")
    assert ok is False

    ok, msg = repo.rename_chat(c1, c1)
    assert ok is True
    assert msg == c1

    ok, msg = repo.rename_chat(c1, c2)
    assert ok is False

    ok, msg = repo.rename_chat(c1, "Renamed_Chat")
    assert ok is True
    assert msg == "Renamed_Chat"
    assert repo.load_chat("Renamed_Chat") == [{"role": "user", "content": "hi"}]

    assert repo.delete_chat("Renamed_Chat") is True
    assert "Renamed_Chat" not in repo.list_chats()
