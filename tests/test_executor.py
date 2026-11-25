from unittest.mock import MagicMock, patch
from models import SessionStateData, BotStateEnum, IntentOutput, IntentType
from llm_handler.executor import (
    is_cancel_command,
    cancel_active_flow,
    RiskAgentExecutor,
)


def test_is_cancel_command():
    assert is_cancel_command("annulla") is True
    assert is_cancel_command(" /Cancel! ") is True
    assert is_cancel_command("never mind.") is True
    assert is_cancel_command("analyze threat alpha") is False


def test_cancel_active_flow(sample_threat_factory):
    backup = sample_threat_factory(
        threat_id="t1",
        threat="Threat Alpha",
        affected_assets="Service A",
        security_measures="Control A",
        likelihood=2,
        impact=3,
        risk_factor=6,
    )
    modified = backup.model_copy(update={"likelihood": 5, "risk_factor": 15})
    state = SessionStateData(
        bot_state=BotStateEnum.DISCUSSING_THREAT,
        pending_threat={"threat": "Threat Beta"},
        active_threat_id="t1",
        active_threat_backup=backup,
        assessment_results=[modified],
    )
    msg, updated = cancel_active_flow(state)
    assert updated.bot_state == BotStateEnum.IDLE
    assert updated.pending_threat is None
    assert updated.active_threat_id is None
    assert updated.active_threat_backup is None
    assert updated.assessment_results[0].likelihood == 2
    assert "Pending threat analysis cancelled" in msg
    assert "Threat Alpha" in msg

    msg_idle, _ = cancel_active_flow(SessionStateData())
    assert msg_idle == "Operation cancelled."


@patch(
    "llm_handler.executor.handle_asset_extract",
    return_value=("assets ok", SessionStateData()),
)
@patch(
    "llm_handler.executor.handle_discuss_modify",
    return_value=("discuss ok", SessionStateData()),
)
@patch(
    "llm_handler.executor.handle_reevaluate_filtered",
    return_value=("reeval ok", SessionStateData()),
)
@patch(
    "llm_handler.executor.handle_batch_assessment",
    return_value=("batch ok", SessionStateData()),
)
@patch(
    "llm_handler.executor.handle_rag_info", return_value=("rag ok", SessionStateData())
)
@patch(
    "llm_handler.executor.handle_single_threat_analysis",
    return_value=("single ok", SessionStateData()),
)
def test_executor_routing_and_overrides(
    mock_single,
    mock_rag,
    mock_batch,
    mock_reeval,
    mock_discuss,
    mock_asset,
    mock_rag_engine,
):
    factory = MagicMock()
    intent_chain = MagicMock()
    factory.get_intent_chain.return_value = intent_chain
    analyzer = MagicMock()
    extractor = MagicMock()

    executor = RiskAgentExecutor(
        factory, mock_rag_engine, analyzer, asset_extractor=extractor
    )

    state_pending = SessionStateData(pending_threat={"threat": "Threat Alpha"})
    out, st = executor.process_message("annulla", state_pending)
    assert "cancelled" in out.lower()
    assert len(st.chat_history) == 2

    cases = [
        (IntentType.SINGLE_THREAT_ANALYSIS, "single ok", mock_single),
        (IntentType.RAG_INFO, "rag ok", mock_rag),
        (IntentType.BATCH_ASSESSMENT, "batch ok", mock_batch),
        (IntentType.REEVALUATE_FILTERED, "reeval ok", mock_reeval),
        (IntentType.DISCUSS_MODIFY, "discuss ok", mock_discuss),
        (IntentType.ASSET_EXTRACT, "assets ok", mock_asset),
    ]
    for intent_type, expected_text, mock_fn in cases:
        intent_chain.invoke.return_value = IntentOutput(
            intent=intent_type, extracted_entities={}
        )
        out, st = executor.process_message("test input", SessionStateData())
        assert out == expected_text
        assert mock_fn.called
        assert len(st.chat_history) == 2

    intent_chain.invoke.side_effect = RuntimeError("Router failure")
    out_unk, _ = executor.process_message("???", SessionStateData())
    assert "Intent not recognized" in out_unk

    intent_chain.side_effect = None
    intent_chain.invoke.return_value = IntentOutput(intent=IntentType.RAG_INFO)
    out_p, _ = executor.process_message(
        "Service A and Control A",
        SessionStateData(pending_threat={"threat": "Threat Alpha"}),
    )
    assert out_p == "single ok"

    out_d, _ = executor.process_message(
        "Change score", SessionStateData(active_threat_id="t1")
    )
    assert out_d == "discuss ok"
