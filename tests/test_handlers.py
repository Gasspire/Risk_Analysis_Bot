from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from models import SessionStateData, BotStateEnum, ExtractedAssetRecord
from core.asset_extractor import AssetExtractionPipelineResult
from llm_handler.handlers import (
    handle_rag_info,
    handle_single_threat_analysis,
    handle_batch_assessment,
    handle_reevaluate_filtered,
    handle_asset_extract,
    handle_discuss_modify,
)


def test_handle_rag_info_success_and_error(mock_rag_engine):
    state = SessionStateData(general_context="Synthetic Platform")
    factory = MagicMock()
    chain = MagicMock()
    chain.invoke.return_value = SimpleNamespace(content="Reference policy summary.")
    factory.get_rag_info_chain.return_value = chain

    resp, _ = handle_rag_info("Explain policy", state, mock_rag_engine, factory)
    assert "Reference policy summary." in resp

    chain.invoke.side_effect = RuntimeError("LLM unreachable")
    err_resp, _ = handle_rag_info("Explain policy", state, mock_rag_engine, factory)
    assert "An error occurred while querying the knowledge base" in err_resp


@patch("llm_handler.handlers.ChatPromptTemplate")
def test_handle_single_threat_analysis_missing_and_complete(
    mock_prompt_cls, sample_threat_factory
):
    state = SessionStateData()
    analyzer = MagicMock()
    factory = MagicMock()
    factory.get_llm.return_value = MagicMock()

    mock_chain = MagicMock()
    mock_prompt = MagicMock()
    mock_prompt.__or__.return_value.__or__.return_value = mock_chain
    mock_prompt_cls.from_messages.return_value = mock_prompt

    mock_chain.invoke.return_value = {
        "threat": "Token Theft",
        "affected_assets": "null",
        "security_measures": None,
    }
    resp, updated = handle_single_threat_analysis(
        "Analyze token theft", state, analyzer, factory, {}
    )
    assert updated.bot_state == BotStateEnum.AWAITING_THREAT_DETAILS
    assert updated.pending_threat["threat"] == "Token Theft"
    assert "affected assets" in resp

    mock_chain.invoke.return_value = {
        "threat": "Token Theft",
        "affected_assets": "Auth Gateway",
        "security_measures": "MFA",
    }
    analyzer.evaluate_single_threat.return_value = sample_threat_factory(
        threat_id="t1",
        threat="Token Theft",
        affected_assets="Auth Gateway",
        security_measures="MFA",
    )
    resp2, updated2 = handle_single_threat_analysis(
        "Auth Gateway and MFA", updated, analyzer, factory, {}
    )
    assert updated2.bot_state == BotStateEnum.IDLE
    assert updated2.pending_threat is None
    assert len(updated2.assessment_results) == 1
    assert "Threat Analysis Completed" in resp2


def test_handle_batch_assessment(sample_threat_factory):
    analyzer = MagicMock()

    empty_resp, _ = handle_batch_assessment(SessionStateData(), analyzer)
    assert "No threats present" in empty_resp

    r1 = sample_threat_factory(
        threat_id="1", threat="LowThreat", likelihood=1, impact=2, risk_factor=2
    )
    r2 = sample_threat_factory(
        threat_id="2", threat="HighThreat", likelihood=4, impact=5, risk_factor=20
    )
    analyzer.evaluate_list_of_threats.return_value = [r1, r2]
    state = SessionStateData(assessment_results=[r1, r2])

    resp, _ = handle_batch_assessment(state, analyzer)
    assert "Batch Assessment Completed (2 threats processed)" in resp
    assert resp.index("HighThreat") < resp.index("LowThreat")


@patch("llm_handler.handlers.ChatPromptTemplate")
def test_handle_reevaluate_filtered_variants(mock_prompt_cls, sample_session_state):
    analyzer = MagicMock()
    empty_resp, _ = handle_reevaluate_filtered(SessionStateData(), analyzer, {})
    assert "No threats present" in empty_resp

    r1, r2 = sample_session_state.assessment_results

    mock_chain = MagicMock()
    mock_prompt = MagicMock()
    mock_prompt.__or__.return_value.__or__.return_value = mock_chain
    mock_prompt_cls.from_messages.return_value = mock_prompt
    mock_chain.invoke.return_value = {
        "filter_keyword": "Auth",
        "additional_rule_l": "Assume public exposure",
        "additional_rule_i": "null",
        "general_observation": "High-priority environment",
    }

    factory = MagicMock()
    factory.get_llm.return_value = MagicMock()
    analyzer.evaluate_list_of_threats.return_value = [
        r1.model_copy(update={"likelihood": 5, "risk_factor": 20})
    ]

    resp, updated = handle_reevaluate_filtered(
        sample_session_state,
        analyzer,
        {},
        user_input="Reevaluate Auth",
        chain_factory=factory,
    )
    assert "1 threats re-assessed" in resp
    assert updated.assessment_results[0].risk_factor == 20
    assert updated.assessment_results[1].risk_factor == 4

    analyzer.evaluate_list_of_threats.return_value = [r1]
    resp_num, _ = handle_reevaluate_filtered(
        SessionStateData(assessment_results=[r1, r2]),
        analyzer,
        {"filter_condition": "impact > 3"},
    )
    assert "1 threats re-assessed" in resp_num

    analyzer.evaluate_list_of_threats.return_value = [r2]
    resp_num2, _ = handle_reevaluate_filtered(
        SessionStateData(assessment_results=[r1, r2]),
        analyzer,
        {"filter_condition": "likelihood < 3"},
    )
    assert "1 threats re-assessed" in resp_num2


def test_handle_asset_extract_branches():
    extractor = MagicMock()
    state = SessionStateData(
        general_context="Synthetic Platform",
        extracted_assets=[
            ExtractedAssetRecord(
                name="Auth Gateway", description="Descrizione non disponibile."
            )
        ],
    )

    extractor.extract_assets_pipeline.side_effect = RuntimeError("NLP error")
    err_resp, _ = handle_asset_extract(state, extractor)
    assert "An error occurred during asset extraction" in err_resp

    extractor.extract_assets_pipeline.side_effect = None
    extractor.extract_assets_pipeline.return_value = AssetExtractionPipelineResult(
        candidate_terms_count=0
    )
    zero_resp, _ = handle_asset_extract(state, extractor)
    assert "No system documents or general context found" in zero_resp

    extractor.extract_assets_pipeline.return_value = AssetExtractionPipelineResult(
        candidate_terms_count=10, clusters_after_merge=2, final_assets=[]
    )
    none_resp, _ = handle_asset_extract(state, extractor)
    assert "no protectable assets were selected" in none_resp

    extractor.extract_assets_pipeline.return_value = AssetExtractionPipelineResult(
        candidate_terms_count=10,
        optimal_k=3,
        best_silhouette_score=0.5,
        clusters_after_merge=2,
        selected_clusters=[],
        final_assets=["Auth Gateway", "Billing DB"],
        described_assets=[
            ExtractedAssetRecord(
                name="Auth Gateway", description="Handles auth tokens"
            ),
            ExtractedAssetRecord(name="Billing DB", description="Stores invoices"),
        ],
    )
    ok_resp, updated = handle_asset_extract(state, extractor)
    assert "Asset Extraction Completed" in ok_resp
    assert len(updated.extracted_assets) == 2
    assert updated.extracted_assets[0].description == "Handles auth tokens"


@patch("llm_handler.handlers.ChatPromptTemplate")
def test_handle_discuss_modify_flows(mock_prompt_cls, sample_threat_factory):
    factory = MagicMock()
    factory.get_llm.return_value = MagicMock()
    analyzer = MagicMock()

    empty_resp, _ = handle_discuss_modify("discuss", SessionStateData(), factory, {})
    assert "No evaluated threats" in empty_resp

    rec = sample_threat_factory(
        threat_id="t1",
        threat="API Injection",
        affected_assets="Core API",
        security_measures="WAF",
        likelihood=3,
        impact=4,
        risk_factor=12,
    )
    state = SessionStateData(assessment_results=[rec])

    nf_resp, _ = handle_discuss_modify(
        "discuss unknown", state, factory, {"threat_name": "XYZ"}
    )
    assert "not present in the assessment report" in nf_resp

    mock_chain = MagicMock()
    mock_prompt = MagicMock()
    mock_prompt.__or__.return_value.__or__.return_value = mock_chain
    mock_prompt_cls.from_messages.return_value = mock_prompt

    mock_chain.invoke.return_value = {
        "updated_likelihood": 2,
        "updated_likelihood_reasoning": "WAF blocks common payloads.",
        "is_finished": False,
        "response_text": "",
    }
    resp, updated = handle_discuss_modify(
        "Discuss API Injection",
        state,
        factory,
        {"threat_name": "API Injection"},
        analyzer,
    )
    assert updated.bot_state == BotStateEnum.DISCUSSING_THREAT
    assert updated.active_threat_id == "t1"
    assert updated.assessment_results[0].likelihood == 2
    assert updated.assessment_results[0].risk_factor == 8
    assert "cancel" in resp

    mock_chain.invoke.return_value = {
        "additional_rule_l": "Consider WAF",
        "is_finished": True,
        "response_text": "",
    }
    resp2, updated2 = handle_discuss_modify(
        "Re-evaluate with WAF and close", updated, factory, {}, analyzer
    )
    analyzer.evaluate_single_threat.assert_called_once()
    assert (
        analyzer.evaluate_single_threat.call_args.kwargs.get("reclassify_measures")
        is False
    )
    assert (
        analyzer.evaluate_single_threat.call_args.kwargs.get("eval_likelihood") is True
    )
    assert analyzer.evaluate_single_threat.call_args.kwargs.get("eval_impact") is False
    assert updated2.bot_state == BotStateEnum.IDLE
    assert updated2.active_threat_id is None
    assert "Discussion closed" in resp2


@patch("llm_handler.handlers.ChatPromptTemplate")
def test_handle_discuss_modify_disambiguation(mock_prompt_cls, sample_threat_factory):
    factory = MagicMock()
    factory.get_llm.return_value = MagicMock()

    mock_chain = MagicMock()
    mock_prompt = MagicMock()
    mock_prompt.__or__.return_value.__or__.return_value = mock_chain
    mock_prompt_cls.from_messages.return_value = mock_prompt
    mock_chain.invoke.return_value = {
        "is_finished": False,
        "response_text": "",
    }

    rec1 = sample_threat_factory(
        threat_id="t1",
        threat="SQL Injection",
        affected_assets="Database",
    )
    rec2 = sample_threat_factory(
        threat_id="t2",
        threat="API Injection",
        affected_assets="Core API",
    )
    state = SessionStateData(assessment_results=[rec1, rec2])

    resp, state = handle_discuss_modify(
        "Discuss Injection", state, factory, {"threat_name": "Injection"}
    )
    assert state.bot_state == BotStateEnum.AWAITING_THREAT_SELECTION
    assert state.pending_discussion_candidates == ["t1", "t2"]
    assert "1. **SQL Injection**" in resp
    assert "2. **API Injection**" in resp

    invalid_resp, state = handle_discuss_modify("99", state, factory, {})
    assert state.bot_state == BotStateEnum.AWAITING_THREAT_SELECTION
    assert "Invalid selection" in invalid_resp

    sel_resp, state = handle_discuss_modify("2", state, factory, {})
    assert state.bot_state == BotStateEnum.DISCUSSING_THREAT
    assert state.active_threat_id == "t2"
    assert state.pending_discussion_candidates is None
    assert "API Injection" in sel_resp


@patch("llm_handler.handlers.ChatPromptTemplate")
def test_handle_discuss_modify_first_message_includes_state(
    mock_prompt_cls, sample_threat_factory
):
    factory = MagicMock()
    factory.get_llm.return_value = MagicMock()

    mock_chain = MagicMock()
    mock_prompt = MagicMock()
    mock_prompt.__or__.return_value.__or__.return_value = mock_chain
    mock_prompt_cls.from_messages.return_value = mock_prompt
    mock_chain.invoke.return_value = {
        "is_finished": False,
        "response_text": "Let's review this threat together.",
    }

    rec = sample_threat_factory(
        threat_id="t1",
        threat="Phishing",
        affected_assets="Email Gateway",
        security_measures="MFA",
        likelihood=3,
        impact=4,
        risk_factor=12,
    )
    state = SessionStateData(assessment_results=[rec])

    resp, updated = handle_discuss_modify(
        "Discuss Phishing", state, factory, {"threat_name": "Phishing"}
    )
    assert updated.bot_state == BotStateEnum.DISCUSSING_THREAT
    assert "Current Assessment for: **Phishing**" in resp
    assert "Email Gateway" in resp
    assert "MFA" in resp
    assert "3/5" in resp
    assert "4/5" in resp
    assert "12/25" in resp
    assert "Let's review this threat together." in resp


@patch("llm_handler.handlers.ChatPromptTemplate")
def test_handle_discuss_modify_final_message_includes_new_state_and_diff(
    mock_prompt_cls, sample_threat_factory
):
    factory = MagicMock()
    factory.get_llm.return_value = MagicMock()

    mock_chain = MagicMock()
    mock_prompt = MagicMock()
    mock_prompt.__or__.return_value.__or__.return_value = mock_chain
    mock_prompt_cls.from_messages.return_value = mock_prompt

    rec = sample_threat_factory(
        threat_id="t1",
        threat="Phishing",
        affected_assets="Email Gateway",
        security_measures="MFA",
        likelihood=3,
        impact=4,
        risk_factor=12,
    )
    state = SessionStateData(assessment_results=[rec])

    mock_chain.invoke.return_value = {
        "updated_likelihood": 2,
        "updated_likelihood_reasoning": "MFA blocks credential reuse.",
        "is_finished": False,
        "response_text": "Likelihood lowered to 2.",
    }
    _, state = handle_discuss_modify(
        "Discuss Phishing and lower likelihood to 2",
        state,
        factory,
        {"threat_name": "Phishing"},
    )

    mock_chain.invoke.return_value = {
        "is_finished": True,
        "response_text": "Confirmed lower likelihood due to MFA enforcement.",
    }
    final_resp, final_state = handle_discuss_modify(
        "Confirm and close", state, factory, {}
    )

    assert final_state.bot_state == BotStateEnum.IDLE
    assert final_state.active_threat_id is None
    assert final_state.active_threat_backup is None
    assert "Updated Assessment for: **Phishing**" in final_resp
    assert "### Modifications Applied" in final_resp
    assert "`3/5` → `2/5`" in final_resp
    assert "`12/25` → `8/25`" in final_resp
    assert (
        "**Explanation:** Confirmed lower likelihood due to MFA enforcement."
        in final_resp
    )
    assert "Discussion closed." in final_resp
    assert final_resp.index("Updated Assessment for") < final_resp.index(
        "### Modifications Applied"
    )


@patch("llm_handler.handlers.ChatPromptTemplate")
def test_handle_discuss_modify_direct_close_command(
    mock_prompt_cls, sample_threat_factory
):
    factory = MagicMock()
    rec = sample_threat_factory(
        threat_id="t1",
        threat="Phishing",
        likelihood=3,
        impact=4,
        risk_factor=12,
    )
    state = SessionStateData(
        bot_state=BotStateEnum.DISCUSSING_THREAT,
        active_threat_id="t1",
        active_threat_backup=rec.model_copy(deep=True),
        assessment_results=[rec],
    )

    resp, updated = handle_discuss_modify("confirm", state, factory, {})
    factory.get_llm.assert_not_called()
    assert updated.bot_state == BotStateEnum.IDLE
    assert updated.active_threat_id is None
    assert "Updated Assessment for: **Phishing**" in resp
    assert "Discussion closed." in resp


@patch("llm_handler.handlers.ChatPromptTemplate")
def test_handle_discuss_modify_assets_and_measures_reassessment(
    mock_prompt_cls, sample_threat_factory
):
    factory = MagicMock()
    factory.get_llm.return_value = MagicMock()
    analyzer = MagicMock()

    mock_chain = MagicMock()
    mock_prompt = MagicMock()
    mock_prompt.__or__.return_value.__or__.return_value = mock_chain
    mock_prompt_cls.from_messages.return_value = mock_prompt

    rec = sample_threat_factory(
        threat_id="t1",
        threat="Ransomware",
        affected_assets="File Server",
        security_measures="Firewall, Backups",
        likelihood=3,
        impact=4,
        risk_factor=12,
    )
    rec.preventive_measures = "Firewall"
    rec.mitigative_measures = "Backups"
    state = SessionStateData(
        bot_state=BotStateEnum.DISCUSSING_THREAT,
        active_threat_id="t1",
        active_threat_backup=rec.model_copy(deep=True),
        assessment_results=[rec],
    )

    # 1) Modifica asset -> ricalcola senza riclassificare misure
    mock_chain.invoke.return_value = {
        "updated_affected_assets": "File Server, Database",
        "is_finished": False,
        "response_text": "Added Database.",
    }
    resp1, state = handle_discuss_modify(
        "Add Database to affected assets", state, factory, {}, analyzer
    )
    assert state.assessment_results[0].affected_assets == "File Server, Database"
    assert analyzer.evaluate_single_threat.call_count == 1
    assert (
        analyzer.evaluate_single_threat.call_args.kwargs.get("reclassify_measures")
        is False
    )
    assert (
        analyzer.evaluate_single_threat.call_args.kwargs.get("eval_likelihood") is True
    )
    assert analyzer.evaluate_single_threat.call_args.kwargs.get("eval_impact") is True
    assert "File Server, Database" in resp1

    # 2) Modifica tutte le misure di sicurezza -> ricalcola riclassificando
    mock_chain.invoke.return_value = {
        "updated_security_measures": "Firewall, Backups, EDR",
        "is_finished": False,
        "response_text": "Added EDR.",
    }
    _, state = handle_discuss_modify(
        "Add EDR to security measures", state, factory, {}, analyzer
    )
    assert state.assessment_results[0].security_measures == "Firewall, Backups, EDR"
    assert analyzer.evaluate_single_threat.call_count == 2
    assert (
        analyzer.evaluate_single_threat.call_args.kwargs.get("reclassify_measures")
        is True
    )

    # 3) Modifica solo preventive/mitigative -> ricalcola senza riclassificare
    mock_chain.invoke.return_value = {
        "updated_preventive_measures": "Firewall, MFA",
        "updated_mitigative_measures": "Backups, Immutable Snapshots",
        "is_finished": False,
        "response_text": "Updated preventive and mitigative measures.",
    }
    _, state = handle_discuss_modify(
        "Set preventive to Firewall, MFA and mitigative to Backups, Immutable Snapshots",
        state,
        factory,
        {},
        analyzer,
    )
    assert state.assessment_results[0].preventive_measures == "Firewall, MFA"
    assert (
        state.assessment_results[0].mitigative_measures
        == "Backups, Immutable Snapshots"
    )
    assert analyzer.evaluate_single_threat.call_count == 3
    assert (
        analyzer.evaluate_single_threat.call_args.kwargs.get("reclassify_measures")
        is False
    )
    assert (
        analyzer.evaluate_single_threat.call_args.kwargs.get("eval_likelihood") is True
    )
    assert analyzer.evaluate_single_threat.call_args.kwargs.get("eval_impact") is True

    # 4) Modifica solo mitigative_measures -> ricalcola SOLO impact, NON likelihood
    mock_chain.invoke.return_value = {
        "updated_mitigative_measures": "Backups, Immutable Snapshots, Network isolation",
        "is_finished": False,
        "response_text": "Added Network isolation to mitigative measures.",
    }
    _, state = handle_discuss_modify(
        "Add Network isolation to mitigative measures", state, factory, {}, analyzer
    )
    assert analyzer.evaluate_single_threat.call_count == 4
    assert (
        analyzer.evaluate_single_threat.call_args.kwargs.get("eval_likelihood") is False
    )
    assert analyzer.evaluate_single_threat.call_args.kwargs.get("eval_impact") is True

    # 5) Modifica security_measures con punteggi bloccati -> riclassifica misure senza ricalcolare i punteggi
    analyzer.classify_security_measures.return_value = (
        "Anti-Virus",
        "Anti-Virus, Backup",
    )
    mock_chain.invoke.return_value = {
        "updated_security_measures": "Anti-Virus, Backup",
        "updated_likelihood": 3,
        "updated_impact": 4,
        "is_finished": False,
        "response_text": "Removed Network isolation while keeping scores.",
    }
    _, state = handle_discuss_modify(
        "Remove Network isolation, keep likelihood 3 and impact 4",
        state,
        factory,
        {},
        analyzer,
    )
    analyzer.classify_security_measures.assert_called_once_with(
        "Ransomware", "File Server, Database", "Anti-Virus, Backup"
    )
    assert analyzer.evaluate_single_threat.call_count == 4
    assert state.assessment_results[0].preventive_measures == "Anti-Virus"
    assert state.assessment_results[0].mitigative_measures == "Anti-Virus, Backup"

    # 6) Modifica security_measures con keep_likelihood_unchanged / keep_impact_unchanged (senza numeri espliciti)
    analyzer.classify_security_measures.reset_mock()
    analyzer.classify_security_measures.return_value = ("Anti-Virus", "Backup")
    mock_chain.invoke.return_value = {
        "updated_security_measures": "Anti-Virus, Backup",
        "keep_likelihood_unchanged": True,
        "keep_impact_unchanged": True,
        "is_finished": False,
        "response_text": "Scores kept unchanged.",
    }
    state.assessment_results[0].security_measures = "Anti-Virus, Firewall, Backup"
    _, state = handle_discuss_modify(
        "Eliminate Firewall, let the scores unchanged",
        state,
        factory,
        {},
        analyzer,
    )
    analyzer.classify_security_measures.assert_called_once_with(
        "Ransomware", "File Server, Database", "Anti-Virus, Backup"
    )
    assert analyzer.evaluate_single_threat.call_count == 4
    assert state.assessment_results[0].preventive_measures == "Anti-Virus"
    assert state.assessment_results[0].mitigative_measures == "Backup"
