from models import (
    BotStateEnum,
    IntentType,
    IntentOutput,
    ThreatRecord,
    ExtractedAssetRecord,
    SessionStateData,
    MeasureAnalysis,
    TypeOfMeasures,
    ImpactOutput,
    LikelihoodOutput,
    SynthesisOutput,
    ClusterAssetEvaluation,
    AssetExtractionOutput,
    AssetDescriptionsOutput,
)


def test_enums_values():
    assert BotStateEnum.IDLE.value == "IDLE"
    assert BotStateEnum.AWAITING_THREAT_DETAILS.value == "AWAITING_THREAT_DETAILS"
    assert BotStateEnum.DISCUSSING_THREAT.value == "DISCUSSING_THREAT"

    assert IntentType.RAG_INFO.value == "rag_info"
    assert IntentType.SINGLE_THREAT_ANALYSIS.value == "single_threat_analysis"
    assert IntentType.BATCH_ASSESSMENT.value == "batch_assessment"
    assert IntentType.REEVALUATE_FILTERED.value == "reevaluate_filtered"
    assert IntentType.DISCUSS_MODIFY.value == "discuss_modify"
    assert IntentType.ASSET_EXTRACT.value == "asset_extract"
    assert IntentType.UNKNOWN.value == "unknown"


def test_threat_record_and_session_state_defaults():
    record = ThreatRecord(
        id="t1",
        threat="Threat Alpha",
        affected_assets="Primary Service",
        security_measures="Access Control",
    )
    assert record.likelihood == 1
    assert record.impact == 1
    assert record.risk_factor == 1
    assert record.preventive_measures == "Non ancora classificate"
    assert record.mitigative_measures == "Non ancora classificate"

    asset = ExtractedAssetRecord(name="Core Database")
    assert asset.description == "Descrizione non disponibile."

    state = SessionStateData()
    assert state.bot_state == BotStateEnum.IDLE
    assert state.pending_threat is None
    assert state.active_threat_id is None
    assert state.assessment_results == []
    assert state.extracted_assets == []
    assert state.chat_history == []
    assert state.num_iterations == 1


def test_llm_output_models():
    intent_out = IntentOutput(
        intent=IntentType.RAG_INFO,
        extracted_entities={"topic": "Policy A"},
    )
    assert intent_out.intent == IntentType.RAG_INFO
    assert intent_out.extracted_entities["topic"] == "Policy A"

    measure = MeasureAnalysis(
        measure_name="Control X",
        is_preventive=True,
        preventive_rationale="Prevents unauthorized events",
        is_mitigative=False,
        mitigative_rationale="N/A",
    )
    measures_out = TypeOfMeasures(measures=[measure])
    assert len(measures_out.measures) == 1

    impact_out = ImpactOutput(
        worst_case_scenario="Service outage",
        mitigative_controls_evaluation="Partial",
        human_safety_and_business_impact="Operational downtime",
        impact_reasoning="Significant operational impact.",
        impact_score=4,
    )
    assert impact_out.impact_score == 4

    likelihood_out = LikelihoodOutput(
        historical_frequency_analysis="Occasional",
        preventive_controls_evaluation="Strong controls",
        ease_of_execution="Moderate",
        likelihood_reasoning="Controls reduce probability.",
        likelihood_score=2,
    )
    assert likelihood_out.likelihood_score == 2

    synth = SynthesisOutput(synthesized_reasoning="Combined summary.")
    assert synth.synthesized_reasoning == "Combined summary."

    cluster_eval = ClusterAssetEvaluation(
        cluster_id=0,
        contains_protectable_assets=True,
        extracted_assets=["App Server", "Object Store"],
    )
    asset_ext = AssetExtractionOutput(clusters_evaluation=[cluster_eval])
    assert asset_ext.clusters_evaluation[0].extracted_assets == [
        "App Server",
        "Object Store",
    ]

    desc_out = AssetDescriptionsOutput(
        assets=[ExtractedAssetRecord(name="App Server", description="Handles requests.")]
    )
    assert desc_out.assets[0].name == "App Server"
