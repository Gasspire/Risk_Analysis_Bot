import statistics as stat
from unittest.mock import MagicMock, patch
from models import (
    TypeOfMeasures,
    MeasureAnalysis,
    LikelihoodOutput,
    ImpactOutput,
    SynthesisOutput,
)
from core.risk_analyzer import RiskAnalyzer


def _make_analyzer(mock_rag_engine):
    factory = MagicMock()

    cls_chain = MagicMock()
    l_chain = MagicMock()
    i_chain = MagicMock()
    s_chain = MagicMock()

    factory.get_classification_chain.return_value = cls_chain
    factory.get_likelihood_chain.return_value = l_chain
    factory.get_impact_chain.return_value = i_chain
    factory.get_synthesis_chain.return_value = s_chain

    analyzer = RiskAnalyzer(chain_factory=factory, rag_engine=mock_rag_engine)
    return analyzer, cls_chain, l_chain, i_chain, s_chain


def test_classify_security_measures_success_and_fallback(mock_rag_engine):
    analyzer, cls_chain, _, _, _ = _make_analyzer(mock_rag_engine)

    cls_chain.invoke.return_value = TypeOfMeasures(
        measures=[
            MeasureAnalysis(
                measure_name="Control A",
                is_preventive=True,
                preventive_rationale="Blocks unauthorized access",
                is_mitigative=False,
                mitigative_rationale="N/A",
            ),
            MeasureAnalysis(
                measure_name="Control B",
                is_preventive=False,
                preventive_rationale="N/A",
                is_mitigative=True,
                mitigative_rationale="Restores service state",
            ),
        ]
    )
    prev, mit = analyzer.classify_security_measures(
        "Threat X", "Asset Y", "Control A, Control B"
    )
    assert prev == "Control A"
    assert mit == "Control B"

    # Hallucinated measure not in input `measures` must be filtered out
    cls_chain.invoke.return_value = TypeOfMeasures(
        measures=[
            MeasureAnalysis(
                measure_name="Control A",
                is_preventive=True,
                preventive_rationale="Blocks unauthorized access",
                is_mitigative=False,
                mitigative_rationale="N/A",
            ),
            MeasureAnalysis(
                measure_name="Removed Control",
                is_preventive=True,
                preventive_rationale="Hallucinated from RAG",
                is_mitigative=True,
                mitigative_rationale="Hallucinated from RAG",
            ),
        ]
    )
    prev, mit = analyzer.classify_security_measures("Threat X", "Asset Y", "Control A")
    assert prev == "Control A"
    assert mit == "None"

    cls_chain.invoke.return_value = TypeOfMeasures(measures=[])
    prev, mit = analyzer.classify_security_measures("Threat X", "Asset Y", "None")
    assert prev == "None"
    assert mit == "None"

    cls_chain.invoke.side_effect = RuntimeError("LLM down")
    prev, mit = analyzer.classify_security_measures(
        "Threat X", "Asset Y", "Control A, Control B"
    )
    assert prev == "Control A, Control B"
    assert mit == "Control A, Control B"


def test_evaluate_likelihood_and_impact_success_and_fallback(mock_rag_engine):
    analyzer, _, l_chain, i_chain, _ = _make_analyzer(mock_rag_engine)

    l_chain.invoke.return_value = LikelihoodOutput(
        historical_frequency_analysis="Freq",
        preventive_controls_evaluation="Strong",
        ease_of_execution="Hard",
        likelihood_reasoning="Low likelihood",
        likelihood_score=2,
    )
    res_l = analyzer.evaluate_likelihood(
        "Threat X", "Asset Y", "Control A", "ctx", "", ""
    )
    assert res_l.likelihood_score == 2

    l_chain.invoke.side_effect = RuntimeError("Timeout")
    res_l_err = analyzer.evaluate_likelihood(
        "Threat X", "Asset Y", "Control A", "ctx", "", ""
    )
    assert res_l_err.likelihood_score == 1
    assert res_l_err.likelihood_reasoning == "Errore di calcolo."

    i_chain.invoke.return_value = ImpactOutput(
        worst_case_scenario="Service disruption",
        mitigative_controls_evaluation="Weak",
        human_safety_and_business_impact="High operational loss",
        impact_reasoning="High impact",
        impact_score=4,
    )
    res_i = analyzer.evaluate_impact("Threat X", "Asset Y", "Control B", "ctx", "", "")
    assert res_i.impact_score == 4

    i_chain.invoke.side_effect = RuntimeError("Timeout")
    res_i_err = analyzer.evaluate_impact(
        "Threat X", "Asset Y", "Control B", "ctx", "", ""
    )
    assert res_i_err.impact_score == 1
    assert res_i_err.impact_reasoning == "Errore di calcolo."


def test_synthesize_reasoning_success_and_fallback(mock_rag_engine):
    analyzer, _, _, _, s_chain = _make_analyzer(mock_rag_engine)

    s_chain.invoke.return_value = SynthesisOutput(
        synthesized_reasoning="Unified reason."
    )
    assert (
        analyzer.synthesize_reasoning("Threat X", "Likelihood", 3, ["r1", "r2"])
        == "Unified reason."
    )

    s_chain.invoke.side_effect = RuntimeError("Fail")
    assert (
        analyzer.synthesize_reasoning("Threat X", "Likelihood", 3, ["r1", "r2"]) == "r1"
    )
    assert (
        analyzer.synthesize_reasoning("Threat X", "Likelihood", 3, [])
        == "Errore di calcolo nella sintesi."
    )


def test_evaluate_single_and_list_of_threats_single_and_multi_iterations(
    mock_rag_engine, sample_threat_factory
):
    analyzer, cls_chain, l_chain, i_chain, s_chain = _make_analyzer(mock_rag_engine)

    cls_chain.invoke.return_value = TypeOfMeasures(
        measures=[
            MeasureAnalysis(
                measure_name="Control Z",
                is_preventive=True,
                preventive_rationale="Blocks",
                is_mitigative=True,
                mitigative_rationale="Contains",
            )
        ]
    )
    l_chain.invoke.return_value = LikelihoodOutput(
        historical_frequency_analysis="Often",
        preventive_controls_evaluation="Partial",
        ease_of_execution="Medium",
        likelihood_reasoning="L reason",
        likelihood_score=3,
    )
    i_chain.invoke.return_value = ImpactOutput(
        worst_case_scenario="Outage",
        mitigative_controls_evaluation="Partial",
        human_safety_and_business_impact="Moderate",
        impact_reasoning="I reason",
        impact_score=4,
    )
    s_chain.invoke.return_value = SynthesisOutput(synthesized_reasoning="Synthesized.")

    rec = sample_threat_factory(
        threat_id="t1",
        threat="Threat A",
        affected_assets="Service A",
        security_measures="Control Z",
    )
    out = analyzer.evaluate_single_threat(
        rec, "Synthetic Platform", "", "", num_iterations=1
    )
    assert out.likelihood == 3
    assert out.impact == 4
    assert out.risk_factor == 12
    assert out.likelihood_reasoning == "L reason"
    assert out.preventive_measures == "Control Z"

    out_multi = analyzer.evaluate_list_of_threats(
        [
            sample_threat_factory(
                threat_id="t2",
                threat="Threat B",
                affected_assets="Service B",
                security_measures="Control Z",
            )
        ],
        "Synthetic Platform",
        "",
        "",
        num_iterations=3,
    )
    assert len(out_multi) == 1
    assert out_multi[0].likelihood_reasoning == "Synthesized."
    assert out_multi[0].impact_reasoning == "Synthesized."

    with patch(
        "core.risk_analyzer.stat.mode", side_effect=stat.StatisticsError("no mode")
    ):
        rec3 = sample_threat_factory(
            threat_id="t3",
            threat="Threat C",
            affected_assets="Service C",
            security_measures="Control Z",
        )
        out_fallback = analyzer.evaluate_single_threat(
            rec3, "Synthetic Platform", "", "", num_iterations=2
        )
        assert out_fallback.likelihood == 3
        assert out_fallback.impact == 4
