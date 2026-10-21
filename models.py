from enum import Enum
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class BotStateEnum(str, Enum):
    IDLE = "IDLE"
    AWAITING_THREAT_DETAILS = "AWAITING_THREAT_DETAILS"
    DISCUSSING_THREAT = "DISCUSSING_THREAT"


class IntentType(str, Enum):
    RAG_INFO = "rag_info"
    SINGLE_THREAT_ANALYSIS = "single_threat_analysis"
    BATCH_ASSESSMENT = "batch_assessment"
    REEVALUATE_FILTERED = "reevaluate_filtered"
    DISCUSS_MODIFY = "discuss_modify"
    UNKNOWN = "unknown"


class IntentOutput(BaseModel):
    intent: IntentType = Field(
        description="The identified intent. Must be exactly one of the defined Enum values."
    )
    extracted_entities: Dict[str, Any] = Field(
        default_factory=dict,
        description="Any extracted entities from the user prompt like threat name, affected assets, or security measures."
    )


class ThreatRecord(BaseModel):
    id: str
    threat: str
    affected_assets: str
    security_measures: str
    preventive_measures: str = "Non ancora classificate"
    mitigative_measures: str = "Non ancora classificate"
    likelihood: int = 1
    impact: int = 1
    risk_factor: int = 1
    likelihood_reasoning: str = "Da valutare."
    impact_reasoning: str = "Da valutare."


class SessionStateData(BaseModel):
    bot_state: BotStateEnum = BotStateEnum.IDLE
    pending_threat: Optional[Dict[str, Optional[str]]] = None
    active_threat_id: Optional[str] = None
    assessment_results: List[ThreatRecord] = Field(default_factory=list)
    chat_history: List[Dict[str, str]] = Field(default_factory=list)
    general_context: str = ""
    custom_rules_likelihood: str = ""
    custom_rules_impact: str = ""
    num_iterations: int = 1


class MeasureAnalysis(BaseModel):
    measure_name: str = Field(description="The exact name of the security measure from the input.")
    is_preventive: bool = Field(description="True if the measure acts BEFORE the incident to reduce its probability.")
    preventive_rationale: str = Field(description="Brief explanation of why it acts preventively, or 'N/A' if it does not.")
    is_mitigative: bool = Field(description="True if the measure acts DURING or AFTER the incident (including detection) to reduce damage.")
    mitigative_rationale: str = Field(description="Brief explanation of why it acts mitigatively, or 'N/A' if it does not.")


class TypeOfMeasures(BaseModel):
    measures: List[MeasureAnalysis] = Field(description="List containing the analysis for EVERY security measure provided in the input.")


class ImpactOutput(BaseModel):
    worst_case_scenario: str = Field(description="Describe the worst credible physical, logical, or data damage to the explicitly affected assets, assuming the threat is fully successful.")
    mitigative_controls_evaluation: str = Field(description="Evaluate if the provided mitigative controls actually limit the physical/business damage, or if they merely detect/record it.")
    human_safety_and_business_impact: str = Field(description="Assess the specific consequences on human life, physical property, data confidentiality, and organizational continuity based on the mitigated worst-case scenario.")
    impact_reasoning: str = Field(description="A concise summary (max 60 words) combining the previous three points to justify the final impact level.")
    impact_score: int = Field(description="Integer value from 1 to 5 representing the final impact score based strictly on the scales.")


class LikelihoodOutput(BaseModel):
    historical_frequency_analysis: str = Field(description="Analyze the natural or historical frequency of the threat. Is it continuous, frequent, occasional, or rare?")
    preventive_controls_evaluation: str = Field(description="Evaluate how the explicitly provided preventive controls obstruct the threat.")
    ease_of_execution: str = Field(description="Assess how easily the threat can materialize given the specific environment and remaining vulnerabilities.")
    likelihood_reasoning: str = Field(description="A concise summary (max 60 words) combining the previous three points to justify the final likelihood level.")
    likelihood_score: int = Field(description="Integer value from 1 to 5 representing the final likelihood score based strictly on the scales.")


class SynthesisOutput(BaseModel):
    synthesized_reasoning: str = Field(description="A coherent, single reasoning of max 80 words that strictly justifies the provided Final Score, ignoring conflicting information.")