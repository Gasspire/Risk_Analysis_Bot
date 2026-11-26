import statistics as stat
from typing import cast
from typing import List, Tuple
from models import (
    ThreatRecord,
    LikelihoodOutput,
    ImpactOutput,
    TypeOfMeasures,
    SynthesisOutput,
)
from core.llm_chains import LLMChainFactory
from core.rag_engine import RAGEngine


class RiskAnalyzer:
    def __init__(self, chain_factory: LLMChainFactory, rag_engine: RAGEngine):
        self.chain_factory = chain_factory
        self.rag_engine = rag_engine

        self.classification_chain = self.chain_factory.get_classification_chain()
        self.likelihood_chain = self.chain_factory.get_likelihood_chain()
        self.impact_chain = self.chain_factory.get_impact_chain()
        self.synthesis_chain = self.chain_factory.get_synthesis_chain()

    def classify_security_measures(
        self, threat: str, assets: str, measures: str
    ) -> Tuple[str, str]:
        """Classifica le misure di sicurezza in preventive e mitigative."""
        query_class = f"Security Measures: {measures} | Threat: {threat}"
        rag_context = self.rag_engine.retrieve_context(query_class)

        try:
            res_class = cast(
                TypeOfMeasures,
                self.classification_chain.invoke(
                    {
                        "rag_context": rag_context,
                        "threat": threat,
                        "affected_assets": assets,
                        "security_measures": measures,
                    }
                ),
            )
            measures_lower = measures.lower()
            valid_measures = [
                m
                for m in res_class.measures
                if m.measure_name and m.measure_name.strip().lower() in measures_lower
            ]
            prev_list = [m.measure_name for m in valid_measures if m.is_preventive]
            mit_list = [m.measure_name for m in valid_measures if m.is_mitigative]

            preventive_str = ", ".join(prev_list) if prev_list else "None"
            mitigative_str = ", ".join(mit_list) if mit_list else "None"
            return preventive_str, mitigative_str
        except Exception as e:
            print(f"Errore nella classificazione delle misure: {e}")
            return measures, measures

    def evaluate_likelihood(
        self,
        threat: str,
        assets: str,
        preventive_measures: str,
        rag_context: str,
        rules_l: str,
        general_context: str,
    ) -> LikelihoodOutput:
        try:
            return cast(
                LikelihoodOutput,
                self.likelihood_chain.invoke(
                    {
                        "general_context": general_context,
                        "rag_context": rag_context,
                        "threat": threat,
                        "affected_assets": assets,
                        "security_measures": preventive_measures,
                        "analyst_custom_rules": rules_l,
                    }
                ),
            )
        except Exception as e:
            print(f"Errore calcolo Likelihood per '{threat}': {e}")
            return LikelihoodOutput(
                historical_frequency_analysis="N/D - Errore elaborazione LLM.",
                preventive_controls_evaluation="N/D - Errore elaborazione LLM.",
                ease_of_execution="N/D - Errore elaborazione LLM.",
                likelihood_reasoning="Errore di calcolo.",
                likelihood_score=1,
            )

    def evaluate_impact(
        self,
        threat: str,
        assets: str,
        mitigative_measures: str,
        rag_context: str,
        rules_i: str,
        general_context: str,
    ) -> ImpactOutput:
        try:
            return cast(
                ImpactOutput,
                self.impact_chain.invoke(
                    {
                        "general_context": general_context,
                        "rag_context": rag_context,
                        "threat": threat,
                        "affected_assets": assets,
                        "security_measures": mitigative_measures,
                        "analyst_custom_rules": rules_i,
                    }
                ),
            )
        except Exception as e:
            print(f"Errore calcolo Impatto per '{threat}': {e}")
            return ImpactOutput(
                worst_case_scenario="N/D - Errore elaborazione LLM.",
                mitigative_controls_evaluation="N/D - Errore elaborazione LLM.",
                human_safety_and_business_impact="N/D - Errore elaborazione LLM.",
                impact_reasoning="Errore di calcolo.",
                impact_score=1,
            )

    def synthesize_reasoning(
        self, threat: str, type_score: str, final_score: int, reasonings: List[str]
    ) -> str:
        formatted_reasonings = "\n".join(f"- {r}" for r in reasonings)
        try:
            res = cast(
                SynthesisOutput,
                self.synthesis_chain.invoke(
                    {
                        "threat": threat,
                        "type": type_score,
                        "final_score": final_score,
                        "reasonings": formatted_reasonings,
                    }
                ),
            )
            return res.synthesized_reasoning
        except Exception as e:
            print(f"Errore sintesi reasoning per '{threat}': {e}")
            return reasonings[0] if reasonings else "Errore di calcolo nella sintesi."

    def evaluate_single_threat(
        self,
        record: ThreatRecord,
        general_context: str,
        rules_l: str,
        rules_i: str,
        num_iterations: int = 1,
        reclassify_measures: bool = True,
        eval_likelihood: bool = True,
        eval_impact: bool = True,
    ) -> ThreatRecord:
        """Valuta una singola minaccia aggiornando e restituendo un oggetto ThreatRecord."""
        if reclassify_measures:
            prev_str, mit_str = self.classify_security_measures(
                record.threat, record.affected_assets, record.security_measures
            )
        else:
            prev_str = record.preventive_measures
            mit_str = record.mitigative_measures

        rag_context_l = (
            self.rag_engine.retrieve_context(
                f"Threat: {record.threat} | Assets: {record.affected_assets} | Preventive Measures: {prev_str}",
                k=6,
            )
            if eval_likelihood
            else ""
        )
        rag_context_i = (
            self.rag_engine.retrieve_context(
                f"Threat: {record.threat} | Assets: {record.affected_assets} | Mitigative Measures: {mit_str}",
                k=6,
            )
            if eval_impact
            else ""
        )

        l_scores, l_reasons = [], []
        i_scores, i_reasons = [], []

        for _ in range(num_iterations):
            if eval_likelihood:
                res_l = self.evaluate_likelihood(
                    record.threat,
                    record.affected_assets,
                    prev_str,
                    rag_context_l,
                    rules_l,
                    general_context,
                )
                l_scores.append(res_l.likelihood_score)
                l_reasons.append(res_l.likelihood_reasoning)

            if eval_impact:
                res_i = self.evaluate_impact(
                    record.threat,
                    record.affected_assets,
                    mit_str,
                    rag_context_i,
                    rules_i,
                    general_context,
                )
                i_scores.append(res_i.impact_score)
                i_reasons.append(res_i.impact_reasoning)

        if eval_likelihood:
            if num_iterations == 1:
                final_l_score = l_scores[0]
                final_l_reasoning = l_reasons[0]
            else:
                try:
                    final_l_score = stat.mode(l_scores)
                except stat.StatisticsError:
                    final_l_score = l_scores[0] if l_scores else 1
                valid_l_reasons = [
                    r for r, s in zip(l_reasons, l_scores) if s == final_l_score
                ]
                final_l_reasoning = self.synthesize_reasoning(
                    record.threat, "Likelihood", final_l_score, valid_l_reasons
                )
            record.likelihood = final_l_score
            record.likelihood_reasoning = final_l_reasoning

        if eval_impact:
            if num_iterations == 1:
                final_i_score = i_scores[0]
                final_i_reasoning = i_reasons[0]
            else:
                try:
                    final_i_score = stat.mode(i_scores)
                except stat.StatisticsError:
                    final_i_score = i_scores[0] if i_scores else 1
                valid_i_reasons = [
                    r for r, s in zip(i_reasons, i_scores) if s == final_i_score
                ]
                final_i_reasoning = self.synthesize_reasoning(
                    record.threat, "Impact", final_i_score, valid_i_reasons
                )
            record.impact = final_i_score
            record.impact_reasoning = final_i_reasoning

        record.risk_factor = record.likelihood * record.impact
        record.preventive_measures = prev_str
        record.mitigative_measures = mit_str

        return record

    def evaluate_list_of_threats(
        self,
        threat_list: List[ThreatRecord],
        general_context: str,
        rules_l: str,
        rules_i: str,
        num_iterations: int = 1,
    ) -> List[ThreatRecord]:
        """Valuta una lista completa di oggetti ThreatRecord."""
        return [
            self.evaluate_single_threat(
                record=record,
                general_context=general_context,
                rules_l=rules_l,
                rules_i=rules_i,
                num_iterations=num_iterations,
            )
            for record in threat_list
        ]
