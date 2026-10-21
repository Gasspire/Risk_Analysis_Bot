from typing import Tuple, cast
from models import SessionStateData, BotStateEnum, IntentOutput, IntentType
from core.rag_engine import RAGEngine
from core.risk_analyzer import RiskAnalyzer
from core.llm_chains import LLMChainFactory
from llm_handler.handlers import (
    handle_rag_info,
    handle_single_threat_analysis,
    handle_batch_assessment,
    handle_reevaluate_filtered,
    handle_discuss_modify,
)


class RiskAgentExecutor:
    """
    Orchestratore puro per la gestione del dialogo e delle operazioni del Bot.
    Riceve lo stato immutabile o mutabile e restituisce la risposta formattata e lo stato finale aggiornato.
    """

    def __init__(
        self,
        chain_factory: LLMChainFactory,
        rag_engine: RAGEngine,
        risk_analyzer: RiskAnalyzer,
    ):
        self.chain_factory = chain_factory
        self.rag_engine = rag_engine
        self.risk_analyzer = risk_analyzer
        self.intent_chain = self.chain_factory.get_intent_chain()

    def process_message(
        self, user_input: str, state: SessionStateData
    ) -> Tuple[str, SessionStateData]:
        has_pending = state.pending_threat is not None
        has_active_discussion = state.active_threat_id is not None

        # Determinazione dell'intento
        try:
            intent_res: IntentOutput = cast(IntentOutput,self.intent_chain.invoke({"input": user_input}))
            intent = intent_res.intent
            entities = intent_res.extracted_entities or {}
        except Exception as e:
            print(f"Routing error: {e}")
            intent = IntentType.UNKNOWN
            entities = {}

        # Forced Routing in base allo stato della macchina a stati
        if has_pending:
            intent = IntentType.SINGLE_THREAT_ANALYSIS
        elif has_active_discussion:
            intent = IntentType.DISCUSS_MODIFY

        # Dispatcher dell'Handler
        if intent == IntentType.SINGLE_THREAT_ANALYSIS:
            output_text, updated_state = handle_single_threat_analysis(
                user_input=user_input,
                state=state,
                risk_analyzer=self.risk_analyzer,
                chain_factory=self.chain_factory,
                entities=entities,
            )
        elif intent == IntentType.RAG_INFO:
            output_text, updated_state = handle_rag_info(
                user_input=user_input,
                state=state,
                rag_engine=self.rag_engine,
                chain_factory=self.chain_factory,
            )
        elif intent == IntentType.BATCH_ASSESSMENT:
            output_text, updated_state = handle_batch_assessment(
                state=state,
                risk_analyzer=self.risk_analyzer,
            )
        elif intent == IntentType.REEVALUATE_FILTERED:
            output_text, updated_state = handle_reevaluate_filtered(
                state=state,
                risk_analyzer=self.risk_analyzer,
                entities=entities,
                user_input=user_input,
            )
        elif intent == IntentType.DISCUSS_MODIFY:
            output_text, updated_state = handle_discuss_modify(
                user_input=user_input,
                state=state,
                chain_factory=self.chain_factory,
                entities=entities,
            )
        else:
            output_text = "Intent not recognized. Please reformulate your request."
            updated_state = state

        # Aggiunta alla cronologia chat dello stato
        updated_state.chat_history.append({"role": "user", "content": user_input})
        updated_state.chat_history.append({"role": "assistant", "content": output_text})

        return output_text, updated_state