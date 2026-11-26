from typing import Tuple, Optional, cast
from models import SessionStateData, BotStateEnum, IntentOutput, IntentType
from core.rag_engine import RAGEngine
from core.risk_analyzer import RiskAnalyzer
from core.llm_chains import LLMChainFactory
from core.asset_extractor import AssetExtractor
from llm_handler.handlers import (
    handle_rag_info,
    handle_single_threat_analysis,
    handle_batch_assessment,
    handle_reevaluate_filtered,
    handle_discuss_modify,
    handle_asset_extract,
)


CANCEL_PHRASES = {
    "annulla",
    "annulla tutto",
    "annulla modifiche",
    "annulla analisi",
    "esci",
    "esci dalla discussione",
    "stop",
    "basta",
    "cancel",
    "cancel all",
    "exit",
    "abort",
    "quit",
    "nevermind",
    "never mind",
}


def is_cancel_command(user_input: str) -> bool:
    """Restituisce True se il messaggio corrisponde a un comando di annullamento/uscita."""
    normalized = user_input.strip().lstrip("/").rstrip(".,;:!?").strip().lower()
    return normalized in CANCEL_PHRASES


def cancel_active_flow(state: SessionStateData) -> Tuple[str, SessionStateData]:
    """Annulla il flusso pendente o la discussione attiva, ripristinando eventuali modifiche."""
    messages = []

    if state.pending_threat is not None:
        state.pending_threat = None
        messages.append(
            "Pending threat analysis cancelled and temporary data discarded."
        )

    if state.pending_discussion_candidates:
        state.pending_discussion_candidates = None
        state.pending_discussion_input = None
        messages.append("Threat selection cancelled.")

    if state.active_threat_id is not None:
        restored_name = None
        if state.active_threat_backup is not None:
            for idx, rec in enumerate(state.assessment_results):
                if rec.id == state.active_threat_id:
                    state.assessment_results[idx] = (
                        state.active_threat_backup.model_copy(deep=True)
                    )
                    restored_name = state.active_threat_backup.threat
                    break
        state.active_threat_id = None
        state.active_threat_backup = None
        if restored_name:
            messages.append(
                f"Discussion cancelled. Any modifications to **{restored_name}** have been discarded."
            )
        else:
            messages.append("Discussion cancelled.")

    state.bot_state = BotStateEnum.IDLE
    return " ".join(messages) if messages else "Operation cancelled.", state


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
        asset_extractor: Optional[
            AssetExtractor
        ] = None,  # Se non passato inizializzo io
    ):
        self.chain_factory = (
            chain_factory  # Factory che gli passa poi la catena da utilizzare
        )
        self.rag_engine = rag_engine  # Engine RAG
        self.risk_analyzer = risk_analyzer  # Classe che contiene la logica dell'analisi
        self.asset_extractor = asset_extractor or AssetExtractor(
            chain_factory=chain_factory
        )
        self.intent_chain = (
            self.chain_factory.get_intent_chain()
        )  # mi prendo solo la catena del riconoscitore

    def process_message(
        self, user_input: str, state: SessionStateData
    ) -> Tuple[str, SessionStateData]:
        has_pending = (
            state.pending_threat is not None
        )  # Controllo se c'è una minaccia in sospeso
        has_pending_selection = bool(state.pending_discussion_candidates)
        has_active_discussion = (
            state.active_threat_id is not None
        )  # Controllo se ci sono discussioni attive

        if (
            has_pending or has_pending_selection or has_active_discussion
        ) and is_cancel_command(user_input):
            output_text, updated_state = cancel_active_flow(state)
            updated_state.chat_history.append({"role": "user", "content": user_input})
            updated_state.chat_history.append(
                {"role": "assistant", "content": output_text}
            )
            return output_text, updated_state

        try:
            intent_res: IntentOutput = cast(
                IntentOutput,
                self.intent_chain.invoke(
                    {"input": user_input}
                ),  # Invoco il riconoscitore di intenti
            )
            intent = intent_res.intent  # prendo l'intento
            entities = (
                intent_res.extracted_entities or {}
            )  # prendo le eventuali entità estratte

        except Exception as e:
            print(f"Routing error: {e}")
            intent = IntentType.UNKNOWN
            entities = {}

        if (
            has_pending
        ):  # se c'è una minaccia in sospeso, assegna l'intento di analisi singola
            intent = IntentType.SINGLE_THREAT_ANALYSIS
        elif (
            has_pending_selection or has_active_discussion
        ):  # se c'è una selezione o discussione attiva, assegna l'intento di discussione
            intent = IntentType.DISCUSS_MODIFY

        # In base all'intento, scelgo l'handler
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
                chain_factory=self.chain_factory,
            )

        elif intent == IntentType.DISCUSS_MODIFY:
            output_text, updated_state = handle_discuss_modify(
                user_input=user_input,
                state=state,
                chain_factory=self.chain_factory,
                entities=entities,
                risk_analyzer=self.risk_analyzer,
            )
        elif intent == IntentType.ASSET_EXTRACT:
            output_text, updated_state = handle_asset_extract(
                state=state,
                asset_extractor=self.asset_extractor,
            )

        else:
            output_text = "Intent not recognized. Please reformulate your request."
            updated_state = state

        updated_state.chat_history.append(
            {"role": "user", "content": user_input}
        )  # Aggiorno la chat
        updated_state.chat_history.append({"role": "assistant", "content": output_text})

        return output_text, updated_state
