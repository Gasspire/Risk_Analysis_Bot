import json
import uuid
import streamlit as st

from config import DEFAULT_CONTEXT
from models import SessionStateData, ThreatRecord, BotStateEnum
from storage.session_repository import SessionRepository
from storage.chat_repository import ChatRepository
from core.llm_chains import LLMChainFactory
from core.rag_engine import RAGEngine
from core.risk_analyzer import RiskAnalyzer
from llm_handler.executor import RiskAgentExecutor
from llm_handler.formatting import get_risk_indicator


st.set_page_config(page_title="Risk Assessment Copilot", layout="wide")

@st.cache_resource
def get_services():
    repository = SessionRepository()
    chat_repository = ChatRepository()
    chain_factory = LLMChainFactory()
    rag_engine = RAGEngine()
    risk_analyzer = RiskAnalyzer(chain_factory=chain_factory, rag_engine=rag_engine)
    agent_executor = RiskAgentExecutor(
        chain_factory=chain_factory,
        rag_engine=rag_engine,
        risk_analyzer=risk_analyzer,
    )
    return repository, chat_repository, rag_engine, risk_analyzer, agent_executor

repository, chat_repository, rag_engine, risk_analyzer, agent_executor = get_services()

if "state_data" not in st.session_state:
    loaded_state = repository.load(default_context=DEFAULT_CONTEXT)
    st.session_state.state_data = loaded_state

state: SessionStateData = st.session_state.state_data

# Inizializzazione repository chat e migrazione se cartella vuota
available_chats = chat_repository.list_chats()
if not available_chats:
    if state.chat_history:
        initial_name = "Chat_Iniziale"
        chat_repository.save_chat(initial_name, state.chat_history)
        available_chats = [initial_name]
    else:
        initial_name = chat_repository.create_chat("Chat_1")
        available_chats = [initial_name]

if "current_chat" not in st.session_state or st.session_state.current_chat not in available_chats:
    st.session_state.current_chat = available_chats[0]
    state.chat_history = chat_repository.load_chat(st.session_state.current_chat)

def refresh_conversational_context(target_state: SessionStateData):
    """Refresha il contesto conversazionale: azzera stato bot e minacce pendenti (Opzione A)."""
    target_state.pending_threat = None
    target_state.active_threat_id = None
    target_state.bot_state = BotStateEnum.IDLE

st.title("Risk Assessment Copilot")

with st.sidebar:
    st.header("💬 Sessioni Chat")

    current_chat_index = (
        available_chats.index(st.session_state.current_chat)
        if st.session_state.current_chat in available_chats
        else 0
    )

    selected_chat = st.selectbox(
        "Seleziona Chat:",
        options=available_chats,
        index=current_chat_index,
        key="chat_select_box",
    )

    if selected_chat != st.session_state.current_chat:
        st.session_state.current_chat = selected_chat
        state.chat_history = chat_repository.load_chat(selected_chat)
        refresh_conversational_context(state)
        repository.save(state)
        st.rerun()

    with st.expander("✏️ Rinomina chat corrente"):
        rename_col1, rename_col2 = st.columns([3, 1])
        with rename_col1:
            new_chat_title = st.text_input(
                "Nuovo nome chat:",
                value=st.session_state.current_chat,
                label_visibility="collapsed",
                key="rename_chat_title_input",
            )
        with rename_col2:
            if st.button("Salva", help="Salva il nuovo nome per questa chat"):
                if new_chat_title.strip() and new_chat_title.strip() != st.session_state.current_chat:
                    ok, res = chat_repository.rename_chat(st.session_state.current_chat, new_chat_title.strip())
                    if ok:
                        st.session_state.current_chat = res
                        st.rerun()
                    else:
                        st.error(res)

    st.markdown("**Nuova Conversazione:**")
    new_chat_col1, new_chat_col2 = st.columns([3, 1])
    with new_chat_col1:
        new_chat_name_input = st.text_input(
            "Nome Chat",
            placeholder="es. Analisi_Cloud",
            label_visibility="collapsed",
            key="new_chat_name_input",
        )
    with new_chat_col2:
        if st.button("➕ Crea", help="Crea una nuova chat e refresha il contesto"):
            new_name = chat_repository.create_chat(new_chat_name_input.strip() if new_chat_name_input else None)
            st.session_state.current_chat = new_name
            state.chat_history = []
            refresh_conversational_context(state)
            repository.save(state)
            st.rerun()

    if len(available_chats) > 1:
        if st.button("🗑️ Elimina Chat Selezionata", help=f"Elimina definitivamente '{st.session_state.current_chat}'"):
            chat_repository.delete_chat(st.session_state.current_chat)
            remaining = chat_repository.list_chats()
            if not remaining:
                new_name = chat_repository.create_chat("Chat_1")
                remaining = [new_name]
            st.session_state.current_chat = remaining[0]
            state.chat_history = chat_repository.load_chat(remaining[0])
            refresh_conversational_context(state)
            repository.save(state)
            st.rerun()

    st.divider()
    st.header("🛠️ Configurazione Assessment")
    st.subheader("1. Setup Iniziale")

    new_context = st.text_area("Contesto Generale:", value=state.general_context, height=150)
    if new_context != state.general_context:
        state.general_context = new_context
        repository.save(state)

    uploaded_threats = st.file_uploader("Carica file JSON delle minacce", type=["json"])
    if uploaded_threats is not None and st.button("Importa Minacce"):
        raw_data = json.load(uploaded_threats)
        imported_records = []
        for t in raw_data:
            record = ThreatRecord(
                id=t.get("id") or str(uuid.uuid4()),
                threat=t.get("threat", "Senza nome"),
                affected_assets=t.get("affected_assets", "N/D"),
                security_measures=t.get("security_measures", "N/D"),
                preventive_measures=t.get("preventive_measures", "Non ancora classificate"),
                mitigative_measures=t.get("mitigative_measures", "Non ancora classificate"),
                likelihood=t.get("likelihood", 1),
                impact=t.get("impact", 1),
                risk_factor=t.get("risk_factor", 1),
                likelihood_reasoning=t.get("likelihood_reasoning", "Da valutare."),
                impact_reasoning=t.get("impact_reasoning", "Da valutare."),
            )
            imported_records.append(record)
        state.assessment_results = imported_records
        repository.save(state)
        st.rerun()

    uploaded_rag = st.file_uploader("Carica documenti (Policy/Asset) per RAG", accept_multiple_files=True, type=["txt"])
    if st.button("Aggiorna Contesto RAG"):
        with st.spinner("Indicizzazione..."):
            text_sources = [
                {"content": f.read().decode("utf-8"), "source_name": f.name}
                for f in uploaded_rag
            ]
            added = rag_engine.add_documents_from_texts(text_sources)
            if added > 0:
                st.success(f"Aggiunti {added} segmenti.")
            else:
                st.warning("Nessun documento processato.")

    st.divider()
    st.subheader("2. Regole di Valutazione")
    new_rules_l = st.text_area("Regole Likelihood:", value=state.custom_rules_likelihood, height=100)
    if new_rules_l != state.custom_rules_likelihood:
        state.custom_rules_likelihood = new_rules_l
        repository.save(state)

    new_rules_i = st.text_area("Regole Impact:", value=state.custom_rules_impact, height=100)
    if new_rules_i != state.custom_rules_impact:
        state.custom_rules_impact = new_rules_i
        repository.save(state)

    st.divider()
    st.subheader("3. Parametri Calcolo LLM")
    state.num_iterations = st.number_input(
        "Numero Iterazioni per Minaccia (Moda/Sintesi):",
        min_value=1,
        max_value=10,
        value=state.num_iterations,
        help="Se >1 calcola la moda statistica dei punteggi."
    )


tab_chat, tab_dati = st.tabs(["Assistente Analisi (Chat)", "Report Minacce"])

with tab_chat:
    col_info1, col_info2 = st.columns([8, 2])
    with col_info1:
        st.caption(f"📁 Chat attiva: **{st.session_state.current_chat}** ({len(state.chat_history)} messaggi)")
    with col_info2:
        if st.button("🔄 Svuota Chat", help="Svuota i messaggi della chat attiva e refresha lo stato conversazionale"):
            state.chat_history = []
            refresh_conversational_context(state)
            chat_repository.save_chat(st.session_state.current_chat, [])
            repository.save(state)
            st.rerun()

    for message in state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt_input := st.chat_input("Invia un comando all'assistente..."):
        with st.chat_message("user"):
            st.markdown(prompt_input)

        with st.chat_message("assistant"):
            with st.spinner("Elaborazione in corso..."):
                output_text, updated_state = agent_executor.process_message(
                    user_input=prompt_input,
                    state=state
                )
                st.session_state.state_data = updated_state
                # Salva sia sul file della chat attiva che sul repository globale
                chat_repository.save_chat(st.session_state.current_chat, updated_state.chat_history)
                repository.save(updated_state)
                st.markdown(output_text)

        st.rerun()

with tab_dati:
    if not state.assessment_results:
        st.info("Nessuna minaccia caricata.")
    else:
        if st.button("🚀 Esegui Valutazione Automatica (LLM) su tutte le minacce"):
            with st.spinner("Valutazione in corso..."):
                state.assessment_results = risk_analyzer.evaluate_list_of_threats(
                    threat_list=state.assessment_results,
                    general_context=state.general_context,
                    rules_l=state.custom_rules_likelihood,
                    rules_i=state.custom_rules_impact,
                    num_iterations=state.num_iterations,
                )
                repository.save(state)
                st.success("Valutazione completata.")
                st.rerun()

        st.subheader("Elenco Minacce e Valutazioni")
        
        with st.form("manual_override_form"):
            submitted = st.form_submit_button("💾 Salva e applica modifiche manuali")
            
            sorted_threats = sorted(state.assessment_results, key=lambda x: x.risk_factor, reverse=True)

            for record in sorted_threats:
                icon, label = get_risk_indicator(record.risk_factor)
                with st.expander(f"{icon} {record.threat} | {label} (Risk Factor: {record.risk_factor})"):
                    col1, col2 = st.columns([7, 3])
                    with col1:
                        st.markdown(f"**Asset Coinvolti:** {record.affected_assets}")
                        st.markdown(f"**Misure di Sicurezza Totali:** {record.security_measures}")
                        st.markdown(f"- 🛡️ **Misure Preventive:** {record.preventive_measures}")
                        st.markdown(f"- 🩹 **Misure Mitigative:** {record.mitigative_measures}")
                        st.markdown("---")
                        st.markdown("**Ragionamento Likelihood:**")
                        st.write(record.likelihood_reasoning)
                        st.markdown("**Ragionamento Impact:**")
                        st.write(record.impact_reasoning)
                    with col2:
                        st.markdown("**Punteggi (Modificabili)**")
                        new_l = st.number_input(
                            "Likelihood (1-5)",
                            min_value=1,
                            max_value=5,
                            value=int(record.likelihood),
                            key=f"l_{record.id}",
                        )
                        new_i = st.number_input(
                            "Impact (1-5)",
                            min_value=1,
                            max_value=5,
                            value=int(record.impact),
                            key=f"i_{record.id}",
                        )
                        # Aggiornamento sui campi del record Pydantic se inviato il form
                        if submitted:
                            record.likelihood = new_l
                            record.impact = new_i
                            record.risk_factor = new_l * new_i

            if submitted:
                repository.save(state)
                st.success("Modifiche salvate con successo.")
                st.rerun()
