import uuid
from typing import Tuple, Dict, Any, Optional, List
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from models import SessionStateData, ThreatRecord, BotStateEnum
from core.risk_analyzer import RiskAnalyzer
from core.rag_engine import RAGEngine
from core.llm_chains import LLMChainFactory
from core.asset_extractor import AssetExtractor
from llm_handler.formatting import get_risk_indicator
import prompt


def handle_rag_info(
    user_input: str,
    state: SessionStateData,
    rag_engine: RAGEngine,
    chain_factory: LLMChainFactory,
) -> Tuple[str, SessionStateData]:

    rag_context = rag_engine.retrieve_context(user_input, k=4)
    chain = chain_factory.get_rag_info_chain()

    try:
        response = chain.invoke(
            {
                "general_context": state.general_context or "None provided.",
                "rag_context": rag_context,
                "user_input": user_input,
            }
        )
        return str(response.content), state

    except Exception as e:
        return f"An error occurred while querying the knowledge base: {e}", state


def handle_single_threat_analysis(
    user_input: str,
    state: SessionStateData,
    risk_analyzer: RiskAnalyzer,
    chain_factory: LLMChainFactory,
    entities: Dict[str, Any],
) -> Tuple[str, SessionStateData]:

    pending = state.pending_threat or {}

    current_threat = pending.get("threat") or entities.get("threat")

    current_assets = (
        pending.get("affected_assets")
        or entities.get("affected_assets")
        or entities.get("assets")
    )

    current_measures = (
        pending.get("security_measures")
        or entities.get("security_measures")
        or entities.get("measures")
    )

    llm_json = chain_factory.get_llm(json_format=True)

    extractor_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", prompt.STATE_EXTRACTOR_PROMPT),
            ("human", "{input}"),
        ]
    )

    chain = extractor_prompt | llm_json | JsonOutputParser()

    extracted_state = {}
    try:
        res = chain.invoke(
            {
                "current_threat": current_threat or "null",
                "current_assets": current_assets or "null",
                "current_measures": current_measures or "null",
                "input": user_input,
            }
        )

        if isinstance(res, dict):
            extracted_state = res

    except Exception as e:
        print(f"Errore estrazione JSON: {e}")

    threat = extracted_state.get("threat") or current_threat
    assets = extracted_state.get("affected_assets") or current_assets
    measures = extracted_state.get("security_measures") or current_measures

    # Pulizia stringhe
    threat = (
        threat.strip()
        if isinstance(threat, str) and threat.strip() and threat.lower() != "null"
        else None
    )
    assets = (
        assets.strip()
        if isinstance(assets, str) and assets.strip() and assets.lower() != "null"
        else None
    )
    measures = (
        measures.strip()
        if isinstance(measures, str) and measures.strip() and measures.lower() != "null"
        else None
    )

    # Aggiornamento stato pendente

    state.pending_threat = {
        "threat": threat,
        "affected_assets": assets,
        "security_measures": measures,
    }

    missing_fields = []
    if not threat:
        missing_fields.append("threat (`threat`)")
    if not assets:
        missing_fields.append("affected assets (`affected_assets`)")
    if not measures:
        missing_fields.append("security measures (`security_measures`)")

    if missing_fields:
        state.bot_state = BotStateEnum.AWAITING_THREAT_DETAILS
        missing_str = ", ".join(missing_fields)
        return (
            f"To proceed with the threat analysis, additional details are required.\n"
            f"Please specify the following missing parameter(s): **{missing_str}**.",
            state,
        )

    # Esecuzione analisi se tutti i campi sono presenti
    state.pending_threat = None
    state.bot_state = BotStateEnum.IDLE

    assert threat is not None and assets is not None and measures is not None

    record = ThreatRecord(
        id=str(uuid.uuid4()),
        threat=threat,
        affected_assets=assets,
        security_measures=measures,
    )

    evaluated_record = risk_analyzer.evaluate_single_threat(
        record=record,
        general_context=state.general_context,
        rules_l=state.custom_rules_likelihood,
        rules_i=state.custom_rules_impact,
        num_iterations=state.num_iterations,
    )
    state.assessment_results.append(evaluated_record)

    icon, label = get_risk_indicator(evaluated_record.risk_factor)
    report = (
        f"### Threat Analysis Completed\n\n"
        f"- **Threat:** {evaluated_record.threat}\n"
        f"- **Affected Assets:** {evaluated_record.affected_assets}\n"
        f"- **Security Measures:** {evaluated_record.security_measures}\n\n"
        f"---\n\n"
        f"**Evaluation Results:**\n"
        f"- 🛡️ **Preventive Measures:** {evaluated_record.preventive_measures}\n"
        f"- 🩹 **Mitigative Measures:** {evaluated_record.mitigative_measures}\n"
        f"- 📈 **Likelihood Score:** {evaluated_record.likelihood}/5\n"
        f"  - *Reasoning:* {evaluated_record.likelihood_reasoning}\n"
        f"- 💥 **Impact Score:** {evaluated_record.impact}/5\n"
        f"  - *Reasoning:* {evaluated_record.impact_reasoning}\n\n"
        f"**Final Risk Factor:** {icon} {evaluated_record.risk_factor}/25 ({label})\n\n"
        f"The threat has been successfully analyzed and added to the assessment report."
    )
    return report, state


def handle_batch_assessment(
    state: SessionStateData,
    risk_analyzer: RiskAnalyzer,
) -> Tuple[str, SessionStateData]:

    if not state.assessment_results:
        return (
            "No threats present in the assessment report to evaluate. Please load a threat file before proceeding.",
            state,
        )

    state.assessment_results = risk_analyzer.evaluate_list_of_threats(
        threat_list=state.assessment_results,
        general_context=state.general_context,
        rules_l=state.custom_rules_likelihood,
        rules_i=state.custom_rules_impact,
        num_iterations=state.num_iterations,
    )

    summary_lines = [
        f"### Batch Assessment Completed ({len(state.assessment_results)} threats processed)\n"
    ]

    sorted_threats = sorted(
        state.assessment_results, key=lambda x: x.risk_factor, reverse=True
    )

    for record in sorted_threats:
        icon, label = get_risk_indicator(record.risk_factor)
        summary_lines.append(
            f"- {icon} **{record.threat}** | Likelihood: {record.likelihood}/5 | Impact: {record.impact}/5 | **Risk Factor: {record.risk_factor}/25 ({label})**"
        )
    return "\n".join(summary_lines), state


def handle_reevaluate_filtered(
    state: SessionStateData,
    risk_analyzer: RiskAnalyzer,
    entities: Dict[str, Any],
    user_input: str = "",
    chain_factory: Optional[LLMChainFactory] = None,
) -> Tuple[str, SessionStateData]:

    if not state.assessment_results: # Controllo che la lista non sia vuota
        return (
            "No threats present in the assessment report to evaluate. Please load a threat file before proceeding.",
            state,
        )

    extracted_rules: Dict[str, Any] = {} 
    if chain_factory and user_input:
        try:
            llm_json = chain_factory.get_llm(json_format=True) # Prendo la catena 
            extractor_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", prompt.REEVALUATE_EXTRACTOR_PROMPT), 
                    ("human", "{input}"),
                ]
            )
            chain = extractor_prompt | llm_json | JsonOutputParser()
            res = chain.invoke({"input": user_input, "entities": entities}) # La catena contiene filter_keyword, additional_rule_l/i general_observation
            if isinstance(res, dict):
                extracted_rules = res # Estraggo i valori dal Json
        except Exception as e:
            print(f"Errore estrazione regole reevaluate: {e}")

    def _clean_opt(val: Any) -> Optional[str]:
        if isinstance(val, str) and val.strip() and val.strip().lower() != "null":
            return val.strip()
        return None

    extracted_filter_kw = _clean_opt(extracted_rules.get("filter_keyword")) # inserisco i valori
    add_rule_l = _clean_opt(extracted_rules.get("additional_rule_l")) # inserisco i valori
    add_rule_i = _clean_opt(extracted_rules.get("additional_rule_i")) # inserisco i valori
    gen_obs = _clean_opt(extracted_rules.get("general_observation")) # inserisco i valori

    filter_value = ( # filtro estratto dal sistema di riconoscitore di intenti
        entities.get("filter_value")
        or entities.get("affected_assets")
        or entities.get("target_filter")
        or extracted_filter_kw
        or ""
    )

    filter_condition = entities.get("filter_condition") or "" # filtra i record in base ai valori numerici

    matched_records: List[ThreatRecord] = [] # Lista dove andranno i record filtrati

    if filter_value: # Filtro per parole chiave
        keywords = [
            k.strip().lower()
            for k in filter_value.replace(",", " ").split()
            if len(k.strip()) > 2
        ]

        matched_records = [
            r
            for r in state.assessment_results
            if any(
                k in r.affected_assets.lower() or k in r.threat.lower()
                for k in keywords
            )
        ]


    if not matched_records and filter_condition: # Filtro per valori numerici
        cond_lower = filter_condition.lower()

        digits = [int(s) for s in cond_lower.split() if s.isdigit()]
        thresh = digits[0] if digits else 3

        if "impact" in cond_lower:
            if ">" in cond_lower:
                matched_records = [
                    r for r in state.assessment_results if r.impact > thresh
                ]
            elif "<" in cond_lower:
                matched_records = [
                    r for r in state.assessment_results if r.impact < thresh
                ]
            else:
                matched_records = [
                    r for r in state.assessment_results if r.impact >= thresh
                ]

        elif "likelihood" in cond_lower:
            if ">" in cond_lower:
                matched_records = [
                    r for r in state.assessment_results if r.likelihood > thresh
                ]

            elif "<" in cond_lower:
                matched_records = [
                    r for r in state.assessment_results if r.likelihood < thresh
                ]

            else:
                matched_records = [
                    r for r in state.assessment_results if r.likelihood >= thresh
                ]


    if not matched_records and user_input: # Fallback in caso non ci siano record filtrati
        for r in state.assessment_results:
            if (
                r.threat.lower() in user_input.lower()
                or r.affected_assets.lower() in user_input.lower()
            ):
                matched_records.append(r)

    if not matched_records: # Se non ci sono record filtrati, prendi tutti i record forse da togliere 
        matched_records = state.assessment_results

    temp_general_context = (
        f"{state.general_context}\n{gen_obs}".strip() # Mette il general context insieme ai commenti 
        if gen_obs
        else state.general_context
    )
    temp_rules_l = (
        f"{state.custom_rules_likelihood}\n{add_rule_l}".strip() # Mette le regole di verosimiglianza insieme ai commenti
        if add_rule_l
        else state.custom_rules_likelihood
    )
    temp_rules_i = (
        f"{state.custom_rules_impact}\n{add_rule_i}".strip() # Mette le regole di impatto insieme ai commenti
        if add_rule_i
        else state.custom_rules_impact
    )

    updated_subset = risk_analyzer.evaluate_list_of_threats( # Esegue la rivalutazione del batch
        threat_list=matched_records,
        general_context=temp_general_context,
        rules_l=temp_rules_l,
        rules_i=temp_rules_i,
        num_iterations=state.num_iterations,
    )

    subset_map = {r.id: r for r in updated_subset} # mi salvo i record rivalutati
    state.assessment_results = [
        subset_map.get(r.id, r) for r in state.assessment_results # metto i nuovi record insieme a quelli non rivalutati
    ]

    summary_lines = [
        f"### Re-evaluation Completed ({len(updated_subset)} threats re-assessed)\n"
    ]

    for record in updated_subset:
        icon, label = get_risk_indicator(record.risk_factor)
        summary_lines.append(
            f"- {icon} **{record.threat}** | Likelihood: {record.likelihood}/5 | Impact: {record.impact}/5 | **Risk Factor: {record.risk_factor}/25 ({label})**"
        )
    return "\n".join(summary_lines), state


def handle_asset_extract(
    state: SessionStateData,
    asset_extractor: AssetExtractor,
) -> Tuple[str, SessionStateData]:
    try:
        res = asset_extractor.extract_assets_pipeline(
            extra_text=state.general_context,
            general_context=state.general_context,
        )
    except Exception as e:
        return f"An error occurred during asset extraction: {e}", state

    if res.candidate_terms_count == 0:
        return (
            "No system documents or general context found to extract assets from. "
            "Please add system description files in `system_documents/` or fill in the General Context.",
            state,
        )

    if not res.final_assets:
        return (
            f"Processed **{res.candidate_terms_count}** candidate terms across **{res.clusters_after_merge}** "
            f"semantic clusters, but no protectable assets were selected.",
            state,
        )

    # Unisce i nuovi asset estratti (con descrizione) alla lista globale evitando duplicati case-insensitive
    existing_map = {a.name.lower(): a for a in state.extracted_assets}
    for item in res.described_assets:
        key = item.name.lower() 
        if key not in existing_map:
            state.extracted_assets.append(item)
            existing_map[key] = item
        elif (
            existing_map[key].description in ("", "Descrizione non disponibile.")
            and item.description
        ):
            existing_map[key].description = item.description

    assets_bullets = "\n".join(
        f"- **{item.name}**: {item.description}" for item in res.described_assets
    )
    report = (
        f"### Asset Extraction Completed\n\n"
        f"**Pipeline Summary:**\n"
        f"- 📝 **Candidate Nouns (spaCy):** {res.candidate_terms_count}\n"
        f"- 🎯 **Optimal K-Means Clusters:** {res.optimal_k} *(Silhouette Score: {res.best_silhouette_score:.3f})*\n"
        f"- 🔗 **Clusters after Cohesion & Merge:** {res.clusters_after_merge} *(Selected by LLM: {len(res.selected_clusters)})*\n"
        f"- 🛡️ **Unique Protectable Assets Identified:** {len(res.described_assets)}\n\n"
        f"---\n\n"
        f"**Extracted Assets:**\n"
        f"{assets_bullets}"
    )
    return report, state


def handle_discuss_modify(
    user_input: str,
    state: SessionStateData,
    chain_factory: LLMChainFactory,
    entities: Dict[str, Any],
    risk_analyzer: Optional[RiskAnalyzer] = None,
) -> Tuple[str, SessionStateData]:

    active_id = state.active_threat_id # ID del threat attivo
    target_record: Optional[ThreatRecord] = None

    if active_id: # Se non c'è, significa che siamo appena entrati nella funzione
        target_record = next(
            (r for r in state.assessment_results if r.id == active_id), None
        )

    if not target_record: # Altrimenti cerco il threat in base a quello che mi hanno passato 
        if not state.assessment_results: # Fallback nel caso la lista sia vuota
            return (
                "No evaluated threats are present in the assessment report. Please load or evaluate threats first before discussing them.",
                state,
            )

        search_term = ( # Prendo quello che mi hanno passato nei campi (l'LLM potrebbe non per forza scrivere sempre il target correttamente, è una misura di sicurezza in più)
            entities.get("threat_name")
            or entities.get("threat")
            or entities.get("target_threat")
            or entities.get("target_record")
            or ""
        )
        clean_term = (
            search_term.strip(" '\".,;:!?").lower()
            if isinstance(search_term, str)
            else ""
        )
       
        if clean_term: # controllo se c'è il termine da cercare
            target_record = next(
                (
                    r
                    for r in state.assessment_results #cerco il termine nella minaccia o viceversa
                    if clean_term in r.threat.lower() or r.threat.lower() in clean_term
                ),
                None,
            )


            if not target_record: # Prima non ho trovato nulla
                sig_words = [
                    w
                    for w in clean_term.split() # splitto l'input
                    if len(w) > 3
                    and w not in ["threat", "attack", "risk", "minaccia", "attacco"]
                ]

                if sig_words: #cerco la singola parola
                    target_record = next(
                        (
                            r
                            for r in state.assessment_results
                            if any(w in r.threat.lower() for w in sig_words)
                        ),
                        None,
                    )



        if not target_record: # Ancora nulla 
            user_input_lower = user_input.lower() # Controllo l'input utente
            for r in state.assessment_results:
                if r.threat.lower() in user_input_lower:
                    target_record = r
                    break

            if not target_record:
                for r in state.assessment_results:
                    sig_words = [
                        w
                        for w in r.threat.lower().split()
                        if len(w) > 3
                        and w not in ["threat", "attack", "risk", "minaccia", "attacco"]
                    ]
                    if sig_words and any(w in user_input_lower for w in sig_words):
                        target_record = r
                        break

        if not target_record: # Il bot si arrende 
            return (
                f"The requested threat is not present in the assessment report.",
                state,
            )
        # Arrivato a questo punto, abbiamo una minaccia
        state.active_threat_id = target_record.id
        state.bot_state = BotStateEnum.DISCUSSING_THREAT

    icon, label = get_risk_indicator(target_record.risk_factor)

    llm_json = chain_factory.get_llm(json_format=True)

    prompt_tmpl = ChatPromptTemplate.from_messages(
        [
            ("system", prompt.DISCUSS_MODIFY_PROMPT),
            ("human", "{input}"),
        ]
    )

    chain = prompt_tmpl | llm_json | JsonOutputParser()

    parsed_res = {}

    try:
        res = chain.invoke( # Passo alla chain lo stato attuale
            {
                "threat": target_record.threat,
                "affected_assets": target_record.affected_assets,
                "security_measures": target_record.security_measures,
                "preventive_measures": target_record.preventive_measures,
                "mitigative_measures": target_record.mitigative_measures,
                "likelihood": target_record.likelihood,
                "likelihood_reasoning": target_record.likelihood_reasoning,
                "impact": target_record.impact,
                "impact_reasoning": target_record.impact_reasoning,
                "risk_factor": target_record.risk_factor,
                "risk_label": label,
                "input": user_input,
            }
        )

        if isinstance(res, dict):
            parsed_res = res

    except Exception as e:
        print(f"Errore discuss_modify: {e}")
    # Estrae dalla richiesta i dati "nuovi"
    updated_l = parsed_res.get("updated_likelihood")
    updated_i = parsed_res.get("updated_impact")
    updated_l_reason = parsed_res.get("updated_likelihood_reasoning")
    updated_i_reason = parsed_res.get("updated_impact_reasoning")
    is_finished = parsed_res.get("is_finished", False)
    response_text = parsed_res.get("response_text", "")

    def _clean_rule(val: Any) -> Optional[str]:
        if isinstance(val, str) and val.strip() and val.strip().lower() != "null":
            return val.strip()
        return None

    add_rule_l = _clean_rule(parsed_res.get("additional_rule_l"))
    add_rule_i = _clean_rule(parsed_res.get("additional_rule_i"))
    gen_obs = _clean_rule(parsed_res.get("general_observation"))

    if ( # gestisco i vari casi di rianalisi
        risk_analyzer is not None
        and (add_rule_l or add_rule_i or gen_obs)
        and updated_l is None
        and updated_i is None
    ):
        temp_general_context = (
            f"{state.general_context}\n{gen_obs}".strip()
            if gen_obs
            else state.general_context
        )
        temp_rules_l = (
            f"{state.custom_rules_likelihood}\n{add_rule_l}".strip()
            if add_rule_l
            else state.custom_rules_likelihood
        )
        temp_rules_i = (
            f"{state.custom_rules_impact}\n{add_rule_i}".strip()
            if add_rule_i
            else state.custom_rules_impact
        )
        risk_analyzer.evaluate_single_threat(
            record=target_record,
            general_context=temp_general_context,
            rules_l=temp_rules_l,
            rules_i=temp_rules_i,
            num_iterations=state.num_iterations,
        )
        response_text = ""
    # Salvataggio nuove info
    if updated_l is None and entities.get("target_parameter") == "likelihood":
        val = entities.get("suggested_score")
        if isinstance(val, int) and 1 <= val <= 5:
            updated_l = val

    if updated_i is None and entities.get("target_parameter") == "impact":
        val = entities.get("suggested_score")
        if isinstance(val, int) and 1 <= val <= 5:
            updated_i = val

    if isinstance(updated_l, int) and 1 <= updated_l <= 5:
        target_record.likelihood = updated_l

    if isinstance(updated_i, int) and 1 <= updated_i <= 5:
        target_record.impact = updated_i

    target_record.risk_factor = target_record.likelihood * target_record.impact

    if isinstance(updated_l_reason, str) and updated_l_reason.strip():
        target_record.likelihood_reasoning = updated_l_reason.strip()
    if isinstance(updated_i_reason, str) and updated_i_reason.strip():
        target_record.impact_reasoning = updated_i_reason.strip()
    # Chiusura
    if is_finished:
        state.active_threat_id = None
        state.bot_state = BotStateEnum.IDLE
        if not response_text:
            response_text = f"The modifications for **{target_record.threat}** have been saved. Discussion closed."

    if not response_text:
        new_icon, new_label = get_risk_indicator(target_record.risk_factor)
        response_text = (
            f"### Assessment for: **{target_record.threat}**\n\n"
            f"- **Affected Assets:** {target_record.affected_assets}\n"
            f"- **Security Measures:** {target_record.security_measures}\n"
            f"- 🛡️ **Preventive Measures:** {target_record.preventive_measures}\n"
            f"- 🩹 **Mitigative Measures:** {target_record.mitigative_measures}\n"
            f"- 📈 **Likelihood Score:** {target_record.likelihood}/5\n"
            f"  - *Reasoning:* {target_record.likelihood_reasoning}\n"
            f"- 💥 **Impact Score:** {target_record.impact}/5\n"
            f"  - *Reasoning:* {target_record.impact_reasoning}\n\n"
            f"**Risk Factor:** {new_icon} {target_record.risk_factor}/25 ({new_label})\n\n"
            f"Would you like to make any modifications or ask further questions about this analysis?"
        )

    return response_text, state
