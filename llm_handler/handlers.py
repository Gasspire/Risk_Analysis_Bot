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
            f"Please specify the following missing parameter(s): **{missing_str}**.\n"
            f"*(Type `cancel` to abort and exit.)*",
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

    if not state.assessment_results:  # Controllo che la lista non sia vuota
        return (
            "No threats present in the assessment report to evaluate. Please load a threat file before proceeding.",
            state,
        )

    extracted_rules: Dict[str, Any] = {}
    if chain_factory and user_input:
        try:
            llm_json = chain_factory.get_llm(json_format=True)  # Prendo la catena
            extractor_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", prompt.REEVALUATE_EXTRACTOR_PROMPT),
                    ("human", "{input}"),
                ]
            )
            chain = extractor_prompt | llm_json | JsonOutputParser()
            res = chain.invoke(
                {"input": user_input, "entities": entities}
            )  # La catena contiene filter_keyword, additional_rule_l/i general_observation
            if isinstance(res, dict):
                extracted_rules = res  # Estraggo i valori dal Json
        except Exception as e:
            print(f"Errore estrazione regole reevaluate: {e}")

    def _clean_opt(val: Any) -> Optional[str]:
        if isinstance(val, str) and val.strip() and val.strip().lower() != "null":
            return val.strip()
        return None

    extracted_filter_kw = _clean_opt(
        extracted_rules.get("filter_keyword")
    )  # inserisco i valori
    add_rule_l = _clean_opt(
        extracted_rules.get("additional_rule_l")
    )  # inserisco i valori
    add_rule_i = _clean_opt(
        extracted_rules.get("additional_rule_i")
    )  # inserisco i valori
    gen_obs = _clean_opt(
        extracted_rules.get("general_observation")
    )  # inserisco i valori

    filter_value = (  # filtro estratto dal sistema di riconoscitore di intenti
        entities.get("filter_value")
        or entities.get("affected_assets")
        or entities.get("target_filter")
        or extracted_filter_kw
        or ""
    )

    filter_condition = (
        entities.get("filter_condition") or ""
    )  # filtra i record in base ai valori numerici

    matched_records: List[ThreatRecord] = []  # Lista dove andranno i record filtrati

    if filter_value:  # Filtro per parole chiave
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

    if not matched_records and filter_condition:  # Filtro per valori numerici
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

    if (
        not matched_records and user_input
    ):  # Fallback in caso non ci siano record filtrati
        for r in state.assessment_results:
            if (
                r.threat.lower() in user_input.lower()
                or r.affected_assets.lower() in user_input.lower()
            ):
                matched_records.append(r)

    if (
        not matched_records
    ):  # Se non ci sono record filtrati, prendi tutti i record forse da togliere
        matched_records = state.assessment_results

    temp_general_context = (
        f"{state.general_context}\n{gen_obs}".strip()  # Mette il general context insieme ai commenti
        if gen_obs
        else state.general_context
    )
    temp_rules_l = (
        f"{state.custom_rules_likelihood}\n{add_rule_l}".strip()  # Mette le regole di verosimiglianza insieme ai commenti
        if add_rule_l
        else state.custom_rules_likelihood
    )
    temp_rules_i = (
        f"{state.custom_rules_impact}\n{add_rule_i}".strip()  # Mette le regole di impatto insieme ai commenti
        if add_rule_i
        else state.custom_rules_impact
    )

    updated_subset = (
        risk_analyzer.evaluate_list_of_threats(  # Esegue la rivalutazione del batch
            threat_list=matched_records,
            general_context=temp_general_context,
            rules_l=temp_rules_l,
            rules_i=temp_rules_i,
            num_iterations=state.num_iterations,
        )
    )

    subset_map = {r.id: r for r in updated_subset}  # mi salvo i record rivalutati
    state.assessment_results = [
        subset_map.get(r.id, r)
        for r in state.assessment_results  # metto i nuovi record insieme a quelli non rivalutati
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


_THREAT_STOP_WORDS = {"threat", "attack", "risk", "minaccia", "attacco"}

_CLOSE_PHRASES = {
    "chiudi",
    "chiudi discussione",
    "conferma",
    "confermo",
    "confirm",
    "salva",
    "salva e chiudi",
    "fine",
    "fatto",
    "basta così",
    "basta cosi",
    "va bene così",
    "va bene cosi",
    "perfetto così",
    "perfetto cosi",
    "close",
    "close discussion",
    "save",
    "save and close",
    "done",
    "finish",
    "finished",
    "that's all",
    "thats all",
    "no further changes",
}


def _is_direct_close_command(user_input: str) -> bool:
    normalized = user_input.strip().lstrip("/").rstrip(".,;:!?").strip().lower()
    return normalized in _CLOSE_PHRASES


def _dedupe_records(records: List[ThreatRecord]) -> List[ThreatRecord]:
    seen = set()
    unique: List[ThreatRecord] = []
    for rec in records:
        if rec.id not in seen:
            seen.add(rec.id)
            unique.append(rec)
    return unique


def _find_matching_threats(
    user_input: str,
    entities: Dict[str, Any],
    assessment_results: List[ThreatRecord],
) -> List[ThreatRecord]:
    """Trova tutti i ThreatRecord che corrispondono alla ricerca dell'utente."""
    search_term = (
        entities.get("threat_name")
        or entities.get("threat")
        or entities.get("target_threat")
        or entities.get("target_record")
        or ""
    )
    clean_term = (
        search_term.strip(" '\".,;:!?").lower() if isinstance(search_term, str) else ""
    )

    if clean_term:
        exact_matches = [
            r for r in assessment_results if r.threat.lower() == clean_term
        ]
        if exact_matches:
            return _dedupe_records(exact_matches)

        substr_matches = [
            r
            for r in assessment_results
            if clean_term in r.threat.lower() or r.threat.lower() in clean_term
        ]
        if substr_matches:
            return _dedupe_records(substr_matches)

        sig_words = [
            w for w in clean_term.split() if len(w) > 3 and w not in _THREAT_STOP_WORDS
        ]
        if sig_words:
            word_matches = [
                r
                for r in assessment_results
                if any(w in r.threat.lower() for w in sig_words)
            ]
            if word_matches:
                return _dedupe_records(word_matches)

    user_input_lower = user_input.lower()
    input_substr_matches = [
        r for r in assessment_results if r.threat.lower() in user_input_lower
    ]
    if input_substr_matches:
        return _dedupe_records(input_substr_matches)

    input_word_matches = []
    for r in assessment_results:
        sig_words = [
            w
            for w in r.threat.lower().split()
            if len(w) > 3 and w not in _THREAT_STOP_WORDS
        ]
        if sig_words and any(w in user_input_lower for w in sig_words):
            input_word_matches.append(r)

    return _dedupe_records(input_word_matches)


def _resolve_candidate_selection(
    user_input: str, candidates: List[ThreatRecord]
) -> Optional[ThreatRecord]:
    """Risolve la scelta dell'utente tra più minacce candidate (per indice o nome)."""
    cleaned = user_input.strip().lstrip("#").rstrip(".,;:!?").strip()
    if cleaned.isdigit():
        idx = int(cleaned) - 1
        if 0 <= idx < len(candidates):
            return candidates[idx]

    lowered = cleaned.lower()
    if not lowered:
        return None

    exact = [c for c in candidates if c.threat.lower() == lowered]
    if len(exact) == 1:
        return exact[0]

    substr = [
        c
        for c in candidates
        if lowered in c.threat.lower() or c.threat.lower() in lowered
    ]
    if len(substr) == 1:
        return substr[0]

    return None


def _format_candidate_selection_prompt(
    candidates: List[ThreatRecord], invalid_choice: bool = False
) -> str:
    header = (
        "Invalid selection. Please reply with the number or name of one of the matching threats:\n\n"
        if invalid_choice
        else "Multiple threats match your request. Please select which one you want to discuss:\n\n"
    )
    lines = [
        f"{idx}. **{rec.threat}** (Assets: {rec.affected_assets})"
        for idx, rec in enumerate(candidates, start=1)
    ]
    return (
        header
        + "\n".join(lines)
        + "\n\n*(Reply with the number or threat name, or type `annulla` / `cancel` to abort.)*"
    )


def _format_threat_state(
    record: ThreatRecord, title_prefix: str = "Assessment for"
) -> str:
    """Formatta lo stato completo di una minaccia in Markdown."""
    icon, label = get_risk_indicator(record.risk_factor)
    return (
        f"### {title_prefix}: **{record.threat}**\n\n"
        f"- **Affected Assets:** {record.affected_assets}\n"
        f"- **Security Measures:** {record.security_measures}\n"
        f"- 🛡️ **Preventive Measures:** {record.preventive_measures}\n"
        f"- 🩹 **Mitigative Measures:** {record.mitigative_measures}\n"
        f"- 📈 **Likelihood Score:** {record.likelihood}/5\n"
        f"  - *Reasoning:* {record.likelihood_reasoning}\n"
        f"- 💥 **Impact Score:** {record.impact}/5\n"
        f"  - *Reasoning:* {record.impact_reasoning}\n\n"
        f"**Risk Factor:** {icon} {record.risk_factor}/25 ({label})"
    )


def _format_final_discussion_summary(
    record: ThreatRecord,
    backup: Optional[ThreatRecord],
    explanation: str = "",
) -> str:
    """Formatta il messaggio finale con il nuovo stato della minaccia e il riepilogo delle modifiche."""
    state_block = _format_threat_state(record, title_prefix="Updated Assessment for")

    changes: List[str] = []
    if backup is not None:
        field_diffs = [
            ("Affected Assets", backup.affected_assets, record.affected_assets),
            ("Security Measures", backup.security_measures, record.security_measures),
            (
                "Preventive Measures",
                backup.preventive_measures,
                record.preventive_measures,
            ),
            (
                "Mitigative Measures",
                backup.mitigative_measures,
                record.mitigative_measures,
            ),
            (
                "Likelihood Score",
                f"{backup.likelihood}/5",
                f"{record.likelihood}/5",
            ),
            (
                "Likelihood Reasoning",
                backup.likelihood_reasoning,
                record.likelihood_reasoning,
            ),
            (
                "Impact Score",
                f"{backup.impact}/5",
                f"{record.impact}/5",
            ),
            (
                "Impact Reasoning",
                backup.impact_reasoning,
                record.impact_reasoning,
            ),
            (
                "Risk Factor",
                f"{backup.risk_factor}/25",
                f"{record.risk_factor}/25",
            ),
        ]
        for label, old_val, new_val in field_diffs:
            if old_val != new_val:
                changes.append(f"- **{label}:** `{old_val}` → `{new_val}`")

    changes_block = (
        "\n".join(changes)
        if changes
        else "- No modifications were made to the threat assessment."
    )

    parts = [
        state_block,
        f"### Modifications Applied\n\n{changes_block}",
    ]
    if explanation and explanation.strip():
        parts.append(f"**Explanation:** {explanation.strip()}")
    parts.append(
        f"The modifications for **{record.threat}** have been saved. Discussion closed."
    )
    return "\n\n".join(parts)


def handle_discuss_modify(
    user_input: str,
    state: SessionStateData,
    chain_factory: LLMChainFactory,
    entities: Dict[str, Any],
    risk_analyzer: Optional[RiskAnalyzer] = None,
) -> Tuple[str, SessionStateData]:

    active_id = state.active_threat_id  # ID del threat attivo
    entered_new_discussion = False
    target_record: Optional[ThreatRecord] = None
    effective_user_input = user_input

    if active_id:  # Se non c'è, significa che siamo appena entrati nella funzione
        target_record = next(
            (r for r in state.assessment_results if r.id == active_id), None
        )

    if not target_record and state.pending_discussion_candidates:
        candidates = [
            r
            for r in state.assessment_results
            if r.id in state.pending_discussion_candidates
        ]
        if not candidates:
            state.pending_discussion_candidates = None
            state.pending_discussion_input = None
            state.bot_state = BotStateEnum.IDLE
            return (
                "The matching threats are no longer available in the assessment report.",
                state,
            )

        selected = _resolve_candidate_selection(user_input, candidates)
        if not selected:
            return (
                _format_candidate_selection_prompt(candidates, invalid_choice=True),
                state,
            )

        target_record = selected
        if state.pending_discussion_input:
            effective_user_input = state.pending_discussion_input
        state.pending_discussion_candidates = None
        state.pending_discussion_input = None
        state.active_threat_id = target_record.id
        state.active_threat_backup = target_record.model_copy(deep=True)
        state.bot_state = BotStateEnum.DISCUSSING_THREAT
        entered_new_discussion = True

    if (
        not target_record
    ):  # Altrimenti cerco il threat in base a quello che mi hanno passato
        if not state.assessment_results:  # Fallback nel caso la lista sia vuota
            return (
                "No evaluated threats are present in the assessment report. Please load or evaluate threats first before discussing them.",
                state,
            )

        matches = _find_matching_threats(
            user_input=user_input,
            entities=entities,
            assessment_results=state.assessment_results,
        )

        if not matches:  # Il bot si arrende
            return (
                "The requested threat is not present in the assessment report.",
                state,
            )

        if len(matches) > 1:
            state.pending_discussion_candidates = [r.id for r in matches]
            state.pending_discussion_input = user_input
            state.bot_state = BotStateEnum.AWAITING_THREAT_SELECTION
            return _format_candidate_selection_prompt(matches), state

        target_record = matches[0]
        # Arrivato a questo punto, abbiamo una minaccia
        state.active_threat_id = target_record.id
        state.active_threat_backup = target_record.model_copy(deep=True)
        state.bot_state = BotStateEnum.DISCUSSING_THREAT
        entered_new_discussion = True

    icon, label = get_risk_indicator(target_record.risk_factor)

    if not entered_new_discussion and _is_direct_close_command(effective_user_input):
        backup_record = state.active_threat_backup
        state.active_threat_id = None
        state.active_threat_backup = None
        state.bot_state = BotStateEnum.IDLE
        return (
            _format_final_discussion_summary(
                record=target_record,
                backup=backup_record,
            ),
            state,
        )

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
        res = chain.invoke(  # Passo alla chain lo stato attuale
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
                "input": effective_user_input,
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
    show_current_state = bool(parsed_res.get("show_current_state", False))
    keep_l = bool(parsed_res.get("keep_likelihood_unchanged", False))
    keep_i = bool(parsed_res.get("keep_impact_unchanged", False))
    response_text = parsed_res.get("response_text", "")

    def _clean_rule(val: Any) -> Optional[str]:
        if isinstance(val, str) and val.strip() and val.strip().lower() != "null":
            return val.strip()
        return None

    add_rule_l = _clean_rule(parsed_res.get("additional_rule_l"))
    add_rule_i = _clean_rule(parsed_res.get("additional_rule_i"))
    gen_obs = _clean_rule(parsed_res.get("general_observation"))

    new_assets = _clean_rule(parsed_res.get("updated_affected_assets"))
    new_sec_measures = _clean_rule(parsed_res.get("updated_security_measures"))
    new_prev_measures = _clean_rule(parsed_res.get("updated_preventive_measures"))
    new_mit_measures = _clean_rule(parsed_res.get("updated_mitigative_measures"))

    assets_changed = False
    all_measures_changed = False
    prev_changed = False
    mit_changed = False

    if new_assets is not None and new_assets != target_record.affected_assets:
        target_record.affected_assets = new_assets
        assets_changed = True

    if (
        new_sec_measures is not None
        and new_sec_measures != target_record.security_measures
    ):
        target_record.security_measures = new_sec_measures
        all_measures_changed = True

    if not all_measures_changed:
        if (
            new_prev_measures is not None
            and new_prev_measures != target_record.preventive_measures
        ):
            target_record.preventive_measures = new_prev_measures
            prev_changed = True

        if (
            new_mit_measures is not None
            and new_mit_measures != target_record.mitigative_measures
        ):
            target_record.mitigative_measures = new_mit_measures
            mit_changed = True

    specific_measures_changed = prev_changed or mit_changed
    needs_reeval = (
        assets_changed
        or all_measures_changed
        or specific_measures_changed
        or bool(add_rule_l or add_rule_i or gen_obs)
    )

    if risk_analyzer is not None and needs_reeval:
        reclassify = all_measures_changed and not specific_measures_changed
        if assets_changed or all_measures_changed or gen_obs:
            eval_l = (updated_l is None) and (not keep_l)
            eval_i = (updated_i is None) and (not keep_i)
        else:
            eval_l = (
                (prev_changed or bool(add_rule_l))
                and (updated_l is None)
                and (not keep_l)
            )
            eval_i = (
                (mit_changed or bool(add_rule_i))
                and (updated_i is None)
                and (not keep_i)
            )

        if eval_l or eval_i:
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
                reclassify_measures=reclassify,
                eval_likelihood=eval_l,
                eval_impact=eval_i,
            )
            if not is_finished:
                response_text = ""
        elif reclassify:
            prev_str, mit_str = risk_analyzer.classify_security_measures(
                target_record.threat,
                target_record.affected_assets,
                target_record.security_measures,
            )
            target_record.preventive_measures = prev_str
            target_record.mitigative_measures = mit_str
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
        backup_record = state.active_threat_backup
        state.active_threat_id = None
        state.active_threat_backup = None
        state.bot_state = BotStateEnum.IDLE
        response_text = _format_final_discussion_summary(
            record=target_record,
            backup=backup_record,
            explanation=response_text,
        )

    if entered_new_discussion and not is_finished:
        state_block = _format_threat_state(
            target_record, title_prefix="Current Assessment for"
        )
        follow_up = (
            response_text.strip()
            if response_text and response_text.strip()
            else "Would you like to make any modifications or ask further questions about this analysis?"
        )
        response_text = (
            f"{state_block}\n\n"
            f"{follow_up}\n\n"
            f"*(Type `confirm`/`conferma` to save and exit, or `annulla`/`cancel` to discard changes.)*"
        )
    elif show_current_state and not is_finished:
        state_block = _format_threat_state(
            target_record, title_prefix="Current Assessment for"
        )
        response_text = (
            f"{state_block}\n\n"
            f"Would you like to make any modifications or ask further questions about this analysis?\n\n"
            f"*(Type `confirm`/`conferma` to save and exit, or `annulla`/`cancel` to discard changes.)*"
        )
    elif not response_text:
        state_block = _format_threat_state(target_record)
        response_text = (
            f"{state_block}\n\n"
            f"Would you like to make any modifications or ask further questions about this analysis?\n\n"
            f"*(Type `confirm`/`conferma` to save and exit, or `annulla`/`cancel` to discard changes.)*"
        )

    return response_text, state
