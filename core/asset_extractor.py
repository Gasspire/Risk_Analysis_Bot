import os
import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any, cast
import numpy as np
import spacy
from spacy.language import Language
from langchain_ollama import OllamaEmbeddings
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

from config import ASSET_DOCS_DIR, BASE_URL, EMBEDDING_MODEL
from core.llm_chains import LLMChainFactory
from models import (
    AssetExtractionOutput,
    ClusterAssetEvaluation,
    ExtractedAssetRecord,
    AssetDescriptionsOutput,
)


@dataclass
class SemanticCluster:
    """Rappresenta un cluster semantico di termini candidati con i relativi embedding e centroide."""
    cluster_id: int
    terms: List[str]
    embeddings: np.ndarray  # shape: (n_terms, dim)
    centroid: np.ndarray    # shape: (dim,)
    cohesion_score: float = 1.0


@dataclass
class AssetExtractionPipelineResult:
    """Contiene i risultati completi e le metriche intermedie dell'intera pipeline (Step 1-8 + Descrizioni)."""
    candidate_terms_count: int = 0
    optimal_k: int = 0
    best_silhouette_score: float = 0.0
    clusters_after_cohesion: int = 0
    clusters_after_merge: int = 0
    selected_clusters: List[ClusterAssetEvaluation] = field(default_factory=list)
    final_assets: List[str] = field(default_factory=list)
    described_assets: List[ExtractedAssetRecord] = field(default_factory=list)


class AssetExtractor:
    """
    Gestore della pipeline di estrazione degli asset dai documenti di sistema.
    Caricamento documenti e estrazione linguistica dei sostantivi tramite spaCy.
    """

    def __init__(
        self,
        chain_factory: Optional[LLMChainFactory] = None,
        docs_dir: str = ASSET_DOCS_DIR,
        spacy_model: str = "en_core_web_sm",
        embedding_model: str = EMBEDDING_MODEL,
        base_url: str = BASE_URL,
    ):
        self.chain_factory = chain_factory or LLMChainFactory(base_url=base_url)
        self.docs_dir = docs_dir
        self.spacy_model = spacy_model
        self._nlp: Optional[Language] = None
        self.embedder = OllamaEmbeddings(model=embedding_model, base_url=base_url)

        if not os.path.exists(self.docs_dir):
            os.makedirs(self.docs_dir, exist_ok=True)

    def _get_nlp(self) -> Language:
        """Carica pigramente (lazy load) il modello spaCy inglese."""
        if self._nlp is None:
            try:
                self._nlp = spacy.load(self.spacy_model)
            except OSError as e:
                raise RuntimeError(
                    f"Modello spaCy '{self.spacy_model}' non trovato. "
                    f"Esegui: python -m spacy download {self.spacy_model}"
                ) from e
        return self._nlp

    def save_uploaded_documents(self, text_sources: List[Dict[str, str]]) -> int:
        """
        Salva i documenti caricati (es. da UI) nella cartella docs_dir (system_documents).
        Formato di ciascun elemento: {"content": str, "source_name": str}
        Restituisce il numero di file salvati.
        """
        if not text_sources:
            return 0

        os.makedirs(self.docs_dir, exist_ok=True)
        saved_count = 0

        for src in text_sources:
            content = src.get("content", "").strip()
            raw_name = src.get("source_name", "document.txt")
            safe_name = os.path.basename(raw_name)
            if not safe_name.lower().endswith((".txt", ".md")):
                safe_name += ".txt"

            if content:
                target_path = os.path.join(self.docs_dir, safe_name)
                with open(target_path, "w", encoding="utf-8") as f:
                    f.write(content)
                saved_count += 1

        return saved_count

    def list_documents(self) -> List[str]:
        """Restituisce l'elenco dei file .txt/.md presenti in docs_dir."""
        if not os.path.exists(self.docs_dir):
            return []
        return sorted(
            f for f in os.listdir(self.docs_dir)
            if f.lower().endswith((".txt", ".md")) and os.path.isfile(os.path.join(self.docs_dir, f))
        )

    def load_documents_text(self, extra_text: str = "") -> str:
        """
        Legge tutti i file .txt e .md dalla cartella dei documenti di sistema
        e li concatena con eventuale testo aggiuntivo (es. general_context).
        """
        texts: List[str] = []

        if extra_text and extra_text.strip():
            texts.append(extra_text.strip())

        if os.path.exists(self.docs_dir):
            for fname in sorted(os.listdir(self.docs_dir)):
                if fname.lower().endswith((".txt", ".md")):
                    fpath = os.path.join(self.docs_dir, fname)
                    if os.path.isfile(fpath):
                        with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                            content = f.read().strip()
                            if content:
                                texts.append(content)

        return "\n\n".join(texts)

    def extract_candidate_nouns(
        self,
        text: str,
        include_noun_chunks: bool = True,
        min_length: int = 2,
    ) -> List[str]:
        """
        Estrae i sostantivi (NOUN, PROPN) e opzionalmente i sintagmi nominali (noun_chunks)
        dal testo inglese usando spaCy, applicando lemmatizzazione, rimozione stopword
        e deduplicazione iniziale.
        """
        if not text or not text.strip(): 
            return [] # Torniamo nulla se il testo è vuoto

        nlp = self._get_nlp() 
        doc = nlp(text)# Passiamo il testo al modello

        candidates: List[str] = []
        seen: set[str] = set()

        def _add_candidate(raw_term: str):
            # Pulisce spazi multipli e caratteri non alfanumerici ai bordi
            cleaned = re.sub(r"\s+", " ", raw_term).strip(" -_.,;:!?()[]{}\"'/\\")
            cleaned_lower = cleaned.lower()
            if (
                len(cleaned_lower) >= min_length # scartiamo se è troppo corto 
                and not cleaned_lower.isdigit() # scartiamo se è un numero
                and cleaned_lower not in seen # scartiamo se lo abbiamo già visto
            ):
                seen.add(cleaned_lower) # aggiungiamo alla lista dei visti
                candidates.append(cleaned_lower) # aggiungiamo ai candidati

        def _clean_lemma(token) -> str:
            raw_text = token.text.strip().lower()
            lemma = token.lemma_.strip().lower()
            # In inglese moderno "data" è usato come sostantivo invariabile (evita il latino "datum")
            if raw_text == "data" or lemma == "datum":
                return "data"
            return lemma

        # 1. Estrazione sintagmi nominali composti (es. "infusion pump", "server room")
        chunk_words: set[str] = set()
        if include_noun_chunks:
            for chunk in doc.noun_chunks:
                raw_tokens = [
                    t for t in chunk
                    if t.pos_ in ("NOUN", "PROPN", "ADJ")
                    and not t.is_stop
                    and not t.is_punct
                    and not t.is_space
                    and not t.like_num
                ]
                # Regola grammaticale inglese: in un noun chunk composto, un modificatore iniziale
                # non può essere un sostantivo plurale (NNS, es. "Hosts clinical database", "Provides access").
                # Quando spaCy trova un NNS seguito da altri token nel chunk, è un verbo alla 3a pers. sing. (VBZ) senza soggetto.
                if len(raw_tokens) >= 2 and raw_tokens[0].tag_ == "NNS":
                    raw_tokens = raw_tokens[1:]

                content_tokens = [_clean_lemma(t) for t in raw_tokens]

                # Aggiungiamo i chunk composti (2-4 parole) che terminano con un sostantivo
                if 2 <= len(content_tokens) <= 4 and chunk.root.pos_ in ("NOUN", "PROPN"):
                    phrase = " ".join(content_tokens)
                    _add_candidate(phrase)
                    chunk_words.update(content_tokens)

        # 2. Estrazione singoli sostantivi (NOUN e PROPN)
        for token in doc:
            if (
                token.pos_ in ("NOUN", "PROPN")
                and not token.is_stop
                and not token.is_punct
                and not token.is_space
                and not token.like_num
            ):
                lemma = _clean_lemma(token)
                # Scarta sigle/abbreviazioni corte (<= 3 caratteri) che sono già parte di un sintagma composto
                if len(lemma) <= 3 and lemma in chunk_words:
                    continue
                _add_candidate(lemma)

        return candidates

    def compute_embeddings(self, terms: List[str], batch_size: int = 64) -> np.ndarray:
        """
        Converte la lista di sostantivi candidati in vettori di embedding (a blocchi/batch
        per evitare il crash del runner interno di Ollama su liste grandi)
        e li normalizza con norma L2 (così il prodotto scalare coincide con la cosine similarity).
        """
        if not terms:
            return np.empty((0, 0), dtype=np.float32)

        all_vectors: List[List[float]] = []
        for i in range(0, len(terms), batch_size):
            batch = terms[i : i + batch_size]
            batch_vectors = self.embedder.embed_documents(batch)
            all_vectors.extend(batch_vectors)

        matrix = np.asarray(all_vectors, dtype=np.float32)
        return normalize(matrix, norm="l2", axis=1)

    def cluster_kmeans_optimal(
        self,
        terms: List[str],
        embeddings: np.ndarray,
        k_min: int = 2,
        k_max: Optional[int] = None,
        random_state: int = 42,
    ) -> Tuple[List[SemanticCluster], float]:
        """
        Applica K-Means iterando su diversi valori di k per massimizzare il Silhouette Score
        (calcolato con distanza del coseno).
        Restituisce la lista di SemanticCluster ottimali e il miglior Silhouette Score ottenuto.
        """
        n_samples = len(terms)
        if n_samples == 0 or embeddings.size == 0:
            return [], 0.0

        # Caso limite: troppi pochi elementi per fare clustering (meno di 3 termini)
        if n_samples < 3:
            centroid = normalize(embeddings.mean(axis=0, keepdims=True), norm="l2")[0]
            return [
                SemanticCluster(
                    cluster_id=0,
                    terms=list(terms),
                    embeddings=embeddings,
                    centroid=centroid,
                )
            ], 1.0

        # Se k_max non è specificato, lo fissiamo in modo adattivo (es. tra 2 e max 30 o N//2)
        if k_max is None:
            k_max = min(30, max(k_min + 1, n_samples // 3))
        k_max = min(k_max, n_samples - 1)
        k_min = max(2, min(k_min, k_max))

        best_k = k_min
        best_score = -1.0
        best_labels: Optional[np.ndarray] = None

        for k in range(k_min, k_max + 1):
            kmeans = KMeans(n_clusters=k, n_init=10, random_state=random_state)
            labels = kmeans.fit_predict(embeddings)

            # Il silhouette score richiede almeno 2 cluster distinti non vuoti
            if len(set(labels)) < 2:
                continue

            score = float(silhouette_score(embeddings, labels, metric="cosine"))
            if score > best_score:
                best_score = score
                best_k = k
                best_labels = labels

        if best_labels is None:
            kmeans = KMeans(n_clusters=best_k, n_init=10, random_state=random_state)
            best_labels = kmeans.fit_predict(embeddings)

        # Costruiamo gli oggetti SemanticCluster a partire dalle etichette migliori
        clusters: List[SemanticCluster] = []
        for cid in range(best_k):
            mask = best_labels == cid
            if not np.any(mask):
                continue
            cluster_terms = [terms[i] for i in np.where(mask)[0]]
            cluster_embs = embeddings[mask]
            # Centroide normalizzato L2 (direzione media del cluster sulla sfera unitaria)
            centroid = normalize(cluster_embs.mean(axis=0, keepdims=True), norm="l2")[0]
            clusters.append(
                SemanticCluster(
                    cluster_id=len(clusters),
                    terms=cluster_terms,
                    embeddings=cluster_embs,
                    centroid=centroid,
                )
            )

        return clusters, best_score

    @staticmethod
    def _compute_internal_cohesion(embeddings: np.ndarray) -> float:
        """
        Calcola la coesione semantica interna di un cluster come la media della
        cosine similarity tra tutte le coppie distinte di elementi nel cluster.
        """
        n = embeddings.shape[0]
        if n < 2:
            return 0.0

        sim_matrix = cosine_similarity(embeddings)
        # Prendiamo solo il triangolo superiore (escludendo la diagonale dove sim == 1.0)
        triu_indices = np.triu_indices(n, k=1)
        pairwise_sims = sim_matrix[triu_indices]
        return float(np.mean(pairwise_sims)) if pairwise_sims.size > 0 else 0.0

    def filter_clusters_by_cohesion(
        self,
        clusters: List[SemanticCluster],
        min_cohesion: float = 0.48,
        min_cluster_size: int = 2,
    ) -> List[SemanticCluster]:
        """
        Valuta la coesione semantica interna di ciascun cluster tramite cosine similarity
        tra gli elementi del gruppo ed elimina i cluster che non superano la soglia
        (o che contengono meno di min_cluster_size elementi).
        """
        filtered: List[SemanticCluster] = []

        for cluster in clusters:
            if len(cluster.terms) < min_cluster_size:
                continue

            cohesion = self._compute_internal_cohesion(cluster.embeddings)
            if cohesion >= min_cohesion:
                cluster.cohesion_score = cohesion
                cluster.cluster_id = len(filtered)
                filtered.append(cluster)

        return filtered

    def merge_similar_clusters(
        self,
        clusters: List[SemanticCluster],
        max_centroid_similarity: float = 0.86,
        max_merged_terms: int = 35,
    ) -> List[SemanticCluster]:
        """
        Calcola la cosine similarity tra i centroidi dei cluster e fonde in un unico
        cluster quelli la cui similarità supera 'max_centroid_similarity'
        (evitando l'effetto valanga grazie al limite 'max_merged_terms').
        """
        if len(clusters) < 2:
            return clusters

        current = list(clusters)

        while len(current) >= 2:
            centroids = np.vstack([c.centroid for c in current])
            sim_matrix = cosine_similarity(centroids)
            # Azzera la diagonale per cercare la coppia distinta più simile
            np.fill_diagonal(sim_matrix, -1.0)

            # Escludiamo le coppie la cui fusione supererebbe max_merged_terms
            for row_i in range(len(current)):
                for col_j in range(row_i + 1, len(current)):
                    if len(current[row_i].terms) + len(current[col_j].terms) > max_merged_terms:
                        sim_matrix[row_i, col_j] = -1.0
                        sim_matrix[col_j, row_i] = -1.0

            max_idx = int(np.argmax(sim_matrix))
            i, j = divmod(max_idx, len(current))
            max_sim = float(sim_matrix[i, j])

            if max_sim < max_centroid_similarity:
                break

            # Garantiamo i < j per fondere j dentro i
            if i > j:
                i, j = j, i

            c1, c2 = current[i], current[j]
            merged_terms = c1.terms + [t for t in c2.terms if t not in c1.terms]
            merged_embs = np.vstack([c1.embeddings, c2.embeddings])
            merged_centroid = normalize(merged_embs.mean(axis=0, keepdims=True), norm="l2")[0]
            merged_cohesion = self._compute_internal_cohesion(merged_embs)

            current[i] = SemanticCluster(
                cluster_id=i,
                terms=merged_terms,
                embeddings=merged_embs,
                centroid=merged_centroid,
                cohesion_score=merged_cohesion,
            )
            current.pop(j)

        # Riassegna gli ID sequenziali 0..M-1
        for idx, c in enumerate(current):
            c.cluster_id = idx

        return current

    def select_asset_clusters_with_llm(
        self,
        clusters: List[SemanticCluster],
        general_context: str = "",
        clusters_batch_size: int = 8,
    ) -> Tuple[List[str], List[ClusterAssetEvaluation]]:
        """
        Invia i cluster semantici sopravvissuti all'LLM (in batch di 'clusters_batch_size'
        per evitare il troncamento del JSON di output), che seleziona i cluster
        contenenti asset da proteggere ed estrae i nomi canonici degli asset.
        """
        if not clusters:
            return [], []

        chain = self.chain_factory.get_asset_extraction_chain()
        raw_assets: List[str] = []
        selected_evals: List[ClusterAssetEvaluation] = []

        for i in range(0, len(clusters), clusters_batch_size):
            batch_clusters = clusters[i : i + clusters_batch_size]
            cluster_lines = [
                f"Cluster {c.cluster_id}: [{', '.join(c.terms)}]"
                for c in batch_clusters
            ]
            clusters_text = "\n".join(cluster_lines)

            try:
                res: Optional[AssetExtractionOutput] = cast(
                    Optional[AssetExtractionOutput],
                    chain.invoke({
                        "general_context": general_context or "None provided.",
                        "clusters_text": clusters_text,
                    }),
                )
            except Exception as e:
                print(f"Errore LLM su batch cluster {i}..{i + len(batch_clusters) - 1}: {e}")
                continue

            if res and res.clusters_evaluation:
                for ev in res.clusters_evaluation:
                    if ev.contains_protectable_assets and ev.extracted_assets:
                        selected_evals.append(ev)
                        raw_assets.extend(ev.extracted_assets)

        return raw_assets, selected_evals

    def deduplicate_assets(
        self,
        assets: List[str],
        semantic_threshold: float = 0.88,
    ) -> List[str]:
        """
        Rimuove eventuali duplicati dalla lista degli asset selezionati dall'LLM:
        1. Deduplicazione lessicale e lemmatizzata (case-insensitive).
        2. Deduplicazione semantica tramite cosine similarity sugli embedding.
        """
        if not assets:
            return []

        nlp = self._get_nlp()
        unique_assets: List[str] = []
        seen_lemmas: set[str] = set()

        for asset in assets:
            cleaned = re.sub(r"\s+", " ", asset).strip(" -_.,;:!?()[]{}\"'/\\")
            if not cleaned:
                continue

            doc = nlp(cleaned.lower())
            lemma_key = " ".join(
                t.lemma_.lower() for t in doc if not t.is_punct and not t.is_space and not t.is_stop
            ).strip()
            if not lemma_key:
                lemma_key = cleaned.lower()

            if lemma_key not in seen_lemmas:
                seen_lemmas.add(lemma_key)
                unique_assets.append(cleaned)

        if len(unique_assets) < 2:
            return unique_assets

        # Deduplicazione semantica con cosine similarity sugli embedding
        embs = self.compute_embeddings(unique_assets)
        sim_matrix = cosine_similarity(embs)

        final_assets: List[str] = []
        kept_indices: List[int] = []

        for idx, asset_name in enumerate(unique_assets):
            is_duplicate = False
            for kept_idx in kept_indices:
                if sim_matrix[idx, kept_idx] >= semantic_threshold:
                    is_duplicate = True
                    break
            if not is_duplicate:
                kept_indices.append(idx)
                final_assets.append(asset_name)

        return final_assets

    @staticmethod
    def _extract_relevant_snippets(
        full_text: str,
        batch_assets: List[str],
        max_chars: int = 6000,
    ) -> str:
        """
        Estrae dal testo completo i paragrafi/righe in cui compaiono gli asset del batch
        (o le parole che li compongono), fornendo all'LLM il contesto esatto del documento.
        """
        if not full_text:
            return "No document text provided."

        paragraphs = [p.strip() for p in re.split(r"\n+", full_text) if p.strip()]
        keywords: set[str] = set()
        for asset in batch_assets:
            keywords.add(asset.lower())
            for w in asset.lower().split():
                if len(w) > 3:
                    keywords.add(w)

        scored_paragraphs: List[Tuple[int, str]] = []
        for p in paragraphs:
            p_lower = p.lower()
            hits = sum(1 for kw in keywords if kw in p_lower)
            if hits > 0:
                scored_paragraphs.append((hits, p))

        scored_paragraphs.sort(key=lambda x: x[0], reverse=True)

        selected: List[str] = []
        total_len = 0
        for _, p in scored_paragraphs:
            if total_len + len(p) > max_chars:
                break
            selected.append(p)
            total_len += len(p)

        if not selected:
            return full_text[:max_chars]

        return "\n".join(selected)

    def generate_asset_descriptions(
        self,
        assets: List[str],
        full_text: str,
        general_context: str = "",
        batch_size: int = 15,
    ) -> List[ExtractedAssetRecord]:
        """
        Genera una descrizione concisa (1-2 frasi) per ciascun asset estratto,
        basandosi strettamente sui documenti di sistema forniti.
        """
        if not assets:
            return []

        chain = self.chain_factory.get_asset_description_chain()
        described_map: Dict[str, str] = {}

        for i in range(0, len(assets), batch_size):
            batch = assets[i : i + batch_size]
            snippets = self._extract_relevant_snippets(full_text, batch)
            assets_list_str = "\n".join(f"- {a}" for a in batch)

            try:
                res: Optional[AssetDescriptionsOutput] = cast(
                    Optional[AssetDescriptionsOutput],
                    chain.invoke({
                        "general_context": general_context or "None provided.",
                        "document_context": snippets,
                        "assets_list": assets_list_str,
                    }),
                )
                if res and res.assets:
                    for item in res.assets:
                        if item.name and item.description:
                            described_map[item.name.strip().lower()] = item.description.strip()
            except Exception as e:
                print(f"Errore generazione descrizioni batch {i}..{i + len(batch) - 1}: {e}")

        records: List[ExtractedAssetRecord] = []
        for asset_name in assets:
            desc = described_map.get(asset_name.lower())
            if not desc:
                # Fallback se il nome restituito dall'LLM aveva una lieve differenza
                desc = next(
                    (v for k, v in described_map.items() if asset_name.lower() in k or k in asset_name.lower()),
                    "Asset identified within the system architecture documentation.",
                )
            records.append(ExtractedAssetRecord(name=asset_name, description=desc))

        return records

    def extract_assets_pipeline(
        self,
        extra_text: str = "",
        general_context: str = "",
        k_min: int = 2,
        k_max: Optional[int] = None,
        min_cohesion: float = 0.48,
        max_centroid_similarity: float = 0.86,
        dedup_similarity_threshold: float = 0.86,
    ) -> AssetExtractionPipelineResult:
        """
        Esegue l'intera pipeline di estrazione degli asset (Step 1 -> Step 8).
        """
        result = AssetExtractionPipelineResult()

        # 1. Caricamento testi ed estrazione sostantivi con spaCy
        full_text = self.load_documents_text(extra_text=extra_text)
        if not full_text:
            return result

        candidate_terms = self.extract_candidate_nouns(full_text)
        result.candidate_terms_count = len(candidate_terms)
        if not candidate_terms:
            return result

        # 2. Calcolo degli embedding normalizzati L2
        embeddings = self.compute_embeddings(candidate_terms)

        # 3 & 4. K-Means dinamico con Silhouette Score ottimale
        clusters, best_sil = self.cluster_kmeans_optimal(
            terms=candidate_terms,
            embeddings=embeddings,
            k_min=k_min,
            k_max=k_max,
        )
        result.optimal_k = len(clusters)
        result.best_silhouette_score = best_sil

        # 5. Filtro per coesione semantica interna (cosine similarity)
        cohesive_clusters = self.filter_clusters_by_cohesion(
            clusters=clusters,
            min_cohesion=min_cohesion,
        )
        result.clusters_after_cohesion = len(cohesive_clusters)

        # Fallback: se la soglia di coesione ha scartato tutti i cluster, manteniamo i top cluster
        if not cohesive_clusters and clusters:
            cohesive_clusters = clusters
            result.clusters_after_cohesion = len(cohesive_clusters)

        # 6. Fusione dei cluster con centroidi troppo simili
        merged_clusters = self.merge_similar_clusters(
            clusters=cohesive_clusters,
            max_centroid_similarity=max_centroid_similarity,
        )
        result.clusters_after_merge = len(merged_clusters)

        # 7. Selezione dei cluster con asset da proteggere tramite LLM
        raw_assets, selected_evals = self.select_asset_clusters_with_llm(
            clusters=merged_clusters,
            general_context=general_context,
        )
        result.selected_clusters = selected_evals

        # 8. Rimozione dei duplicati (lessicale + semantica)
        result.final_assets = self.deduplicate_assets(
            assets=raw_assets,
            semantic_threshold=dedup_similarity_threshold,
        )

        # 9. Generazione delle descrizioni per ogni asset basandosi sul testo fornito
        result.described_assets = self.generate_asset_descriptions(
            assets=result.final_assets,
            full_text=full_text,
            general_context=general_context,
        )

        return result



