from unittest.mock import MagicMock, patch
import numpy as np
import pytest
from models import (
    AssetExtractionOutput,
    ClusterAssetEvaluation,
    AssetDescriptionsOutput,
    ExtractedAssetRecord,
)
from core.asset_extractor import AssetExtractor, SemanticCluster


@patch("core.asset_extractor.OllamaEmbeddings")
def _make_extractor(tmp_path, mock_emb_cls):
    mock_emb = MagicMock()
    mock_emb_cls.return_value = mock_emb
    mock_factory = MagicMock()
    extractor = AssetExtractor(
        chain_factory=mock_factory, docs_dir=str(tmp_path / "docs")
    )
    return extractor, mock_emb, mock_factory


def test_document_management_and_text_loading(tmp_path):
    extractor, _, _ = _make_extractor(tmp_path)

    assert extractor.save_uploaded_documents([]) == 0
    saved = extractor.save_uploaded_documents(
        [
            {"content": "Billing database cluster.", "source_name": "arch.txt"},
            {"content": "Telemetry sensor node.", "source_name": "nodes"},
            {"content": "   ", "source_name": "empty.txt"},
        ]
    )
    assert saved == 2
    assert extractor.list_documents() == ["arch.txt", "nodes.txt"]

    combined = extractor.load_documents_text(extra_text="Synthetic platform context")
    assert "Synthetic platform context" in combined
    assert "Billing database cluster." in combined
    assert "Telemetry sensor node." in combined


def test_get_nlp_missing_model_raises(tmp_path):
    extractor, _, _ = _make_extractor(tmp_path)
    extractor.spacy_model = "nonexistent_spacy_model_xyz"
    with pytest.raises(RuntimeError, match="non trovato"):
        extractor._get_nlp()


def test_extract_candidate_nouns_spacy(tmp_path):
    extractor, _, _ = _make_extractor(tmp_path)

    assert extractor.extract_candidate_nouns("") == []

    text = (
        "The API gateway hosts user data and billing records. "
        "The telemetry node (VPN / DNS) connects to the core router - firewall."
    )
    candidates = extractor.extract_candidate_nouns(text)
    assert "api" in candidates
    assert "data" in candidates
    assert "telemetry node" in candidates
    assert "vpn" in candidates
    assert "dns" in candidates
    assert "hosts" not in candidates

    no_chunks = extractor.extract_candidate_nouns(text, include_noun_chunks=False)
    assert "data" in no_chunks


def test_compute_embeddings_and_clustering(tmp_path):
    extractor, mock_emb, _ = _make_extractor(tmp_path)

    empty_res = extractor.compute_embeddings([])
    assert empty_res.size == 0

    mock_emb.embed_documents.side_effect = [
        [[1.0, 0.0, 0.0], [0.9, 0.1, 0.0]],
        [[0.0, 1.0, 0.0]],
    ]
    embs = extractor.compute_embeddings(["a", "b", "c"], batch_size=2)
    assert embs.shape == (3, 3)
    np.testing.assert_allclose(np.linalg.norm(embs, axis=1), np.ones(3), atol=1e-5)

    assert extractor.cluster_kmeans_optimal([], np.empty((0, 0))) == ([], 0.0)
    small_clusters, sil = extractor.cluster_kmeans_optimal(["server", "db"], embs[:2])
    assert len(small_clusters) == 1
    assert sil == 1.0

    vec_cluster = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.98, 0.05, 0.0],
            [0.0, 1.0, 0.0],
            [0.05, 0.98, 0.0],
            [0.0, 0.0, 1.0],
            [0.05, 0.0, 0.98],
        ],
        dtype=np.float32,
    )
    terms = ["gateway", "proxy", "firewall", "router", "broker", "queue"]
    clusters, score = extractor.cluster_kmeans_optimal(
        terms, vec_cluster, k_min=2, k_max=4, silhouette_threshold=None
    )
    assert len(clusters) >= 2
    assert score > 0.5


def test_cohesion_filtering_and_merging(tmp_path):
    extractor, _, _ = _make_extractor(tmp_path)

    assert extractor.filter_clusters_by_cohesion([]) == []
    assert extractor._compute_internal_cohesion(np.array([[1.0, 0.0]])) == 0.0

    c1 = SemanticCluster(
        cluster_id=0,
        terms=["gateway", "proxy"],
        embeddings=np.array([[1.0, 0.0], [0.99, 0.1]], dtype=np.float32),
        centroid=np.array([1.0, 0.0], dtype=np.float32),
    )
    c_single = SemanticCluster(
        cluster_id=1,
        terms=["cdn"],
        embeddings=np.array([[0.98, 0.05]], dtype=np.float32),
        centroid=np.array([0.98, 0.05], dtype=np.float32),
    )
    c_low = SemanticCluster(
        cluster_id=2,
        terms=["desk", "cloud"],
        embeddings=np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        centroid=np.array([0.707, 0.707], dtype=np.float32),
    )

    filtered = extractor.filter_clusters_by_cohesion(
        [c1, c_single, c_low], min_cohesion=0.6
    )
    assert len(filtered) == 1
    assert "cdn" in filtered[0].terms

    assert extractor.merge_similar_clusters([filtered[0]]) == [filtered[0]]
    c_dup = SemanticCluster(
        cluster_id=1,
        terms=["edge router"],
        embeddings=np.array([[0.99, 0.02]], dtype=np.float32),
        centroid=np.array([0.99, 0.02], dtype=np.float32),
    )
    merged = extractor.merge_similar_clusters(
        [filtered[0], c_dup], max_centroid_similarity=0.8
    )
    assert len(merged) == 1
    assert "edge router" in merged[0].terms


def test_select_dedup_snippets_descriptions_and_pipeline(tmp_path):
    extractor, mock_emb, mock_factory = _make_extractor(tmp_path)

    assert extractor.select_asset_clusters_with_llm([]) == ([], [])

    ext_chain = MagicMock()
    ext_chain.invoke.return_value = AssetExtractionOutput(
        clusters_evaluation=[
            ClusterAssetEvaluation(
                cluster_id=0,
                contains_protectable_assets=True,
                extracted_assets=["Auth Gateway", "Auth gateways", "Firewall"],
            ),
            ClusterAssetEvaluation(
                cluster_id=1,
                contains_protectable_assets=False,
                extracted_assets=[],
            ),
        ]
    )
    mock_factory.get_asset_extraction_chain.return_value = ext_chain

    sample_clusters = [
        SemanticCluster(
            cluster_id=0,
            terms=["auth", "gateway", "firewall"],
            embeddings=np.eye(3, dtype=np.float32),
            centroid=np.array([1.0, 0.0, 0.0], dtype=np.float32),
        )
    ]
    raw_assets, evals = extractor.select_asset_clusters_with_llm(
        sample_clusters, "Synthetic Platform"
    )
    assert len(evals) == 1
    assert "Auth Gateway" in raw_assets

    assert extractor.deduplicate_assets([]) == []
    mock_emb.embed_documents.return_value = [
        [1.0, 0.0],
        [0.0, 1.0],
    ]
    deduped = extractor.deduplicate_assets(
        ["Auth Gateway", "Auth gateways", "Firewall"]
    )
    assert deduped == ["Auth Gateway", "Firewall"]

    assert (
        extractor._extract_relevant_snippets("", ["Firewall"])
        == "No document text provided."
    )
    snippets = extractor._extract_relevant_snippets(
        "Unrelated line.\nThe Firewall protects the Auth Gateway.", ["Firewall"]
    )
    assert "Firewall protects" in snippets
    fallback_snip = extractor._extract_relevant_snippets("Alpha beta gamma.", ["Zeta"])
    assert "Alpha beta gamma." in fallback_snip

    assert extractor.generate_asset_descriptions([], "text") == []
    desc_chain = MagicMock()
    desc_chain.invoke.return_value = AssetDescriptionsOutput(
        assets=[
            ExtractedAssetRecord(name="Auth Gateway", description="Handles user login.")
        ]
    )
    mock_factory.get_asset_description_chain.return_value = desc_chain

    described = extractor.generate_asset_descriptions(
        ["Auth Gateway", "Firewall"],
        "The Auth Gateway and Firewall.",
        "Synthetic Platform",
    )
    assert described[0].description == "Handles user login."
    assert "system architecture documentation" in described[1].description

    empty_res = extractor.extract_assets_pipeline(extra_text="")
    assert empty_res.candidate_terms_count == 0

    mock_emb.embed_documents.side_effect = lambda batch: [
        [1.0 if i % 2 == 0 else 0.0, 0.0 if i % 2 == 0 else 1.0, 0.1 * i]
        for i in range(len(batch))
    ]
    pipe_res = extractor.extract_assets_pipeline(
        extra_text="The Auth Gateway validates tokens and the Firewall filters external traffic."
    )
    assert pipe_res.candidate_terms_count > 0
    assert len(pipe_res.described_assets) >= 1
