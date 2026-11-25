from unittest.mock import MagicMock, patch
from core.llm_chains import LLMChainFactory


@patch("core.llm_chains.ChatOllama")
def test_get_llm_options(mock_chat_ollama):
    factory = LLMChainFactory(
        model_name="llama3.1", temperature=0.2, base_url="http://localhost:11434"
    )

    factory.get_llm()
    mock_chat_ollama.assert_called_with(
        model="llama3.1",
        temperature=0.2,
        base_url="http://localhost:11434",
        num_ctx=4096,
        num_predict=800,
    )

    factory.get_llm(num_predict=1024, json_format=True, temperature=0.0, seed=42)
    mock_chat_ollama.assert_called_with(
        model="llama3.1",
        temperature=0.0,
        base_url="http://localhost:11434",
        num_ctx=4096,
        num_predict=1024,
        format="json",
        seed=42,
    )


@patch("core.llm_chains.ChatOllama")
def test_chain_builders(mock_chat_ollama):
    mock_llm = MagicMock()
    mock_llm.with_structured_output.return_value = MagicMock()
    mock_chat_ollama.return_value = mock_llm

    factory = LLMChainFactory()

    assert factory.get_classification_chain() is not None
    assert factory.get_likelihood_chain() is not None
    assert factory.get_impact_chain() is not None
    assert factory.get_synthesis_chain() is not None
    assert factory.get_intent_chain() is not None
    assert factory.get_rag_info_chain() is not None
    assert factory.get_asset_extraction_chain() is not None
    assert factory.get_asset_description_chain() is not None

    assert mock_llm.with_structured_output.call_count == 7
