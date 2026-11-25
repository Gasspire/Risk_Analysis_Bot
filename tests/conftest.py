from unittest.mock import MagicMock
import pytest
from models import SessionStateData, ThreatRecord


@pytest.fixture
def sample_threat_factory():
    """Factory fixture to build domain-agnostic ThreatRecord instances."""

    def _make(
        threat_id: str = "t-1",
        threat: str = "Unauthorized Access",
        affected_assets: str = "Core API",
        security_measures: str = "MFA, Firewall",
        likelihood: int = 2,
        impact: int = 3,
        risk_factor: int = 6,
    ) -> ThreatRecord:
        return ThreatRecord(
            id=threat_id,
            threat=threat,
            affected_assets=affected_assets,
            security_measures=security_measures,
            likelihood=likelihood,
            impact=impact,
            risk_factor=risk_factor,
        )

    return _make


@pytest.fixture
def sample_session_state(sample_threat_factory) -> SessionStateData:
    """Provides a self-contained SessionStateData for a generic synthetic environment."""
    return SessionStateData(
        general_context="Synthetic web platform handling user authentication and billing.",
        assessment_results=[
            sample_threat_factory(
                threat_id="t-1",
                threat="Credential Stuffing",
                affected_assets="Auth Gateway",
                security_measures="MFA, Rate Limiting",
                likelihood=3,
                impact=4,
                risk_factor=12,
            ),
            sample_threat_factory(
                threat_id="t-2",
                threat="Service Disruption",
                affected_assets="Billing Database",
                security_measures="Daily Backup",
                likelihood=2,
                impact=2,
                risk_factor=4,
            ),
        ],
    )


@pytest.fixture
def mock_rag_engine():
    rag = MagicMock()
    rag.retrieve_context.return_value = "Synthetic reference policy context."
    return rag
