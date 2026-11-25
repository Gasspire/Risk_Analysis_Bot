import pytest
from llm_handler.formatting import get_risk_indicator


@pytest.mark.parametrize(
    "risk_factor,expected",
    [
        (1, ("🟢", "Very low")),
        (3, ("🟢", "Very low")),
        (4, ("🟡", "Low")),
        (6, ("🟡", "Low")),
        (7, ("🟠", "Medium")),
        (12, ("🟠", "Medium")),
        (13, ("🔴", "High")),
        (16, ("🔴", "High")),
        (17, ("🚨", "Very high")),
        (25, ("🚨", "Very high")),
    ],
)
def test_get_risk_indicator_bands(risk_factor, expected):
    assert get_risk_indicator(risk_factor) == expected
