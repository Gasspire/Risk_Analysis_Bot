def get_risk_indicator(risk_factor: int) -> tuple[str, str]:
    """Mappa il risk factor (1-25) a icona e etichetta testuale."""
    if risk_factor <= 3:
        return "🟢", "Very low"
    elif risk_factor <= 6:
        return "🟡", "Low"
    elif risk_factor <= 12:
        return "🟠", "Medium"
    elif risk_factor <= 16:
        return "🔴", "High"
    else:
        return "🚨", "Very high"