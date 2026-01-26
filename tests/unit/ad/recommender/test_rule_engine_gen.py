"""Tests for RuleEngine.generate_recommendations and pattern discovery."""

import pandas as pd
import pytest

from src.meta.ad.recommender.recommendations.rule_engine import RuleEngine


def test_generate_recommendations_returns_dict() -> None:
    df = pd.DataFrame({
        "roas_parsed": [0.5, 0.6, 0.7, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 4.0],
        "layout": ["left"] * 5 + ["center"] * 5,
        "dominant_color": ["red", "red", "blue", "blue", "blue"] * 2,
    })
    engine = RuleEngine()
    out = engine.generate_recommendations(df, discover_patterns=True)
    assert isinstance(out, dict)
    assert "recommendations" in out
    assert "creative_id" in out
    assert "confidence_scores" in out
    assert out["creative_id"] == "aggregated"
    assert isinstance(out["recommendations"], list)


def test_generate_recommendations_no_discover() -> None:
    df = pd.DataFrame({
        "roas_parsed": [1.0, 1.5],
        "layout": ["center", "center"],
    })
    engine = RuleEngine()
    out = engine.generate_recommendations(df, discover_patterns=False)
    assert "recommendations" in out
    assert out["creative_id"] == "aggregated"


def test_generate_recommendations_small_df_no_patterns() -> None:
    df = pd.DataFrame({
        "roas_parsed": [1.0, 1.1, 1.2],
        "layout": ["a", "b", "c"],
    })
    engine = RuleEngine()
    out = engine.generate_recommendations(df, discover_patterns=True)
    assert "recommendations" in out
    assert out["creative_id"] == "aggregated"
