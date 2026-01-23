"""
Unit tests for helpers.py module.
Tests helper functions for rule-based allocation.
"""

from src.adset.utils.helpers import (
    DEFAULT_ADAPTIVE_TARGET_OPTIONS,
    DEFAULT_ADJUSTMENT_FACTOR,
    apply_adaptive_target_adj,
)


class TestHelpers:
    """Test suite for helper functions"""

    def test_adaptive_target_penalty(self):
        """Test adaptive target adjustment with penalty enabled"""
        value = 1.20  # 20% increase
        adaptive_target_roas = 3.0
        static_target_roas = 2.5

        result_value, modifications = apply_adaptive_target_adj(
            value,
            adaptive_target_roas,
            static_target_roas,
            adjustment_factor=DEFAULT_ADJUSTMENT_FACTOR,
            options=DEFAULT_ADAPTIVE_TARGET_OPTIONS,
        )

        assert isinstance(result_value, float)
        assert isinstance(modifications, list)
        # With penalty, should reduce adjustment if adaptive < static
        # Since adaptive (3.0) > static (2.5), penalty doesn't apply
        assert result_value >= value

    def test_adaptive_target_no_penalty(self):
        """Test adaptive target adjustment without penalty"""
        value = 1.20
        adaptive_target_roas = 3.0
        static_target_roas = 2.5

        result_value, modifications = apply_adaptive_target_adj(
            value,
            adaptive_target_roas,
            static_target_roas,
            adjustment_factor=0.3,
            options={"enable_penalty": False, "use_bonus_cap": True},
        )

        assert isinstance(result_value, float)
        assert isinstance(modifications, list)
        assert result_value > 0

    def test_adaptive_below_static(self):
        """Test when adaptive target ROAS is below static"""
        value = 1.20
        adaptive_target_roas = 2.0
        static_target_roas = 2.5

        result_with_penalty, _mods_with = apply_adaptive_target_adj(
            value,
            adaptive_target_roas,
            static_target_roas,
            adjustment_factor=0.3,
            options={"enable_penalty": True, "use_bonus_cap": True},
        )

        result_without_penalty, _mods_without = apply_adaptive_target_adj(
            value,
            adaptive_target_roas,
            static_target_roas,
            adjustment_factor=0.3,
            options={"enable_penalty": False, "use_bonus_cap": True},
        )

        # With penalty, adjustment should be reduced when
        # adaptive < static (if ratio < 0.9)
        # Since 2.0/2.5 = 0.8 < 0.9, penalty applies
        assert result_with_penalty <= result_without_penalty

    def test_adaptive_above_static(self):
        """Test when adaptive target ROAS is above static"""
        value = 1.20
        adaptive_target_roas = 3.5
        static_target_roas = 2.5

        result_value, modifications = apply_adaptive_target_adj(
            value,
            adaptive_target_roas,
            static_target_roas,
            adjustment_factor=DEFAULT_ADJUSTMENT_FACTOR,
            options=DEFAULT_ADAPTIVE_TARGET_OPTIONS,
        )

        # When adaptive > static, should allow larger adjustment
        assert isinstance(result_value, float)
        assert isinstance(modifications, list)
        assert result_value > 0

    def test_adaptive_target_bonus_cap(self):
        """Test adaptive target adjustment with bonus cap"""
        value = 2.0  # Large increase
        adaptive_target_roas = 4.0
        static_target_roas = 2.5

        result_with_cap, _mods_with = apply_adaptive_target_adj(
            value,
            adaptive_target_roas,
            static_target_roas,
            adjustment_factor=0.3,
            options={"enable_penalty": True, "use_bonus_cap": True},
        )

        result_without_cap, _mods_without = apply_adaptive_target_adj(
            value,
            adaptive_target_roas,
            static_target_roas,
            adjustment_factor=0.3,
            options={"enable_penalty": True, "use_bonus_cap": False},
        )

        # With cap, should limit the bonus
        assert result_with_cap <= result_without_cap

    def test_adaptive_none_values(self):
        """Test adaptive target adjustment with None values"""
        value = 1.20

        # Should handle None values gracefully
        result_value, modifications = apply_adaptive_target_adj(
            value,
            None,
            None,
            adjustment_factor=0.3,
            options={"enable_penalty": True, "use_bonus_cap": True},
        )

        assert isinstance(result_value, float)
        assert result_value == value  # Should return original when None
        assert isinstance(modifications, list)
        assert len(modifications) == 0

    def test_adaptive_target_neg_adj(self):
        """Test adaptive target adjustment with negative adjustment factor"""
        value = 0.8  # Decrease
        adaptive_target_roas = 3.0
        static_target_roas = 2.5

        result_value, modifications = apply_adaptive_target_adj(
            value,
            adaptive_target_roas,
            static_target_roas,
            adjustment_factor=DEFAULT_ADJUSTMENT_FACTOR,
            options=DEFAULT_ADAPTIVE_TARGET_OPTIONS,
        )

        # Should handle decreases
        assert isinstance(result_value, float)
        assert isinstance(modifications, list)
        assert result_value > 0

    def test_adaptive_none_options(self):
        """Test adaptive target adjustment with None options"""
        value = 1.20
        adaptive_target_roas = 3.0
        static_target_roas = 2.5

        result_value, modifications = apply_adaptive_target_adj(
            value,
            adaptive_target_roas,
            static_target_roas,
            adjustment_factor=0.3,
            options=None,  # None options should use defaults
        )

        assert isinstance(result_value, float)
        assert isinstance(modifications, list)

    def test_adaptive_zero_static(self):
        """Test adaptive target adjustment with zero static_target_roas"""
        value = 1.20
        adaptive_target_roas = 3.0
        static_target_roas = 0.0  # Zero or negative

        result_value, modifications = apply_adaptive_target_adj(
            value,
            adaptive_target_roas,
            static_target_roas,
            adjustment_factor=0.3,
            options=DEFAULT_ADAPTIVE_TARGET_OPTIONS,
        )

        # Should return original value when static_target_roas <= 0
        assert result_value == value
        assert len(modifications) == 0
