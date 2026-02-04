#!/usr/bin/env python3
"""
Test script for MetricsCalculator.calculate_accuracy robustness.

Tests various edge cases and boundary conditions to ensure the function
handles malformed data gracefully.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.meta.diagnoser.judge.metrics import MetricsCalculator


def test_normal_case():
    """Test normal case with valid data"""
    print("Test 1: Normal case")

    predictions = [
        {"affected_entities": ["ad_1", "ad_2"], "has_issue": True},
        {"affected_entities": ["ad_3"], "has_issue": True},
    ]

    ground_truth = [
        {"affected_entities": ["ad_1", "ad_3"], "has_issue": True},
        {"affected_entities": ["ad_4"], "has_issue": True},
    ]

    result = MetricsCalculator.calculate_accuracy(predictions, ground_truth)

    # TP: ad_1, ad_3 (2)
    # FP: ad_2 (1)
    # FN: ad_4 (1)
    # TN: 0

    print(f"  TP={result.true_positives}, FP={result.false_positives}, "
          f"FN={result.false_negatives}, TN={result.true_negatives}")
    print(f"  Precision={result.precision:.2%}, Recall={result.recall:.2%}, "
          f"F1={result.f1_score:.2%}")
    assert result.true_positives == 2
    assert result.false_positives == 1
    assert result.false_negatives == 1
    print("  ✓ Pass\n")


def test_empty_inputs():
    """Test with empty inputs"""
    print("Test 2: Empty inputs")

    # Both empty
    result = MetricsCalculator.calculate_accuracy([], [])
    assert result.precision == 0.0
    assert result.recall == 0.0
    assert result.f1_score == 0.0
    print("  ✓ Both empty: Pass")

    # Empty predictions, non-empty ground truth
    gt = [{"affected_entities": ["ad_1"], "has_issue": True}]
    result = MetricsCalculator.calculate_accuracy([], gt)
    assert result.false_negatives == 1
    assert result.recall == 0.0
    print("  ✓ Empty predictions: Pass")

    # Non-empty predictions, empty ground truth
    pred = [{"affected_entities": ["ad_1"], "has_issue": True}]
    result = MetricsCalculator.calculate_accuracy(pred, [])
    assert result.false_positives == 1
    assert result.precision == 0.0
    print("  ✓ Empty ground truth: Pass\n")


def test_none_inputs():
    """Test with None inputs"""
    print("Test 3: None inputs")

    result = MetricsCalculator.calculate_accuracy(None, None)
    assert result.precision == 0.0
    assert result.recall == 0.0
    print("  ✓ None inputs: Pass\n")


def test_wrong_type_inputs():
    """Test with wrong type inputs"""
    print("Test 4: Wrong type inputs")

    # String instead of list
    result = MetricsCalculator.calculate_accuracy("not_a_list", [])
    assert result.precision == 0.0
    print("  ✓ String predictions: Pass")

    # Dict instead of list
    result = MetricsCalculator.calculate_accuracy({}, [])
    assert result.precision == 0.0
    print("  ✓ Dict predictions: Pass")

    # Integer instead of list
    result = MetricsCalculator.calculate_accuracy(123, [])
    assert result.precision == 0.0
    print("  ✓ Integer predictions: Pass\n")


def test_malformed_entities():
    """Test with malformed affected_entities"""
    print("Test 5: Malformed entities")

    # String instead of list
    pred = [{"affected_entities": "ad_1"}]
    gt = [{"affected_entities": ["ad_1"], "has_issue": True}]
    result = MetricsCalculator.calculate_accuracy(pred, gt)
    assert result.true_positives == 1
    print("  ✓ String entity: Pass")

    # None entities
    pred = [{"affected_entities": None}]
    result = MetricsCalculator.calculate_accuracy(pred, gt)
    assert result.false_negatives == 1
    print("  ✓ None entities: Pass")

    # Empty string entity
    pred = [{"affected_entities": [""]}]
    result = MetricsCalculator.calculate_accuracy(pred, gt)
    assert result.false_negatives == 1
    print("  ✓ Empty string entity: Pass")

    # Mixed valid and invalid
    pred = [{"affected_entities": ["ad_1", None, "", "ad_2"]}]
    gt = [{"affected_entities": ["ad_1", "ad_2"], "has_issue": True}]
    result = MetricsCalculator.calculate_accuracy(pred, gt)
    assert result.true_positives == 2
    print("  ✓ Mixed valid/invalid: Pass\n")


def test_non_dict_items():
    """Test with non-dict items in lists"""
    print("Test 6: Non-dict items")

    # String in predictions list
    pred = ["not_a_dict", {"affected_entities": ["ad_1"]}]
    gt = [{"affected_entities": ["ad_1"], "has_issue": True}]
    result = MetricsCalculator.calculate_accuracy(pred, gt)
    assert result.true_positives == 1
    print("  ✓ String in predictions: Pass")

    # None in predictions list
    pred = [None, {"affected_entities": ["ad_1"]}]
    result = MetricsCalculator.calculate_accuracy(pred, gt)
    assert result.true_positives == 1
    print("  ✓ None in predictions: Pass\n")


def test_has_issue_variations():
    """Test different has_issue value formats"""
    print("Test 7: has_issue variations")

    pred = [{"affected_entities": ["ad_1", "ad_2"]}]

    # Boolean True
    gt = [{"affected_entities": ["ad_1"], "has_issue": True}]
    result = MetricsCalculator.calculate_accuracy(pred, gt)
    assert result.true_positives == 1
    print("  ✓ Boolean True: Pass")

    # Integer 1
    gt = [{"affected_entities": ["ad_1"], "has_issue": 1}]
    result = MetricsCalculator.calculate_accuracy(pred, gt)
    assert result.true_positives == 1
    print("  ✓ Integer 1: Pass")

    # String "true"
    gt = [{"affected_entities": ["ad_1"], "has_issue": "true"}]
    result = MetricsCalculator.calculate_accuracy(pred, gt)
    assert result.true_positives == 1
    print("  ✓ String 'true': Pass")

    # Missing has_issue (defaults to True)
    gt = [{"affected_entities": ["ad_1"]}]
    result = MetricsCalculator.calculate_accuracy(pred, gt)
    assert result.true_positives == 1
    print("  ✓ Missing has_issue: Pass")

    # Boolean False
    gt = [
        {"affected_entities": ["ad_1"], "has_issue": False},
        {"affected_entities": ["ad_2"], "has_issue": True},  # ad_2 真的有问题
    ]
    result = MetricsCalculator.calculate_accuracy(pred, gt)
    # ad_1: 预测有问题，实际没有 → FP
    # ad_2: 预测有问题，实际有 → TP
    assert result.true_positives == 1  # ad_2
    assert result.false_positives == 1  # ad_1
    print("  ✓ Boolean False: Pass\n")


def test_zero_division():
    """Test division by zero cases"""
    print("Test 8: Division by zero")

    # No predictions (TP=0, FP=0, FN=1)
    pred = []
    gt = [{"affected_entities": ["ad_1"], "has_issue": True}]
    result = MetricsCalculator.calculate_accuracy(pred, gt)
    assert result.precision == 0.0  # 0/(0+0) → 0.0
    assert result.recall == 0.0     # 0/(0+1) → 0.0
    print("  ✓ No predictions: Pass")

    # All correct (TP=3, FP=0, FN=0)
    pred = [
        {"affected_entities": ["ad_1"]},
        {"affected_entities": ["ad_2"]},
        {"affected_entities": ["ad_3"]},
    ]
    gt = [
        {"affected_entities": ["ad_1"], "has_issue": True},
        {"affected_entities": ["ad_2"], "has_issue": True},
        {"affected_entities": ["ad_3"], "has_issue": True},
    ]
    result = MetricsCalculator.calculate_accuracy(pred, gt)
    assert result.precision == 1.0
    assert result.recall == 1.0
    assert result.f1_score == 1.0
    print("  ✓ Perfect prediction: Pass\n")


def test_tn_calculation():
    """Test True Negative calculation"""
    print("Test 9: TN calculation")

    # GT has 3 entities: 2 with issues, 1 without
    # Prediction correctly identifies the 2 with issues
    pred = [
        {"affected_entities": ["ad_1"]},
        {"affected_entities": ["ad_2"]},
    ]
    gt = [
        {"affected_entities": ["ad_1"], "has_issue": True},
        {"affected_entities": ["ad_2"], "has_issue": True},
        {"affected_entities": ["ad_3"], "has_issue": False},  # No issue
    ]

    result = MetricsCalculator.calculate_accuracy(pred, gt)
    assert result.true_positives == 2  # ad_1, ad_2
    assert result.true_negatives == 1  # ad_3 (no issue, not predicted)
    assert result.false_positives == 0
    assert result.false_negatives == 0
    print(f"  ✓ TN={result.true_negatives}: Pass\n")


def main():
    print("=" * 60)
    print("MetricsCalculator.calculate_accuracy Robustness Tests")
    print("=" * 60)
    print()

    try:
        test_normal_case()
        test_empty_inputs()
        test_none_inputs()
        test_wrong_type_inputs()
        test_malformed_entities()
        test_non_dict_items()
        test_has_issue_variations()
        test_zero_division()
        test_tn_calculation()

        print("=" * 60)
        print("All tests passed! ✓")
        print("=" * 60)

    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
